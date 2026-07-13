"""Device-resident batched math environment core.

State is fixed-shape and device-resident:

    concrete/resolved: [B, A, F]
    public concrete/resolved: [B, F]
    market state: [B, F, ...]
    jobs/cumulative work/query memory: [B, A, ...]

The design intentionally pays for masked work over all F formulas to avoid
ragged CPU lists, data-dependent host packing, and per-env mechanism launches.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Sequence

import torch

from agoraforge.envs.math.obs import (
    AGENT_LAST_TURN,
    AGENT_SCALAR_DIM,
    AGENT_TIMESTEP,
    AGENT_VALUE,
    CUMULATIVE_CONJ,
    CUMULATIVE_PROOF,
    FORMULA_FEATURE_DIM,
    FORMULA_IS_TARGET,
    JOB_ACTIVE_TARGET,
    JOB_TAU_REMAINING,
    JOB_TYPE_CONJ,
    JOB_TYPE_PROVE,
    LOCAL_PROVEN,
    PUBLIC_CONCRETE,
    PUBLIC_PROVEN,
    QUERY_PROBABILITY,
    QUERY_PROB_TIME_KNOWN,
    QUERY_RELATED_IN_KNOWN,
    QUERY_RELATED_OUT_KNOWN,
    QUERY_TIME_RATIO,
    learned_formula_feat_dim,
)
from agoraforge.envs.interface import PolicyInputs
from agoraforge.envs.math.config import MathConfig
from agoraforge.envs.math.centralized import CentralizedPlanner
from agoraforge.envs.math.instance import build_static_instance
from agoraforge.envs.math.mechanism import (
    BatchedMarketMechanism,
    mechanism_dims_from_config,
    mechanism_from_config,
)
from agoraforge.envs.math.actions import MATH_ACTION_TYPE_TO_IDX

# Per-model job columns on the formula features. The joint centralized actor zeroes
# them on its env-level formula nodes because per-model job state rides the slot
# nodes and slot->formula edges instead.
_JOINT_FORMULA_ZERO_COLS = [
    JOB_ACTIVE_TARGET,
    JOB_TAU_REMAINING,
    JOB_TYPE_PROVE,
    JOB_TYPE_CONJ,
    CUMULATIVE_PROOF,
    CUMULATIVE_CONJ,
]


# Inductor fuses the densification of a query-weight axis (edge-mask + value mul +
# scatter-add) into a couple of Triton kernels instead of the ~5 tiny eager kernels
# it would otherwise launch. gather_col/gather_row are called 5x/step and are the
# dominant fusable cost in the env step (~half the env-step wall on an L40S). The
# densification is RNG-free; only the edge count E varies (per-epoch graph resample),
# so the E axis is marked dynamic and Dynamo keeps just two graphs (col/row).


def _query_axis_dense_eager(
    query_values: torch.Tensor,   # (B, A, E)
    match_idx: torch.Tensor,      # (E,)
    scatter_idx: torch.Tensor,    # (E,)
    target: torch.Tensor,         # (B, A)
    F: int,
) -> torch.Tensor:
    on = (match_idx.view(1, 1, -1) == target.unsqueeze(-1)).to(query_values.dtype)
    vals = query_values * on
    B, A, E = query_values.shape
    out = query_values.new_zeros((B, A, F))
    idx = scatter_idx.view(1, 1, -1).expand(B, A, E)
    return out.scatter_add(2, idx, vals)


_query_axis_dense_compiled = torch.compile(_query_axis_dense_eager, dynamic=False)


@dataclass
class TensorActions:
    """Fixed-shape action batch consumed by :class:`BatchedMathEnv`.

    ``math_formula`` stores global formula ids, not local visible slots.  The
    rollout adapter can map sampled local slots through the resident
    ``formula_ids`` tensor without leaving the device.
    """

    math_type: torch.Tensor              # [B,A], -1 for no-op
    math_formula: torch.Tensor           # [B,A], global phi or -1
    budget: torch.Tensor                 # [B,A], latent z; cfg maps to tau
    math_mode: torch.Tensor              # [B,A], 0 outward/out, 1 inward/in
    publish_statement: torch.Tensor      # [B,A,F] bool/int
    publish_proof: torch.Tensor          # [B,A,F] bool/int
    market_action: torch.Tensor          # [B,A,F,d_action]


@dataclass
class EnvStateTensors:
    timestep: torch.Tensor
    public_concrete: torch.Tensor
    public_resolved: torch.Tensor
    public_origin: torch.Tensor
    concrete: torch.Tensor
    resolved: torch.Tensor
    is_target: torch.Tensor
    target_theorem: torch.Tensor
    n_targets: torch.Tensor
    market_active: torch.Tensor
    s: torch.Tensor
    cumulative_proof: torch.Tensor
    cumulative_conj: torch.Tensor
    prev_reward: torch.Tensor
    target_resolution_step: torch.Tensor
    job_type: torch.Tensor
    job_target: torch.Tensor
    job_tau: torch.Tensor
    job_mode: torch.Tensor
    query_prob_known: torch.Tensor
    query_prob: torch.Tensor
    query_time: torch.Tensor
    query_related_in_known: torch.Tensor
    query_related_out_known: torch.Tensor
    context_theorems: torch.Tensor
    context_last_touched: torch.Tensor
    qre_src: torch.Tensor
    qre_dst: torch.Tensor
    qre_val: torch.Tensor
    qre_head: torch.Tensor
    graph_weights: torch.Tensor
    truth: torch.Tensor
    query_truth: torch.Tensor
    query_values: torch.Tensor


class BatchedMathEnv:
    """Tensor implementation of a homogeneous batch of mechanism-backed envs."""

    JOB_NONE = 0
    JOB_PROVE = 1
    JOB_CONJ = 2

    @classmethod
    def from_config(
        cls,
        cfg: MathConfig,
        *,
        batch_size: int,
        device: torch.device,
        seeds: Sequence[int] | torch.Tensor | None = None,
        profiler=None,
    ) -> "BatchedMathEnv":
        """Create a rollout batch directly on ``device`` without CPU env objects.

        This is the only constructor. It deliberately samples the latent rollout
        instance with tensor kernels instead of reproducing the Python
        Barabasi/Gibbs sampler bit-for-bit.

        ``centralized`` mode shares the same latent/job/query dynamics but has no
        market mechanism: knowledge is published to all agents each step and the
        reward is the team fitness.
        """
        if cfg.control_mode not in ("market", "bounty_only", "collaborative", "learned", "centralized"):
            raise ValueError("resident config construction targets mechanism-backed or centralized modes")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self = cls.__new__(cls)
        self.cfg = cfg
        self.centralized = cfg.control_mode == "centralized"
        self.device = torch.device(device)
        self.B = int(batch_size)
        # Candidate-population axis (set by ``batched_from_config``): K mechanisms
        # share one env batch, folded block-major into B = K * B_base. K == 1 is the
        # ordinary single-mechanism env and every batched branch below is inert.
        self.K = 1
        self.B_base = self.B
        self._profiler = profiler
        self.budget_latent_cap = float(cfg.budget_latent_cap)
        if self.budget_latent_cap <= 0.0:
            raise ValueError(f"budget_latent_cap must be > 0, got {self.budget_latent_cap}")
        # Per-step stochasticity generator. None draws from the global RNG (the
        # default single-env path). A batched env sets it so each
        # candidate block shares common random numbers and is reproducible.
        self._step_gen = None
        self.A = int(cfg.n_agents)
        self.F = int(cfg.F_size)
        self.N = int(cfg.num_theorems)
        self.mechanism = None if self.centralized else mechanism_from_config(cfg).to(self.device).eval()
        self.dims = None if self.centralized else self.mechanism.dims
        self.planner = CentralizedPlanner(self) if self.centralized else None
        self.dtype = torch.float32
        gen = self._make_generator(seeds)
        self._init_static_tensors_from_config(gen)
        self._init_mutable_tensors_from_config(gen)
        return self

    # Per-env tensors carry a leading B axis; the batched constructor tiles exactly
    # these across the K candidate blocks. Everything else (graph-shared edge
    # indices, dims, cfg) is candidate-independent and stays shared by reference.
    _PER_ENV_TENSORS = tuple(f.name for f in fields(EnvStateTensors))

    @classmethod
    def batched_from_config(
        cls,
        cfg: MathConfig,
        *,
        base_batch_size: int,
        thetas: Sequence[torch.Tensor],
        device: torch.device,
        seeds: Sequence[int] | torch.Tensor | None = None,
        step_seed: int = 0,
        profiler=None,
    ) -> "BatchedMathEnv":
        """Build one env that evaluates K candidate mechanisms in lockstep.

        Constructs a single ``base_batch_size`` env (the candidates share their env
        instance -- the search compares mechanisms on common random numbers), then
        tiles its per-env state block-major into ``B = K * base_batch_size`` and
        attaches a :class:`BatchedMarketMechanism` over ``thetas``. Candidate ``k``
        owns envs ``[k * base_batch_size : (k + 1) * base_batch_size]`` and applies
        ``thetas[k]``; the rollout steps all K at once. ``step_seed`` seeds the
        shared per-step stochasticity stream so each block reproduces what the same
        mechanism would see run alone (parity-tested in ``tests/test_batched_eval.py``).
        """
        K = len(thetas)
        if K < 1:
            raise ValueError("need at least one candidate theta")
        if cfg.control_mode != "learned" and K != 1:
            raise ValueError(
                "closed-form baseline population evaluation supports exactly one arm per batch"
            )
        base = cls.from_config(
            cfg,
            batch_size=base_batch_size,
            device=device,
            seeds=seeds,
            profiler=profiler,
        )
        B_base = base.B

        # A closed-form arm has no candidate theta to bank.  K=1 still uses this
        # constructor so it shares the population model/PPO/evaluation path with a
        # learned candidate, including the explicitly-seeded per-step RNG stream.
        if cfg.control_mode != "learned":
            base._step_gen = torch.Generator(
                device=base.device if base.device.type == "cuda" else torch.device("cpu")
            ).manual_seed(int(step_seed))
            return base

        self = cls.__new__(cls)
        for key, value in base.__dict__.items():
            if key in cls._PER_ENV_TENSORS:
                continue
            self.__dict__[key] = value
        for name in cls._PER_ENV_TENSORS:
            t = getattr(base, name)
            self.__dict__[name] = t.repeat((K,) + (1,) * (t.dim() - 1))

        self.K = K
        self.B_base = B_base
        self.B = K * B_base
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self._step_gen = torch.Generator(device=gen_device).manual_seed(int(step_seed))
        self.mechanism = BatchedMarketMechanism(
            thetas,
            mechanism_dims_from_config(cfg),
            layernorm=bool(cfg.learned_mechanism_layernorm),
        ).to(self.device)
        self.prev_reward = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        self._step_reward = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        return self

    def _step_uniform(self) -> torch.Tensor:
        """A (B, A) uniform for the per-step success draws.

        Batched: one (B_base, A) draw shared across the K candidate blocks (common
        random numbers -- candidate k sees exactly what it would alone). Single-env
        with a step generator: a plain (B, A) draw from it. Otherwise a
        global-RNG draw.
        """
        if self.K > 1:
            u = torch.rand((self.B_base, self.A), device=self.device, generator=self._step_gen)
            return u.repeat(self.K, 1)
        if self._step_gen is not None:
            return torch.rand((self.B, self.A), device=self.device, generator=self._step_gen)
        return torch.rand((self.B, self.A), device=self.device)

    def _sample_proposal(self, proposal_dist: torch.Tensor) -> torch.Tensor:
        """One categorical draw per (B, A) row over the F axis of ``proposal_dist``.

        The default single-env path uses ``torch.multinomial``. The
        batched / step-generator path samples by inverse-CDF over the shared
        ``_step_uniform`` draw so candidate blocks consume aligned random numbers and
        reproduce a single run (``torch.multinomial`` can't share a draw across rows
        whose distributions differ per candidate)."""
        if self.K == 1 and self._step_gen is None:
            return torch.multinomial(
                proposal_dist.reshape(self.B * self.A, self.F), 1
            ).reshape(self.B, self.A)
        probs = proposal_dist / proposal_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cdf = probs.cumsum(dim=-1)
        u = self._step_uniform().unsqueeze(-1)
        return torch.searchsorted(cdf, u, right=True).clamp(max=self.F - 1).squeeze(-1)

    # ------------------------------------------------------------------
    # Construction

    def _make_generator(self, seeds: Sequence[int] | torch.Tensor | None) -> torch.Generator:
        # A single CUDA generator keeps construction device-local. Per-env seed
        # identity is intentionally relaxed on this fast path; the mixed seed
        # preserves epoch/rank variation without host-side object construction.
        if seeds is None:
            seed = 0
        elif torch.is_tensor(seeds):
            seed = int(seeds.detach().cpu().to(torch.long).sum().item())
        else:
            seed = int(sum(int(s) for s in seeds))
        self._base_seed = seed % (2**63 - 1)
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=gen_device).manual_seed(self._base_seed)

    def _init_static_tensors_from_config(self, gen: torch.Generator) -> None:
        instance = build_static_instance(
            self.cfg,
            batch_size=self.B,
            n_agents=self.A,
            dtype=self.dtype,
            device=self.device,
            gen=gen,
        )
        self.neg_ids = instance.neg_ids
        self.theorem_ids = instance.theorem_ids
        self.formula_ids = instance.formula_ids
        self.truth = instance.truth
        self.graph_weights = instance.graph_weights
        self.query_truth = instance.query_truth
        self._qw_src = instance.qw_src
        self._qw_dst = instance.qw_dst
        self.query_values = instance.query_values

    # ------------------------------------------------------------------
    # Sparse per-agent query weights
    #
    # ``query_weights`` is ``graph_weights`` with per-agent log-normal noise; its
    # support is exactly the (sparse, O(F)-edge) graph support, so the dense
    # (B, A, F, F) materialization was almost all zeros and grew as O(B*A*F^2) =
    # O(B*N^3). We instead store the shared edge index ``(B, E)`` once and the
    # per-agent noised values ``(B, A, E)`` (O(B*A*F)), and densify the single
    # target column/row a query reads on demand.

    def _init_mutable_tensors_from_config(self, gen: torch.Generator) -> None:
        self.timestep = torch.zeros(self.B, dtype=torch.long, device=self.device)
        true_formula = torch.where(
            self.truth[:, : self.N].bool(),
            torch.arange(self.N, device=self.device).view(1, self.N),
            torch.arange(self.N, self.F, device=self.device).view(1, self.N),
        )
        theorem = torch.arange(self.N, device=self.device).view(1, self.N)
        n_resolved = max(1, min(self.N, int(round(float(self.cfg.prob_initially_resolved) * self.N))))
        target_capacity = self.N - n_resolved
        n_target = (
            0 if target_capacity <= 0
            else min(target_capacity, max(1, int(round(float(self.cfg.prob_initially_target) * self.N))))
        )
        initially_resolved = (theorem < n_resolved).expand(self.B, self.N)

        # Targets remain uniformly random, but are drawn only from the unresolved suffix.
        target_scores = torch.rand((self.B, self.N), device=self.device, generator=gen)
        target_scores[:, :n_resolved] = float("inf")
        initially_target = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)
        if n_target > 0:
            target_idx = torch.topk(target_scores, k=n_target, largest=False, dim=1).indices
            initially_target.scatter_(1, target_idx, True)

        self.public_concrete = torch.zeros((self.B, self.F), dtype=torch.bool, device=self.device)
        self.public_resolved = torch.zeros((self.B, self.F), dtype=torch.bool, device=self.device)
        self.public_resolved.scatter_(1, true_formula, initially_resolved)
        self.public_concrete.scatter_(1, true_formula, initially_resolved | initially_target)
        self.public_concrete |= self._gather_neg(self.public_concrete)
        self.public_origin = torch.zeros((self.B, self.F, self.A), dtype=torch.bool, device=self.device)

        self.concrete = self.public_concrete.unsqueeze(1).expand(self.B, self.A, self.F).clone()
        self.resolved = self.public_resolved.unsqueeze(1).expand(self.B, self.A, self.F).clone()
        self.is_target = torch.zeros((self.B, self.F), dtype=self.dtype, device=self.device)
        self.is_target.scatter_(1, true_formula, initially_target.to(self.dtype))
        self.is_target.scatter_(1, self.neg_ids[true_formula], initially_target.to(self.dtype))
        # The target-theorem mask (either sign) and target count are fixed for the
        # episode; cache them so the per-step reward and resolution recorder don't
        # rebuild them every one of the ~200 steps.
        self.target_theorem = self.is_target[:, : self.N].bool() | self.is_target[:, self.N :].bool()
        self.n_targets = self.target_theorem.sum(dim=1).to(torch.float32).clamp_min(1.0)

        open_public = self.public_concrete & ~self.public_resolved & ~self._gather_neg(self.public_resolved)
        if self.cfg.control_mode == "market" or self.cfg.control_mode == "learned":
            self.market_active = open_public.clone()
        else:
            self.market_active = torch.zeros_like(open_public)

        if not self.centralized:
            self.s = torch.zeros((self.B, self.F, self.A, self.dims.d_state), dtype=self.dtype, device=self.device)
        self.cumulative_proof = torch.zeros((self.B, self.A, self.F), dtype=self.dtype, device=self.device)
        self.cumulative_conj = torch.zeros((self.B, self.A, self.F), dtype=self.dtype, device=self.device)
        self.prev_reward = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        self._step_reward = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        if not self.centralized:
            self.prev_reward = self.economic_value()
        self._init_resolution_tracking()

        self.job_type = torch.zeros((self.B, self.A), dtype=torch.long, device=self.device)
        self.job_target = torch.full((self.B, self.A), -1, dtype=torch.long, device=self.device)
        self.job_tau = torch.zeros((self.B, self.A), dtype=torch.long, device=self.device)
        self.job_mode = torch.zeros((self.B, self.A), dtype=torch.long, device=self.device)

        self.query_prob_known = torch.zeros((self.B, self.A, self.F), dtype=torch.bool, device=self.device)
        self.query_prob = torch.zeros((self.B, self.A, self.F), dtype=self.dtype, device=self.device)
        self.query_time = torch.zeros((self.B, self.A, self.F), dtype=self.dtype, device=self.device)
        self.query_related_in_known = torch.zeros((self.B, self.A, self.F), dtype=torch.bool, device=self.device)
        self.query_related_out_known = torch.zeros((self.B, self.A, self.F), dtype=torch.bool, device=self.device)
        self._init_related_edge_buffer()
        self._init_formula_context(gen)

    def _context_theorem_cap(self) -> int:
        cap = int(getattr(self.cfg, "obs_formula_cap", 0))
        if cap <= 0 or cap >= self.F:
            return 0
        # The context is theorem-level, but observations are formula-level. Keep
        # both signs of every remembered theorem, so K=16 means 8 theorem pairs.
        return max(1, cap // 2)

    def _init_formula_context(self, gen: torch.Generator | None = None) -> None:
        """Initialize each agent's visible theorem context from concrete theorems."""
        cap = self._context_theorem_cap()
        self.context_theorem_cap = cap
        if cap <= 0:
            self.context_theorems = torch.empty((self.B, self.A, 0), dtype=torch.long, device=self.device)
            self.context_last_touched = torch.empty((self.B, self.A, 0), dtype=torch.long, device=self.device)
            return

        concrete_theorem = self.concrete[:, :, : self.N] | self.concrete[:, :, self.N :]
        keys = torch.rand((self.B, self.A, self.N), dtype=self.dtype, device=self.device, generator=gen)
        fallback = 2.0 + torch.arange(self.N, dtype=self.dtype, device=self.device).view(1, 1, self.N)
        keys = torch.where(concrete_theorem, keys, fallback)
        self.context_theorems = torch.topk(keys, k=min(cap, self.N), dim=-1, largest=False).indices
        if cap > self.N:
            pad = torch.arange(cap - self.N, device=self.device).view(1, 1, -1) % self.N
            pad = pad.expand(self.B, self.A, cap - self.N)
            self.context_theorems = torch.cat([self.context_theorems, pad], dim=-1)
        selected_concrete = concrete_theorem.gather(2, self.context_theorems)
        self.context_last_touched = torch.where(
            selected_concrete,
            torch.zeros_like(self.context_theorems),
            torch.full_like(self.context_theorems, -1),
        )

    def _touch_context_theorems(self, theorem: torch.Tensor, active: torch.Tensor) -> None:
        """Mark theorem ids as recently used, inserting them by LRU eviction."""
        if self.context_theorem_cap <= 0:
            return
        if theorem.dim() == 2:
            theorem = theorem.unsqueeze(-1)
        if active.dim() == 2:
            active = active.unsqueeze(-1)
        theorem = theorem.remainder(self.N).to(torch.long)
        active = active.bool()
        now = self.timestep.view(self.B, 1).expand(self.B, self.A)

        for j in range(theorem.shape[-1]):
            th = theorem[..., j]
            act = active[..., j]
            present = self.context_theorems == th.unsqueeze(-1)
            has_present = present.any(dim=-1)
            touch = act.unsqueeze(-1) & present
            self.context_last_touched = torch.where(
                touch,
                now.unsqueeze(-1),
                self.context_last_touched,
            )

            insert = act & ~has_present
            if not bool(insert.any()):
                continue
            evict = self.context_last_touched.argmin(dim=-1)
            evict_oh = torch.nn.functional.one_hot(evict, self.context_theorem_cap).to(torch.bool)
            replace = insert.unsqueeze(-1) & evict_oh
            self.context_theorems = torch.where(replace, th.unsqueeze(-1), self.context_theorems)
            self.context_last_touched = torch.where(
                replace,
                now.unsqueeze(-1),
                self.context_last_touched,
            )

    def _init_related_edge_buffer(self) -> None:
        """Bounded sparse store of discovered query-related edges.

        ``query_related_edges`` was a dense (B, A, F, F) accumulator (O(B*N^3)),
        but ``_run_query_related`` writes at most ``query_num_related`` edges per
        query and the obs only ever reads the K x K submatrix over the agent's
        current top-K. We keep a per-agent ring of (src, dst, val) entries sized
        to the most edges a rollout can discover (``max_timestep`` queries x
        ``query_num_related``) and densify the K x K view on demand.
        """
        cap = max(1, int(self.cfg.max_timestep) * max(1, int(self.cfg.query_num_related)))
        self._qre_cap = cap
        self.qre_src = torch.full((self.B, self.A, cap), -1, dtype=torch.long, device=self.device)
        self.qre_dst = torch.full((self.B, self.A, cap), -1, dtype=torch.long, device=self.device)
        self.qre_val = torch.zeros((self.B, self.A, cap), dtype=self.dtype, device=self.device)
        # Monotonic per-(b, a) write head; the ring slot is head % cap.
        self.qre_head = torch.zeros((self.B, self.A), dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Action conversion and masks

    def available_action_masks(self, sel_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        open_concrete = self.concrete & ~self.resolved & ~self._gather_neg(self.resolved)
        no_job = self.job_type == self.JOB_NONE
        can_math = no_job.unsqueeze(-1)
        prove = open_concrete & can_math
        conj = self.concrete & can_math
        query_prob = prove
        query_related = conj
        publish_statement = self.concrete & ~self.public_concrete.unsqueeze(1)
        publish_proof = self.resolved & ~self.public_resolved.unsqueeze(1)
        allow_market = self.cfg.control_mode in ("market", "learned")
        market_action = self.market_active.unsqueeze(1).expand(self.B, self.A, self.F) if allow_market else torch.zeros_like(prove)
        if sel_idx is not None:
            # Restrict every per-formula mask to the agent's observed top-K slots so
            # actions can only target a formula the policy actually sees.
            prove = prove.gather(2, sel_idx)
            conj = conj.gather(2, sel_idx)
            query_prob = query_prob.gather(2, sel_idx)
            query_related = query_related.gather(2, sel_idx)
            publish_statement = publish_statement.gather(2, sel_idx)
            publish_proof = publish_proof.gather(2, sel_idx)
            market_action = market_action.gather(2, sel_idx)
        math_type = torch.stack(
            [
                prove.any(dim=-1),
                conj.any(dim=-1),
                query_prob.any(dim=-1),
                query_related.any(dim=-1),
            ],
            dim=-1,
        )
        return {
            "math_type_mask": math_type.to(self.dtype),
            "prove": prove.to(self.dtype),
            "conj": conj.to(self.dtype),
            "query_prob_time": query_prob.to(self.dtype),
            "query_related": query_related.to(self.dtype),
            "publish_statement_mask": publish_statement.to(self.dtype),
            "publish_proof_mask": publish_proof.to(self.dtype),
            "market_action_mask": market_action.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Step

    def step(self, actions: TensorActions) -> torch.Tensor:
        self.timestep += 1
        if self.centralized:
            return self.planner.step(actions)
        valid_market_action = self.market_active.unsqueeze(1).unsqueeze(-1)
        market_action = actions.market_action * valid_market_action.to(self.dtype)

        with self._prof_section("step_publications"):
            proven_this_step, prover_onehot = self._process_publications(actions)
        with self._prof_section("step_advance_markets"):
            self._advance_markets(market_action, proven_this_step, prover_onehot)
        with self._prof_section("step_math_actions"):
            self._process_math_actions(actions)
        with self._prof_section("step_rewards"):
            rewards = self._compute_rewards()
        with self._prof_section("step_record_resolutions"):
            self._record_target_resolutions()
        return rewards.unsqueeze(-1)

    def _prof_section(self, name: str):
        prof = getattr(self, "_profiler", None)
        return prof.section(name) if prof is not None else nullcontext()

    def _process_publications(
        self, actions: TensorActions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        old_public_concrete = self.public_concrete.clone()
        pub_stmt = actions.publish_statement & self.concrete & ~self.public_concrete.unsqueeze(1)
        stmt_any = pub_stmt.any(dim=1)
        stmt_pair = stmt_any | self._gather_neg(stmt_any)
        self.public_concrete |= stmt_pair
        self.concrete |= stmt_pair.unsqueeze(1)

        pub_proof = actions.publish_proof & self.resolved & ~self.public_resolved.unsqueeze(1)
        proof_any = pub_proof.any(dim=1)
        # Agent-order tie break: first publisher gets the event signal.
        ranks = torch.arange(self.A, device=self.device).view(1, self.A, 1)
        ranked_proof = torch.where(pub_proof, ranks, torch.full_like(ranks, self.A))
        prover = ranked_proof.min(dim=1).values
        newly_public = proof_any & ~self.public_resolved
        proven_this_step = newly_public
        valid = newly_public & (prover < self.A)
        prover_onehot = (
            torch.nn.functional.one_hot(prover.clamp(max=self.A - 1), self.A).to(torch.bool)
            & valid.unsqueeze(-1)
        )

        proof_pair = newly_public | self._gather_neg(newly_public)
        publicize_actions = pub_stmt | (pub_proof & ~old_public_concrete.unsqueeze(1))
        publicize_any = publicize_actions.any(dim=1)
        ranked_public = torch.where(publicize_actions, ranks, torch.full_like(ranks, self.A))
        publicizer = ranked_public.min(dim=1).values
        newly_concrete_pair = (publicize_any | self._gather_neg(publicize_any)) & ~old_public_concrete
        valid_publicizer = publicize_any & (publicizer < self.A)
        signed_public_origin = (
            torch.nn.functional.one_hot(publicizer.clamp(max=self.A - 1), self.A).to(torch.bool)
            & valid_publicizer.unsqueeze(-1)
        )
        public_origin_new = (
            signed_public_origin | signed_public_origin.index_select(1, self.neg_ids)
        ) & newly_concrete_pair.unsqueeze(-1)
        self.public_origin |= public_origin_new

        self.public_concrete |= proof_pair
        self.public_resolved |= newly_public
        self.concrete |= proof_pair.unsqueeze(1)
        self.resolved |= newly_public.unsqueeze(1)

        new_open = self.public_concrete & ~self.public_resolved & ~self._gather_neg(self.public_resolved)
        self.market_active |= new_open
        self.market_active |= proven_this_step
        return proven_this_step, prover_onehot

    def _advance_markets(
        self,
        market_action: torch.Tensor,
        proven_this_step: torch.Tensor,
        prover_onehot: torch.Tensor,
    ) -> None:
        neg_proven = self._gather_neg(proven_this_step)
        false_resolved = neg_proven & self.market_active
        theorem_resolved = proven_this_step | false_resolved
        active = self.market_active | theorem_resolved

        last_step = self.timestep >= self.cfg.max_timestep

        # Per-agent env update e[b,f,j] = [theorem_resolved_by_this_agent,
        # proven_true, theorem_resolved, is_last_step, is_target,
        # originally_made_public_by_this_agent]. The first/last bits are private;
        # the public-origin bit is persistent, while the first bit is a one-step
        # resolution event. The other four are per-formula and broadcast across agents.
        e = torch.zeros((self.B, self.F, self.A, self.dims.d_e), dtype=self.dtype, device=self.device)
        e[..., 0] = prover_onehot.to(self.dtype)
        e[..., 1] = proven_this_step.unsqueeze(-1).to(self.dtype)
        e[..., 2] = theorem_resolved.unsqueeze(-1).to(self.dtype)
        e[..., 3] = last_step.to(self.dtype).view(self.B, 1, 1)
        e[..., 4] = self.is_target.unsqueeze(-1)
        e[..., 5] = self.public_origin.to(self.dtype)

        ds, da, de = self.dims.d_state, self.dims.d_action, self.dims.d_e
        action_in = market_action.permute(0, 2, 1, 3)  # (B, A, F, d_action) -> (B, F, A, d_action)
        learned = self.cfg.control_mode == "learned"
        with torch.no_grad():
            # Fold the candidate axis out: candidate k applies its own mechanism to
            # its block of B_base * F market states in one batched launch (K=1 for
            # the ordinary single-mechanism env; baselines are shape-generic).
            M = self.B_base * self.F
            s_k = self.s.reshape(self.K, M, self.A, ds)
            a_k = action_in.reshape(self.K, M, self.A, da)
            e_k = e.reshape(self.K, M, self.A, de)
            if learned:
                s_next, r_contrib = self.mechanism.step(s_k, a_k, e_k)
            else:
                s_next = self.mechanism(s_k, a_k, e_k)
        active_s = active.unsqueeze(-1).unsqueeze(-1)
        self.s = torch.where(active_s, s_next.reshape_as(self.s), self.s)
        if learned:
            self._step_reward = (
                r_contrib.reshape(self.B, self.F, self.A) * active.unsqueeze(-1)
            ).sum(dim=1)

        resolved_pair = self.public_resolved | self._gather_neg(self.public_resolved)
        normal_final = self.timestep >= self.cfg.max_timestep
        final = normal_final.view(self.B, 1)
        self.market_active &= ~resolved_pair
        self.market_active &= ~final

    def _process_math_actions(self, actions: TensorActions) -> None:
        with self._prof_section("math_start_jobs"):
            no_job = self.job_type == self.JOB_NONE
            phi = actions.math_formula.clamp(min=0)
            phi_valid = actions.math_formula >= 0
            concrete_target = self._gather_baf(self.concrete, phi)
            resolved_target = self._gather_baf(self.resolved, phi)
            open_target = concrete_target & ~resolved_target
            open_target &= ~self._gather_baf(self._gather_neg(self.resolved), phi)
            prove_idx = MATH_ACTION_TYPE_TO_IDX["prove"]
            conj_idx = MATH_ACTION_TYPE_TO_IDX["conj"]
            qpt_idx = MATH_ACTION_TYPE_TO_IDX["query_prob_time"]
            qrel_idx = MATH_ACTION_TYPE_TO_IDX["query_related"]

            start_prove = no_job & phi_valid & (actions.math_type == prove_idx) & open_target
            start_conj = no_job & phi_valid & (actions.math_type == conj_idx) & concrete_target
            tau = self._budget_from_action_latent(actions.budget)
            self.job_type = torch.where(start_prove, torch.full_like(self.job_type, self.JOB_PROVE), self.job_type)
            self.job_type = torch.where(start_conj, torch.full_like(self.job_type, self.JOB_CONJ), self.job_type)
            self.job_target = torch.where(start_prove | start_conj, phi, self.job_target)
            self.job_tau = torch.where(start_prove | start_conj, tau, self.job_tau)
            self.job_mode = torch.where(start_conj, actions.math_mode, self.job_mode)

        with self._prof_section("math_query_prob"):
            query_prob = no_job & phi_valid & (actions.math_type == qpt_idx) & open_target
            self._touch_context_theorems(phi, query_prob)
            self._run_query_prob_time(query_prob, phi)
        with self._prof_section("math_query_related"):
            query_related = no_job & phi_valid & (actions.math_type == qrel_idx) & concrete_target
            self._touch_context_theorems(phi, query_related)
            self._run_query_related(query_related, phi, actions.math_mode)

        with self._prof_section("math_advance_jobs"):
            self._advance_jobs()

    def _advance_jobs(self) -> None:
        active_prove = self.job_type == self.JOB_PROVE
        active_conj = self.job_type == self.JOB_CONJ
        target = self.job_target.clamp(min=0)
        with self._prof_section("jobs_touch_context"):
            self._touch_context_theorems(target, active_prove | active_conj)
        with self._prof_section("jobs_proof"):
            self._advance_proof_jobs(active_prove, target)
        with self._prof_section("jobs_conj"):
            self._advance_conj_jobs(active_conj, target)

    def _advance_proof_jobs(self, active: torch.Tensor, target: torch.Tensor) -> None:
        target_oh = torch.nn.functional.one_hot(target, self.F).to(self.dtype)  # (B, A, F)
        # support[b,a] = sum_f resolved[b,a,f] * graph_weights[b, f, target[b,a]].
        # target_oh is one-hot, so the einsum baf,bfg,bag->ba is just the target's
        # graph-weight column gathered and dotted with resolved -- O(F) instead of
        # the O(F^2) contraction.
        gw_col = self.graph_weights.gather(
            2, target.unsqueeze(1).expand(self.B, self.F, self.A)).transpose(1, 2)  # (B, A, F)
        support = (self.resolved.to(self.dtype) * gw_col).sum(dim=-1)
        truth = self._gather_baf(self.truth.unsqueeze(1).expand(self.B, self.A, self.F), target)
        p = truth * (1.0 - (1.0 + support).pow(-float(self.cfg.rho)))
        success = (self._step_uniform() < p) & active
        add = target_oh.to(torch.bool) & success.unsqueeze(-1)
        add_pair = add | self._gather_neg(add)
        self.resolved |= add
        self.concrete |= add_pair
        inc = target_oh * active.unsqueeze(-1).to(self.dtype)
        self.cumulative_proof += inc
        failed = active & ~success
        self.job_tau = torch.where(failed, self.job_tau - 1, self.job_tau)
        done = success | (failed & (self.job_tau <= 0))
        self.job_type = torch.where(done, torch.zeros_like(self.job_type), self.job_type)
        self.job_target = torch.where(done, torch.full_like(self.job_target, -1), self.job_target)

    def _advance_conj_jobs(self, active: torch.Tensor, target: torch.Tensor) -> None:
        weights_in = self._gather_query_col(target)
        weights_out = self._gather_query_row(target)
        use_in = self.job_mode == 1
        weights = torch.where(use_in.unsqueeze(-1), weights_in, weights_out)
        ghost = ~self.concrete
        positive = torch.clamp(weights, min=0.0) * ghost.to(self.dtype)
        mass = positive.sum(dim=-1)
        p = 1.0 - (1.0 + mass).pow(-float(self.cfg.eta))
        success = (self._step_uniform() < p) & active & (ghost.any(dim=-1))
        fallback = torch.zeros_like(positive)
        fallback[..., 0] = 1.0
        proposal_dist = torch.where(
            positive.sum(dim=-1, keepdim=True) > 0,
            positive,
            torch.where(ghost.any(dim=-1, keepdim=True), ghost.to(self.dtype), fallback),
        )
        proposal = self._sample_proposal(proposal_dist)
        add = torch.nn.functional.one_hot(proposal, self.F).to(torch.bool) & success.unsqueeze(-1)
        self.concrete |= add | self._gather_neg(add)
        self._touch_context_theorems(proposal, success)
        edge_val = weights.gather(2, proposal.unsqueeze(-1))
        self._append_conjecture_related_edges(target, proposal, edge_val.squeeze(-1), success, use_in)
        inc = torch.nn.functional.one_hot(target, self.F).to(self.dtype) * active.unsqueeze(-1).to(self.dtype)
        self.cumulative_conj += inc
        self.job_tau = torch.where(active & ~success, self.job_tau - 1, self.job_tau)
        done = success | (active & ~success & (self.job_tau <= 0))
        self.job_type = torch.where(done, torch.zeros_like(self.job_type), self.job_type)
        self.job_target = torch.where(done, torch.full_like(self.job_target, -1), self.job_target)

    # ------------------------------------------------------------------
    # Queries and observations

    def _run_query_prob_time(self, active: torch.Tensor, target: torch.Tensor) -> None:
        p_truth = self._gather_baf(self.query_truth, target)
        resolved_t = self._gather_baf(self.resolved, target)
        neg_resolved_t = self._gather_baf(self._gather_neg(self.resolved), target)
        prob = torch.where(resolved_t, torch.ones_like(p_truth), torch.where(neg_resolved_t, torch.zeros_like(p_truth), p_truth))
        weights = self._gather_query_col(target)
        support = (weights * self.resolved.to(self.dtype)).sum(dim=-1)
        p_step = prob * (1.0 - (1.0 + support).pow(-float(self.cfg.rho)))
        expected = self._truncated_geometric_mean(p_step)
        self.query_prob_known.scatter_(2, target.unsqueeze(-1), active.unsqueeze(-1))
        self.query_prob.scatter_(2, target.unsqueeze(-1), torch.where(active, prob, torch.zeros_like(prob)).unsqueeze(-1))
        self.query_time.scatter_(2, target.unsqueeze(-1), torch.where(active, expected, torch.zeros_like(expected)).unsqueeze(-1))

    def _run_query_related(self, active: torch.Tensor, target: torch.Tensor, mode: torch.Tensor) -> None:
        weights_in = self._gather_query_col(target)
        weights_out = self._gather_query_row(target)
        use_in = mode == 1
        weights = torch.where(use_in.unsqueeze(-1), weights_in, weights_out)
        concrete = self.concrete.to(self.dtype)
        weights = weights * concrete
        weights.scatter_(2, target.unsqueeze(-1), 0.0)
        k = int(self.cfg.query_num_related)
        if k <= 0:
            return
        kk = min(k, self.F)
        vals, idx = torch.topk(weights, k=kk, dim=-1)
        valid = active.unsqueeze(-1) & (vals > 0)
        tgt = target.unsqueeze(-1).expand_as(idx)
        self.query_related_in_known.scatter_(2, target.unsqueeze(-1), (active & use_in).unsqueeze(-1))
        self.query_related_out_known.scatter_(2, target.unsqueeze(-1), (active & ~use_in).unsqueeze(-1))
        src = torch.where(use_in.unsqueeze(-1), idx, tgt)
        dst = torch.where(use_in.unsqueeze(-1), tgt, idx)
        self._append_related_edges(src, dst, vals, valid)
        self._touch_context_theorems(idx, valid)

    def _append_conjecture_related_edges(
        self,
        target: torch.Tensor,
        proposal: torch.Tensor,
        vals: torch.Tensor,
        success: torch.Tensor,
        use_in: torch.Tensor,
    ) -> None:
        src = torch.where(use_in, proposal, target).unsqueeze(-1)
        dst = torch.where(use_in, target, proposal).unsqueeze(-1)
        valid = (success & (vals > 0)).unsqueeze(-1)
        self._append_related_edges(src, dst, vals.unsqueeze(-1), valid)

    def _append_related_edges(
        self, src: torch.Tensor, dst: torch.Tensor, vals: torch.Tensor, valid: torch.Tensor
    ) -> None:
        """Append up to ``kk`` discovered edges per (b, a) into the ring buffer.

        Invalid slots are stored with src/dst = -1 so the obs densify skips them;
        the per-(b, a) write head advances by the count of *appended* slots
        (including invalid ones), wrapping at capacity so re-queries overwrite the
        oldest entries (last write wins, matching the dense overwrite semantics).
        """
        kk = src.shape[-1]
        cap = self._qre_cap
        store_src = torch.where(valid, src, torch.full_like(src, -1))
        store_dst = torch.where(valid, dst, torch.full_like(dst, -1))
        # Per-(b, a) ring positions for this step's kk slots.
        offsets = torch.arange(kk, device=self.device).view(1, 1, kk)
        pos = (self.qre_head.unsqueeze(-1) + offsets) % cap
        self.qre_src.scatter_(2, pos, store_src)
        self.qre_dst.scatter_(2, pos, store_dst)
        self.qre_val.scatter_(2, pos, torch.where(valid, vals, torch.zeros_like(vals)))
        self.qre_head = self.qre_head + kk

    def select_topk_formulas(self) -> torch.Tensor | None:
        """Per-agent theorem-context formula ids, or None when uncapped.

        Each agent owns a fixed-size theorem context initialized as a
        random subset of its concrete theorems. Actions touch active job/query
        targets; successful conjectures and query-related references insert new
        theorem ids by evicting the least-recently touched context theorem.

        Observations are still formula-level, so every remembered theorem emits
        both signed formulas. For the current scale setting, ``obs_formula_cap=16``
        means 8 theorem-context slots and 16 formula rows.

        The centralized planner is always uncapped (full visibility is its
        defining property), so the cap is ignored in that mode.
        """
        if self.centralized:
            return None
        cap = int(getattr(self.cfg, "obs_formula_cap", 0))
        if cap <= 0 or cap >= self.F:
            return None
        pos = self.context_theorems
        neg = pos + self.N
        sel = torch.stack([pos, neg], dim=-1).reshape(self.B, self.A, -1)
        if sel.shape[-1] > cap:
            sel = sel[..., :cap]
        return sel.sort(dim=-1).values

    def policy_inputs(self) -> PolicyInputs:
        """One step's model inputs (the :class:`BatchedEnv` contract).

        ``ctx`` carries the top-k formula selection the policy module scatters
        sampled actions back through (None when uncapped or centralized)."""
        sel_idx = self.select_topk_formulas()
        return PolicyInputs(
            actor_obs=self.actor_obs(sel_idx),
            masks=self.available_action_masks(sel_idx),
            ctx=sel_idx,
        )

    def actor_obs(self, sel_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.centralized:
            # One joint graph per env (B rows): the actor decodes every model slot
            # in a single forward. Centralized is always uncapped (sel_idx is None).
            obs = self._obs(sel_idx=None, joint=True)
            obs.update(self._slot_obs())
            return obs
        return self._obs(sel_idx=sel_idx)

    def _slot_obs(self) -> dict[str, torch.Tensor]:
        """Per-model-slot nodes for the centralized actor: one node per math model.

        Slot node features carry each model's job state (active, tau-remaining,
        type); ``slot_formula_edges`` carry the model's active-job target and its
        cumulative proof/conjecture work per formula. The actor decodes an action
        off every slot in one forward, so there is no focal marker and the env's A
        slots are emitted once per env. Centralized is uncapped, so edges span the
        full formula axis.
        """
        scale = max(float(self.cfg.max_timestep), 1.0)
        active = (self.job_type != self.JOB_NONE).to(self.dtype)
        slot_features = torch.stack(
            [
                active,
                active * self.job_tau.to(self.dtype) / scale,
                (self.job_type == self.JOB_PROVE).to(self.dtype),
                (self.job_type == self.JOB_CONJ).to(self.dtype),
            ],
            dim=-1,
        )  # (B, A, SLOT_FEATURE_DIM)

        target = self.job_target.clamp(min=0)
        job_target = torch.nn.functional.one_hot(target, self.F).to(self.dtype) * active.unsqueeze(-1)
        slot_edges = torch.stack(
            [job_target, self.cumulative_proof / scale, self.cumulative_conj / scale], dim=-1
        )  # (B, A, F, SLOT_EDGE_DIM)
        return {
            "slot_features": slot_features,
            "slot_formula_edges": slot_edges,
        }

    def _densify_related_edges(self, sel_idx: torch.Tensor | None) -> torch.Tensor:
        """Materialize the discovered query-related edges as a dense view.

        Uncapped: full (B, A, F, F). Capped: (B, A, K, K) over the agent's top-K,
        built directly from the sparse ring so the dense F x F never materializes.
        Stored edges with src/dst = -1 (empty/invalid slots) are dropped, and
        endpoints outside the K-window are dropped by the inverse slot map.
        """
        W = self.F if sel_idx is None else sel_idx.shape[-1]
        if sel_idx is None:
            r = self.qre_src.clamp(min=0)
            c = self.qre_dst.clamp(min=0)
            keep = (self.qre_src >= 0) & (self.qre_dst >= 0)
        else:
            # Inverse map: formula id -> top-K slot (-1 if outside the window).
            inv = torch.full((self.B, self.A, self.F), -1, dtype=torch.long, device=self.device)
            slot = torch.arange(W, device=self.device).view(1, 1, W).expand(self.B, self.A, W)
            inv.scatter_(2, sel_idx, slot)
            r = inv.gather(2, self.qre_src.clamp(min=0))
            c = inv.gather(2, self.qre_dst.clamp(min=0))
            keep = (self.qre_src >= 0) & (self.qre_dst >= 0) & (r >= 0) & (c >= 0)
        # Dropped entries scatter into a trailing sentinel slot then get sliced off,
        # so they never clobber a real (0, 0) edge.
        flat = torch.where(keep, r.clamp(min=0) * W + c.clamp(min=0), torch.full_like(r, W * W))
        out = torch.zeros((self.B, self.A, W * W + 1), dtype=self.dtype, device=self.device)
        out.scatter_(2, flat, self.qre_val)
        dense = out[..., : W * W].reshape(self.B, self.A, W, W)
        if self.centralized:
            # Shared visibility: every agent sees the union of discovered edges
            # (weights are non-negative, so max over agents is the union).
            dense = dense.max(dim=1, keepdim=True).values.expand_as(dense)
        return dense

    def _obs(self, *, sel_idx: torch.Tensor | None = None, joint: bool = False) -> dict[str, torch.Tensor]:
        BA = self.B * self.A
        formula_features = torch.zeros((self.B, self.A, self.F, FORMULA_FEATURE_DIM), dtype=self.dtype, device=self.device)
        formula_features[..., LOCAL_PROVEN] = self.resolved.to(self.dtype)
        formula_features[..., PUBLIC_CONCRETE] = self.public_concrete.unsqueeze(1).to(self.dtype)
        formula_features[..., PUBLIC_PROVEN] = self.public_resolved.unsqueeze(1).to(self.dtype)
        target = self.job_target.clamp(min=0)
        active_job_target = torch.nn.functional.one_hot(target, self.F).to(self.dtype) * (self.job_type != self.JOB_NONE).unsqueeze(-1).to(self.dtype)
        formula_features[..., JOB_ACTIVE_TARGET] = active_job_target
        formula_features[..., JOB_TAU_REMAINING] = active_job_target * (
            self.job_tau.to(self.dtype) / max(float(self.cfg.max_timestep), 1.0)
        ).unsqueeze(-1)
        formula_features[..., JOB_TYPE_PROVE] = active_job_target * (self.job_type == self.JOB_PROVE).unsqueeze(-1).to(self.dtype)
        formula_features[..., JOB_TYPE_CONJ] = active_job_target * (self.job_type == self.JOB_CONJ).unsqueeze(-1).to(self.dtype)
        formula_features[..., CUMULATIVE_PROOF] = self.cumulative_proof / max(float(self.cfg.max_timestep), 1.0)
        formula_features[..., CUMULATIVE_CONJ] = self.cumulative_conj / max(float(self.cfg.max_timestep), 1.0)
        formula_features[..., QUERY_PROB_TIME_KNOWN] = self.query_prob_known.to(self.dtype)
        formula_features[..., QUERY_PROBABILITY] = self.query_prob
        formula_features[..., QUERY_TIME_RATIO] = self.query_time / max(float(self.cfg.horizon_H), 1.0)
        formula_features[..., QUERY_RELATED_IN_KNOWN] = self.query_related_in_known.to(self.dtype)
        formula_features[..., QUERY_RELATED_OUT_KNOWN] = self.query_related_out_known.to(self.dtype)
        formula_features[..., FORMULA_IS_TARGET] = self.is_target.unsqueeze(1)

        if self.cfg.control_mode in ("market", "bounty_only", "collaborative", "learned"):
            # Mechanism modes append the agent's own per-formula state s after the
            # shared columns (each agent sees its own state, public info included).
            s = self.s.permute(0, 2, 1, 3)
            formula_features = torch.cat([formula_features, s], dim=-1)

        expected = (
            FORMULA_FEATURE_DIM if self.centralized
            else learned_formula_feat_dim(self.dims.d_state)
        )
        if formula_features.shape[-1] != expected:
            raise RuntimeError(f"unexpected formula feature width {formula_features.shape[-1]} != {expected}")

        agent_scalars = torch.zeros((self.B, self.A, AGENT_SCALAR_DIM), dtype=self.dtype, device=self.device)
        value = self.economic_value()
        agent_scalars[..., AGENT_VALUE] = value / max(float(self.cfg.initial_cash), 1.0)
        agent_scalars[..., AGENT_TIMESTEP] = self.timestep.to(self.dtype).view(self.B, 1) / max(float(self.cfg.max_timestep), 1.0)
        agent_scalars[..., AGENT_LAST_TURN] = (self.timestep >= self.cfg.max_timestep - 1).to(self.dtype).view(self.B, 1)

        mask = self.concrete
        ids = self.formula_ids.view(1, 1, self.F).expand(self.B, self.A, self.F)
        neg_ids = self.neg_ids.view(1, 1, self.F).expand(self.B, self.A, self.F)

        if sel_idx is not None:
            K = sel_idx.shape[-1]
            feat_idx = sel_idx.unsqueeze(-1).expand(self.B, self.A, K, formula_features.shape[-1])
            formula_features = formula_features.gather(2, feat_idx)
            mask = mask.gather(2, sel_idx)
            ids = ids.gather(2, sel_idx)
            # neg_formula_ids must stay aligned with the gathered id axis, so the
            # neg edge (id == neg_id) only fires when both members survive the cap.
            neg_ids = neg_ids.gather(2, sel_idx)
            query_edges = self._densify_related_edges(sel_idx)
            F_out = K
        else:
            query_edges = self._densify_related_edges(None)
            F_out = self.F

        if joint:
            # One env-level formula view for the joint centralized actor. Per-model
            # job state belongs on the slot nodes/edges, so zero its formula-node
            # columns here; shared (public) knowledge is identical across agents in
            # centralized mode, so any agent's row is the env view and the mask is
            # the union of concrete formulas.
            formula_features[..., _JOINT_FORMULA_ZERO_COLS] = 0.0
            mask = self.concrete.any(dim=1, keepdim=True).expand(self.B, self.A, self.F)
            return {
                "formula_features": formula_features[:, 0],
                "formula_mask": mask[:, 0].to(self.dtype),
                "formula_ids": ids[:, 0],
                "neg_formula_ids": neg_ids[:, 0],
                "agent_scalars": agent_scalars[:, 0],
                "query_related_edges": query_edges[:, 0],
            }

        return {
            "formula_features": formula_features.reshape(BA, F_out, formula_features.shape[-1]),
            "formula_mask": mask.reshape(BA, F_out).to(self.dtype),
            "formula_ids": ids.reshape(BA, F_out),
            "neg_formula_ids": neg_ids.reshape(BA, F_out),
            "agent_scalars": agent_scalars.reshape(BA, AGENT_SCALAR_DIM),
            "query_related_edges": query_edges.reshape(BA, F_out, F_out),
        }

    # ------------------------------------------------------------------
    # Values, metrics, helpers

    def economic_value(self) -> torch.Tensor:
        # Learned mechanisms use the rewards returned by F, not a wealth
        # ledger, so their value reads zero (as does centralized planning).
        if self.centralized or self.cfg.control_mode == "learned":
            return torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        # Sum each agent's state over its formulas, then map through the baseline's
        # closed-form value -> per-(batch, agent) wealth.
        return self.mechanism.value(self.s.sum(dim=1))

    def _compute_rewards(self) -> torch.Tensor:
        if not self.centralized and self.cfg.control_mode == "learned":
            return self._step_reward
        value = self.economic_value()
        rewards = value - self.prev_reward
        self.prev_reward = value
        return rewards

    def resolved_target_count(self) -> torch.Tensor:
        true_sign0 = torch.arange(self.N, device=self.device)
        target_theorems = self.target_theorem
        resolved_pair = self.public_resolved[:, : self.N] | self.public_resolved[:, self.N :]
        return (target_theorems & resolved_pair[:, true_sign0]).sum(dim=-1)

    def _resolved_target_mask(self) -> torch.Tensor:
        """(B, N) bool: target theorem t resolved (either sign in public lib).

        Mirrors ``resolved_target_count``'s per-theorem resolved test, kept as a
        mask so the per-step recorder can stamp first-resolution steps.
        """
        target_theorems = self.target_theorem
        resolved_pair = self.public_resolved[:, : self.N] | self.public_resolved[:, self.N :]
        return target_theorems & resolved_pair

    def _init_resolution_tracking(self) -> None:
        """First-resolution step per target theorem, for the outer fitness.

        ``target_resolution_step`` is (B, N), init to the sentinel
        ``max_timestep + 1`` (= "unresolved within horizon"); the per-step
        recorder overwrites a slot once, the first step its target closes. Slots
        for non-targets are left at the sentinel and excluded from the fitness.
        """
        self._fitness_sentinel = int(self.cfg.max_timestep) + 1
        self.target_resolution_step = torch.full(
            (self.B, self.N), self._fitness_sentinel, dtype=torch.long, device=self.device
        )
        # Stamp any target already resolved at construction (carries the env's
        # current timestep).
        self._record_target_resolutions()

    def _record_target_resolutions(self) -> None:
        """Stamp the current step on targets resolving for the first time.

        A target counts as resolved-at-t the first step either sign of its
        theorem enters the public library, and is never overwritten afterward.
        Reads no reward; feeds only the outer fitness.
        """
        newly = self._resolved_target_mask() & (self.target_resolution_step == self._fitness_sentinel)
        step = self.timestep.unsqueeze(1).expand(self.B, self.N)
        self.target_resolution_step = torch.where(newly, step, self.target_resolution_step)

    def fitness(self, tau: float | None = None) -> torch.Tensor:
        """(B,) outer fitness: mean over targets of exp(-t_first/tau) or 0.

        Unresolved targets (still at the sentinel) score 0; envs with no targets score 0. ``tau``
        defaults to ``cfg.fitness_tau``. Outer-only; the dense per-step reward is
        untouched.
        """
        if tau is None:
            tau = float(self.cfg.fitness_tau)
        is_target = self.target_theorem
        resolved = self.target_resolution_step < self._fitness_sentinel
        score = torch.where(
            resolved,
            torch.exp(-self.target_resolution_step.to(torch.float32) / tau),
            torch.zeros_like(self.target_resolution_step, dtype=torch.float32),
        )
        score = score * is_target.to(torch.float32)
        n_targets = is_target.sum(dim=1).to(torch.float32).clamp_min(1.0)
        return score.sum(dim=1) / n_targets

    def eval_metrics(self) -> dict[str, torch.Tensor]:
        """(B,)-valued outer metrics: discounted proving rate and resolved fraction."""
        return {
            # Raw discounted resolution mean_i exp(-t_first/tau): horizon-robust.
            # Outer-only.
            "discounted_resolved": self.fitness(),
            "resolved_frac": (
                self.resolved_target_count().to(torch.float32)
                / self.is_target[:, : self.N].sum(dim=1).to(torch.float32).clamp_min(1.0)
            ),
        }

    def _budget_from_action_latent(self, z: torch.Tensor) -> torch.Tensor:
        capped = torch.clamp(z, max=self.budget_latent_cap)
        return torch.clamp(torch.round(torch.exp(capped)), min=1).to(torch.long)

    def _truncated_geometric_mean(self, p: torch.Tensor) -> torch.Tensor:
        orig_dtype = p.dtype
        p = torch.clamp(p, 0.0, 1.0).to(torch.float64)
        H = float(self.cfg.horizon_H)
        out = torch.full_like(p, H)
        good = p > 1e-12
        r = 1.0 - p
        r_h = r.pow(H)
        denom = 1.0 - r_h
        # E[T | T <= H] for geometric first-success time:
        #   sum_{t=1}^H t p r^(t-1) / (1 - r^H)
        # = (1 - (H+1)r^H + H r^(H+1)) / (p * (1 - r^H)).
        numer = 1.0 - (H + 1.0) * r_h + H * r_h * r
        expected = numer / (p.clamp_min(1e-12) * denom.clamp_min(1e-12))
        expected = torch.where(p < 1e-5, torch.full_like(expected, 0.5 * (H + 1.0)), expected)
        return torch.where(good & (denom > 1e-12), expected, out).to(orig_dtype)

    def _gather_neg(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.neg_ids)

    @staticmethod
    def _gather_baf(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    def _query_axis_dense(
        self, match_idx: torch.Tensor, scatter_idx: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # Inductor/Triton codegen only pays off on CUDA; on CPU it stays eager
        # (and avoids macOS libomp double-init). fused == eager bit-for-bit.
        if self.device.type == "cuda":
            self._mark_query_edges_dynamic()
            return _query_axis_dense_compiled(
                self.query_values, match_idx, scatter_idx, target, self.F
            )
        return _query_axis_dense_eager(
            self.query_values, match_idx, scatter_idx, target, self.F
        )

    def _gather_query_col(self, target: torch.Tensor) -> torch.Tensor:
        """``query_weights[b, a, :, target]``: edge weights INTO ``target``.

        Densified from the sparse rep: keep edges whose dst is the agent's target
        and scatter their values onto the src axis.
        """
        with self._prof_section("query_gather_col"):
            return self._query_axis_dense(self._qw_dst, self._qw_src, target)

    def _mark_query_edges_dynamic(self) -> None:
        # The edge count E is resampled with the graph each epoch, so mark the E
        # axis dynamic on all three edge tensors: one dynamic-E Triton kernel
        # instead of an Inductor recompile per distinct edge count (B/A/F stay
        # static for full specialization).
        torch._dynamo.maybe_mark_dynamic(self.query_values, 2)
        torch._dynamo.maybe_mark_dynamic(self._qw_src, 0)
        torch._dynamo.maybe_mark_dynamic(self._qw_dst, 0)

    def _gather_query_row(self, target: torch.Tensor) -> torch.Tensor:
        """``query_weights[b, a, target, :]``: edge weights FROM ``target``."""
        with self._prof_section("query_gather_row"):
            return self._query_axis_dense(self._qw_src, self._qw_dst, target)
