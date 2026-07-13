"""Device-resident synthetic vanilla and protocol-bound debate.

Debaters see the complete factor graph and alternately add claims to an uncapped
public transcript (bounded only by the debate horizon).  Vanilla debate sends the
whole transcript to the judge.  A learned protocol invokes the judge once, after
the debate, on ``k`` claims chosen by its state direction ``w``; judge feedback is
passed through a final zero-action ``F`` update before ``P`` emits the prediction.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Sequence

import torch

from agoraforge.envs.interface import PolicyInputs
from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX
from agoraforge.envs.debate.config import DebateConfig
from agoraforge.envs.debate.judge import judge_marginals
from agoraforge.envs.debate.latent import sample_claim_graphs
from agoraforge.envs.debate.obs import (
    AGENT_LAST_TURN,
    AGENT_ROLE,
    AGENT_SCALAR_DIM,
    AGENT_TIMESTEP,
    CLAIM_FEATURE_DIM,
    CLAIM_IN_TRANSCRIPT,
    CLAIM_IS_TARGET,
    CLAIM_JUDGE_P,
    CLAIM_JUDGE_SELECTED,
    CLAIM_REVEALED_BY_CON,
    CLAIM_REVEALED_BY_ME,
    CLAIM_REVEALED_BY_PRO,
    CLAIM_REVEAL_STEP_RATIO,
    CLAIM_UNARY,
    claim_feat_dim,
)
from agoraforge.envs.debate.protocol import (
    BatchedDebateMechanism,
    D_E,
    E_IN_TRANSCRIPT,
    E_IS_JUDGE_STEP,
    E_IS_MY_TURN,
    E_IS_TARGET,
    E_JUDGE_P,
    E_JUDGE_SELECTED,
    E_REVEALED_NOW,
    E_ROLE,
    E_SUBMITTED_BY_ME,
    E_TIMESTEP_RATIO,
    mechanism_dims_from_config,
    mechanism_from_config,
)


@dataclass
class TensorActions:
    reveal_type: torch.Tensor
    reveal_claim: torch.Tensor
    signals: torch.Tensor


@dataclass
class EnvStateTensors:
    timestep: torch.Tensor
    couplings: torch.Tensor
    unary: torch.Tensor
    truth: torch.Tensor
    p_full: torch.Tensor
    target_idx: torch.Tensor
    is_target: torch.Tensor
    slots: torch.Tensor
    slot_valid: torch.Tensor
    slot_count: torch.Tensor
    in_transcript: torch.Tensor
    revealed_by: torch.Tensor
    reveal_step: torch.Tensor
    s: torch.Tensor
    judge_slots: torch.Tensor
    judge_slot_valid: torch.Tensor
    judge_selected: torch.Tensor
    judge_p: torch.Tensor
    prior_p: torch.Tensor
    final_p: torch.Tensor


class BatchedDebateEnv:
    centralized = False

    @classmethod
    def from_config(
        cls,
        cfg: DebateConfig,
        *,
        batch_size: int,
        device: torch.device,
        seeds: Sequence[int] | torch.Tensor | None = None,
        profiler=None,
        instance: dict[str, torch.Tensor] | None = None,
    ) -> "BatchedDebateEnv":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self = cls.__new__(cls)
        self.cfg = cfg
        self.device = torch.device(device)
        self.B = int(batch_size)
        self.B_base = self.B
        self.K = 1
        self.A = int(cfg.n_agents)
        self.N = int(cfg.num_claims)
        self.C = min(self.N, int(cfg.max_timestep) + 1)
        self.J = self.C if cfg.control_mode == "vanilla" else int(cfg.judge_claim_cap)
        self.dtype = torch.float32
        self._profiler = profiler
        self._step_count = 0
        self.mechanism = mechanism_from_config(cfg).to(self.device).eval()
        self.dims = self.mechanism.dims
        self.roles = torch.tensor([1.0, -1.0], device=self.device)
        self._init_tensors(self._make_generator(seeds), instance=instance)
        return self

    _PER_ENV_TENSORS = tuple(f.name for f in fields(EnvStateTensors))

    @classmethod
    def batched_from_config(
        cls,
        cfg: DebateConfig,
        *,
        base_batch_size: int,
        thetas: Sequence[torch.Tensor],
        device: torch.device,
        seeds: Sequence[int] | torch.Tensor | None = None,
        step_seed: int = 0,
        profiler=None,
        instance: dict[str, torch.Tensor] | None = None,
    ) -> "BatchedDebateEnv":
        del step_seed  # Debate has no environment-side stochasticity after reset.
        if cfg.control_mode != "learned":
            raise ValueError("batched candidate evaluation is learned-protocol only")
        K = len(thetas)
        if K < 1:
            raise ValueError("need at least one candidate theta")
        instance_rows = (
            int(instance["couplings"].shape[0]) if instance is not None else base_batch_size
        )
        preexpanded = instance_rows == K * base_batch_size
        if instance_rows not in (base_batch_size, K * base_batch_size):
            raise ValueError(
                f"precomputed batch has {instance_rows} rows, expected "
                f"{base_batch_size} or {K * base_batch_size}"
            )
        base = cls.from_config(
            cfg, batch_size=instance_rows, device=device, seeds=seeds,
            profiler=profiler, instance=instance,
        )
        if preexpanded:
            self = base
        else:
            self = cls.__new__(cls)
            for key, value in base.__dict__.items():
                if key not in cls._PER_ENV_TENSORS:
                    self.__dict__[key] = value
            for name in cls._PER_ENV_TENSORS:
                tensor = getattr(base, name)
                self.__dict__[name] = tensor.repeat((K,) + (1,) * (tensor.dim() - 1))
        self.K = K
        self.B_base = base_batch_size
        self.B = K * base_batch_size
        self.mechanism = BatchedDebateMechanism(
            thetas,
            mechanism_dims_from_config(cfg),
            layernorm=bool(cfg.learned_mechanism_layernorm),
        ).to(self.device)
        return self

    def _make_generator(self, seeds) -> torch.Generator:
        if seeds is None:
            seed_list = [0]
        elif torch.is_tensor(seeds):
            seed_list = [int(v) for v in seeds.detach().cpu().tolist()]
        else:
            seed_list = [int(v) for v in seeds]
        seed = 0
        for value in seed_list:
            seed = (seed * 1000003 + value) % (2**63 - 1)
        sample_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=sample_device).manual_seed(seed)

    def _init_tensors(
        self, gen: torch.Generator, *, instance: dict[str, torch.Tensor] | None = None,
    ) -> None:
        B, N, A, C, J = self.B, self.N, self.A, self.C, self.J
        sample_device = gen.device if hasattr(gen, "device") else torch.device("cpu")
        if instance is None:
            couplings, unary, truth, p_full = sample_claim_graphs(
                self.cfg, B, sample_device, gen,
            )
            target_idx = torch.randint(
                0, N, (B,), device=sample_device, generator=gen,
            )
        else:
            required = {"couplings", "unary", "truth", "p_full", "target_idx"}
            missing = required - instance.keys()
            if missing:
                raise ValueError(f"precomputed debate instance missing {sorted(missing)}")
            couplings, unary, truth, p_full = (
                instance[name] for name in ("couplings", "unary", "truth", "p_full")
            )
            target_idx = instance["target_idx"]
            if int(couplings.shape[0]) != B:
                raise ValueError(
                    f"precomputed batch has {couplings.shape[0]} rows, expected {B}"
                )
        self.couplings = couplings.to(self.device)
        self.unary = unary.to(self.device)
        self.truth = truth.to(self.device)
        self.p_full = p_full.to(self.device)
        self.target_idx = target_idx.to(self.device, dtype=torch.long)
        self.is_target = torch.nn.functional.one_hot(self.target_idx, N).to(self.dtype)

        self.timestep = torch.zeros(B, dtype=torch.long, device=self.device)
        self.slots = torch.zeros((B, C), dtype=torch.long, device=self.device)
        self.slots[:, 0] = self.target_idx
        self.slot_valid = torch.zeros((B, C), dtype=torch.bool, device=self.device)
        self.slot_valid[:, 0] = True
        self.slot_count = torch.ones(B, dtype=torch.long, device=self.device)
        self.in_transcript = self.is_target.bool().clone()
        self.revealed_by = torch.zeros((B, N, A), dtype=torch.bool, device=self.device)
        self.reveal_step = torch.zeros((B, N), dtype=torch.long, device=self.device)
        self.s = torch.zeros((B, N, A, self.dims.d_state), dtype=self.dtype, device=self.device)

        self.judge_slots = torch.zeros((B, J), dtype=torch.long, device=self.device)
        self.judge_slot_valid = torch.zeros((B, J), dtype=torch.bool, device=self.device)
        self.judge_selected = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        self.judge_p = torch.zeros((B, N), dtype=self.dtype, device=self.device)
        b_target = self.unary.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)
        self.prior_p = torch.sigmoid(b_target)
        self.final_p = self.prior_p.clone()

    # ------------------------------------------------------------------
    # Actions and stepping

    def _reveal_permission(self) -> torch.Tensor:
        turn = (self.timestep % self.A).view(self.B, 1)
        agent = torch.arange(self.A, device=self.device).view(1, self.A)
        return agent == turn

    def available_action_masks(self) -> dict[str, torch.Tensor]:
        permission = self._reveal_permission()
        capacity = (self.slot_count < self.C).view(self.B, 1, 1)
        claim_mask = ~self.in_transcript.unsqueeze(1) & permission.unsqueeze(-1) & capacity
        reveal_possible = claim_mask.any(dim=-1)
        reveal_type = torch.stack([torch.ones_like(reveal_possible), reveal_possible], dim=-1)
        if self.cfg.control_mode == "learned":
            signal = self.in_transcript.unsqueeze(1) & permission.unsqueeze(-1)
        else:
            signal = torch.zeros_like(claim_mask)
        return {
            "reveal_type_mask": reveal_type.to(self.dtype),
            "reveal_claim_mask": claim_mask.to(self.dtype),
            "signal_mask": signal.to(self.dtype),
        }

    def policy_inputs(self) -> PolicyInputs:
        return PolicyInputs(
            actor_obs=self.actor_obs(),
            masks=self.available_action_masks(),
        )

    def step(self, actions: TensorActions) -> torch.Tensor:
        permission = self._reveal_permission()
        self._step_count += 1
        self.timestep += 1
        with self._prof_section("step_reveals"):
            revealed_now = self._process_reveals(actions, permission)

        rewards = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        if self.cfg.control_mode == "learned":
            with self._prof_section("step_protocol"):
                rewards += self._debate_updates(actions, permission, revealed_now)

        at_horizon = self._step_count >= self.cfg.max_timestep
        if at_horizon:
            finishing = torch.ones(self.B, dtype=torch.bool, device=self.device)
            with self._prof_section("step_judge"):
                self._invoke_judge(finishing)
            if self.cfg.control_mode == "learned":
                with self._prof_section("step_judge_feedback"):
                    rewards += self._judge_feedback(finishing)
                    prediction = self._protocol_prediction()
                    self.final_p = torch.where(finishing, prediction, self.final_p)
            else:
                target_p = self.judge_p.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)
                self.final_p = torch.where(finishing, target_p, self.final_p)
                rewards += (
                    finishing.to(self.dtype).unsqueeze(1)
                    * self.roles.view(1, self.A)
                    * (target_p - 0.5).unsqueeze(1)
                )

        return rewards.unsqueeze(-1)

    def _process_reveals(self, actions, permission) -> torch.Tensor:
        revealed_now = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)
        reveal_idx = REVEAL_TYPE_TO_IDX["reveal"]
        for agent in range(self.A):
            claim = actions.reveal_claim[:, agent]
            claim_safe = claim.clamp(min=0)
            wants = (
                (actions.reveal_type[:, agent] == reveal_idx)
                & permission[:, agent]
                & (claim >= 0)
                & (self.slot_count < self.C)
            )
            already = self.in_transcript.gather(1, claim_safe.unsqueeze(1)).squeeze(1)
            valid = wants & ~already
            slot_idx = self.slot_count.clamp(max=self.C - 1).unsqueeze(1)
            old_slot = self.slots.gather(1, slot_idx)
            self.slots.scatter_(
                1, slot_idx, torch.where(valid.unsqueeze(1), claim_safe.unsqueeze(1), old_slot)
            )
            old_valid = self.slot_valid.gather(1, slot_idx)
            self.slot_valid.scatter_(1, slot_idx, old_valid | valid.unsqueeze(1))
            self.slot_count += valid.to(torch.long)
            onehot = torch.nn.functional.one_hot(claim_safe, self.N).bool() & valid.unsqueeze(1)
            self.in_transcript |= onehot
            self.revealed_by[:, :, agent] |= onehot
            revealed_now |= onehot
            self.reveal_step = torch.where(
                onehot, self.timestep.unsqueeze(1).expand_as(self.reveal_step), self.reveal_step
            )
        return revealed_now

    def _base_env_input(self, agent: int, revealed_now: torch.Tensor) -> torch.Tensor:
        e = torch.zeros((self.B, self.N, D_E), dtype=self.dtype, device=self.device)
        e[..., E_SUBMITTED_BY_ME] = (
            self.revealed_by[:, :, agent] & revealed_now
        ).to(self.dtype)
        e[..., E_IN_TRANSCRIPT] = self.in_transcript.to(self.dtype)
        e[..., E_REVEALED_NOW] = revealed_now.to(self.dtype)
        e[..., E_IS_TARGET] = self.is_target
        e[..., E_ROLE] = self.roles[agent]
        e[..., E_IS_MY_TURN] = 1.0
        e[..., E_TIMESTEP_RATIO] = (
            self.timestep.to(self.dtype) / float(self.cfg.max_timestep)
        ).unsqueeze(1)
        return e

    def _mechanism_turn(self, a, e, agent: int):
        # Fold the candidate axis out (K=1 for the ordinary single-mechanism env).
        s = self.s.reshape(
            self.K, self.B_base, self.N, self.A, self.dims.d_state
        )
        a_k = a.reshape(self.K, self.B_base, self.N, self.dims.d_action)
        e_k = e.reshape(self.K, self.B_base, self.N, D_E)
        own_next, contribution = self.mechanism.turn(s, a_k, e_k, agent)
        return own_next.reshape(self.B, self.N, self.dims.d_state), contribution.reshape(self.B, self.N)

    def _debate_updates(self, actions, permission, revealed_now) -> torch.Tensor:
        rewards = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        for agent in range(self.A):
            e = self._base_env_input(agent, revealed_now)
            a = actions.signals[:, agent]
            own_next, contribution = self._mechanism_turn(a, e, agent)
            active = permission[:, agent].unsqueeze(1) & self.in_transcript
            self.s[:, :, agent] = torch.where(
                active.unsqueeze(-1), own_next, self.s[:, :, agent]
            )
            rewards[:, agent] = (contribution * active.to(self.dtype)).sum(dim=1)
        return rewards

    def _invoke_judge(self, finishing: torch.Tensor) -> None:
        if self.cfg.control_mode == "vanilla":
            slots = self.slots
            valid = self.slot_valid
        else:
            s = self.s.reshape(
                self.K, self.B_base, self.N, self.A, self.dims.d_state
            )
            scores = self.mechanism.judge_query_scores(s).reshape(self.B, self.N)
            scores = scores.masked_fill(~self.in_transcript, float("-inf"))
            slots = scores.topk(self.J, dim=1).indices
            valid = self.in_transcript.gather(1, slots)
        self.judge_slots = torch.where(finishing.unsqueeze(1), slots, self.judge_slots)
        self.judge_slot_valid = torch.where(finishing.unsqueeze(1), valid, self.judge_slot_valid)
        slot_p = judge_marginals(
            self.couplings, self.unary, self.judge_slots, self.judge_slot_valid
        )
        selected = self._scatter_slots(self.judge_slot_valid.to(self.dtype)).bool()
        values = self._scatter_slots(slot_p * self.judge_slot_valid.to(self.dtype))
        self.judge_selected = torch.where(finishing.unsqueeze(1), selected, self.judge_selected)
        self.judge_p = torch.where(finishing.unsqueeze(1), values, self.judge_p)

    def _scatter_slots(self, values: torch.Tensor) -> torch.Tensor:
        sentinel = torch.full_like(self.judge_slots, self.N)
        indices = torch.where(self.judge_slot_valid, self.judge_slots, sentinel)
        out = torch.zeros((self.B, self.N + 1), dtype=values.dtype, device=self.device)
        out.scatter_(1, indices, values)
        return out[:, : self.N]

    def _judge_feedback(self, finishing: torch.Tensor) -> torch.Tensor:
        rewards = torch.zeros((self.B, self.A), dtype=self.dtype, device=self.device)
        zeros = torch.zeros(
            (self.B, self.N, self.dims.d_action), dtype=self.dtype, device=self.device
        )
        no_reveals = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)
        for agent in range(self.A):
            e = self._base_env_input(agent, no_reveals)
            e[..., E_IS_MY_TURN] = 0.0
            e[..., E_JUDGE_SELECTED] = self.judge_selected.to(self.dtype)
            e[..., E_JUDGE_P] = self.judge_p
            e[..., E_IS_JUDGE_STEP] = 1.0
            own_next, contribution = self._mechanism_turn(zeros, e, agent)
            active = finishing.unsqueeze(1) & self.in_transcript
            self.s[:, :, agent] = torch.where(
                active.unsqueeze(-1), own_next, self.s[:, :, agent]
            )
            rewards[:, agent] = (contribution * active.to(self.dtype)).sum(dim=1)
        return rewards

    def _protocol_prediction(self) -> torch.Tensor:
        s = self.s.reshape(self.K, self.B_base, self.N, self.A, self.dims.d_state)
        transcript = self.in_transcript.reshape(self.K, self.B_base, self.N)
        return self.mechanism.predict(s, transcript).reshape(self.B)

    def _prof_section(self, name: str):
        profiler = getattr(self, "_profiler", None)
        return profiler.section(name) if profiler is not None else nullcontext()

    # ------------------------------------------------------------------
    # Observations

    def actor_obs(self) -> dict[str, torch.Tensor]:
        return self._obs()

    def _obs(self) -> dict[str, torch.Tensor]:
        B, N, A = self.B, self.N, self.A
        features = torch.zeros(
            (B, A, N, CLAIM_FEATURE_DIM), dtype=self.dtype, device=self.device
        )
        features[..., CLAIM_UNARY] = self.unary.unsqueeze(1)
        features[..., CLAIM_IS_TARGET] = self.is_target.unsqueeze(1)
        features[..., CLAIM_IN_TRANSCRIPT] = self.in_transcript.unsqueeze(1).to(self.dtype)
        features[..., CLAIM_REVEALED_BY_ME] = self.revealed_by.permute(0, 2, 1).to(self.dtype)
        features[..., CLAIM_REVEALED_BY_PRO] = self.revealed_by[:, :, 0].unsqueeze(1).to(self.dtype)
        features[..., CLAIM_REVEALED_BY_CON] = self.revealed_by[:, :, 1].unsqueeze(1).to(self.dtype)
        features[..., CLAIM_REVEAL_STEP_RATIO] = (
            self.reveal_step.to(self.dtype) / float(self.cfg.max_timestep)
        ).unsqueeze(1)
        features[..., CLAIM_JUDGE_SELECTED] = self.judge_selected.unsqueeze(1).to(self.dtype)
        features[..., CLAIM_JUDGE_P] = self.judge_p.unsqueeze(1)
        features = torch.cat([features, self.s.permute(0, 2, 1, 3)], dim=-1)

        expected = claim_feat_dim(self.dims.d_state)
        if features.shape[-1] != expected:
            raise RuntimeError(f"unexpected claim feature width {features.shape[-1]} != {expected}")

        scalars = torch.zeros((B, A, AGENT_SCALAR_DIM), dtype=self.dtype, device=self.device)
        scalars[..., AGENT_TIMESTEP] = (
            self.timestep.to(self.dtype) / float(self.cfg.max_timestep)
        ).unsqueeze(1)
        scalars[..., AGENT_LAST_TURN] = (
            self.timestep >= self.cfg.max_timestep - 1
        ).to(self.dtype).unsqueeze(1)
        scalars[..., AGENT_ROLE] = self.roles.view(1, A)
        couplings = self.couplings.unsqueeze(1).expand(B, A, N, N)
        return {
            "claim_features": features.reshape(B * A, N, features.shape[-1]),
            "claim_mask": torch.ones((B * A, N), dtype=self.dtype, device=self.device),
            "agent_scalars": scalars.reshape(B * A, AGENT_SCALAR_DIM),
            "claim_couplings": couplings.reshape(B * A, N, N),
        }

    # ------------------------------------------------------------------
    # Outer objective

    def _target_full_p(self) -> torch.Tensor:
        return self.p_full.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)

    def fitness(self) -> torch.Tensor:
        """Closeness of the protocol estimate to the full-graph target marginal."""
        return 1.0 - (self.final_p - self._target_full_p()).pow(2)

    def anchor_metrics(self) -> dict[str, torch.Tensor]:
        full = self._target_full_p()
        return {
            "acc_prior": 1.0 - (self.prior_p - full).pow(2),
            "acc_full": torch.ones_like(full),
        }

    def eval_metrics(self) -> dict[str, torch.Tensor]:
        anchors = self.anchor_metrics()
        return {
            "judge_acc": self.fitness(),
            "judge_acc_prior": anchors["acc_prior"],
            "judge_acc_full": anchors["acc_full"],
        }
