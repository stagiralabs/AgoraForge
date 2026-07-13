"""MathConfig: single dataclass holding all math-env hyperparameters."""

from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(kw_only=True)
class MathConfig:
    env_name = "math"

    # ── 1a. Agent & formula universes ──
    num_theorems: int                   # base theorem count n; runtime formula universe has size 2n
    F_size: int = field(init=False)     # derived runtime formula count
    n_agents: int                       # |N|

    # ── 1b. Latent environment structure (instance data) ──
    theorem_graph_m: float              # Barabási–Albert attachment count (expected edges per theorem; may be fractional)
    truth_map: Optional[np.ndarray] = None          # shape (num_theorems,), values {0,1}; picks true pair member
    utility_weights: Optional[Dict[Tuple[int, int], float]] = None  # formula-id weights, w(psi,phi)

    # Factor-graph formula-graph distribution. Weight variables
    # are discretized to `n_weight_levels` values in [weight_min, weight_max]; each weight
    # factor penalizes "high weight ∧ source true ∧ target false" with strength
    # `weight_penalty` (0 ⇒ independent-uniform truths and weights). Truths are drawn by
    # `truth_gibbs_sweeps` Gibbs sweeps over the weight-marginalized truth MRF.
    n_weight_levels: int
    weight_min: float
    weight_max: float
    weight_penalty: float
    truth_gibbs_sweeps: int

    # Random *unary* factors layered on alongside the Barabási–Albert structure.
    # Each truth variable T_t gets `bias_t * T_t`
    # with bias_t ~ Uniform[-truth_unary_cutoff, +truth_unary_cutoff] (favouring one pair
    # member true), and each weight variable W gets `bias_W * w` with
    # bias_W ~ Uniform[-weight_unary_cutoff, +weight_unary_cutoff] (tilting it high/low).
    # `age_directional_weight_bias` adds a deterministic tilt: lower theorem id -> higher
    # theorem id gets +bias*w; higher -> lower gets -bias*w. Theorem id is creation order
    # in the BA graph, so lower id means earlier theorem.
    # A cutoff of 0 disables that random family; age_directional_weight_bias=0 disables
    # the deterministic tilt. All three at 0 recover the pairwise-only model.
    truth_unary_cutoff: float
    weight_unary_cutoff: float
    age_directional_weight_bias: float = 0.0

    # ── 1c. Initial world state (b_0 specification) ──
    prob_initially_resolved: float
    prob_initially_target: float
    initial_cash: float
    negative_return_penalty: float
    max_timestep: int
    # ── 1d. Proof kernel hyperparameters ──
    rho: float

    # ── 1e. Conjecture kernel hyperparameters ──
    eta: float

    # ── 1f. Query model hyperparameters ──
    horizon_H: int
    query_truth_prior_correct: float
    query_weight_noise: float
    query_num_related: int

    # ── 1g. Market mechanism ──
    bounty_demand: float
    # ── 1h. Market design variants ──
    # First-prover bonus: cash paid to the agent who publishes a proof that
    # newly resolves a public formula — internalizes the proving externality
    # (otherwise a YES holder can free-ride on others' proofs). 0.0 = off.
    first_prover_bonus: float = 0.0
    # Concentrate the first-prover bonus on targets: when True, the bonus is paid
    # only for resolving a TARGET theorem, not any public formula (the diffuse
    # any-formula bonus spreads proving effort onto non-targets; gating it on
    # targets focuses effort like the planner).
    prover_bonus_targets_only: bool = False
    # ── 1i. Control mode ──
    # Every decentralized market is an *mechanism*: a per-formula/per-agent state
    # evolved by a single map (envs/math/mechanism.py). The hand-coded baselines
    # are exact closed-form mechanisms (mechanism.BaselineMarketMechanism); the
    # 'learned' mode is the searched-MLP mechanism.
    # 'market': trade via the clearing mechanism; reward = marked-wealth deltas.
    #   first_prover_bonus / prover_bonus_targets_only select flat / fp3 / tgt.
    # 'bounty_only': decentralized like 'market' but with NO trading — agents
    # still prove and publish and the first prover collects the bonus, but the
    # mechanism's market dynamics are off (no markets created, market actions
    # masked), so reward reduces to first-prover bonus deltas. Isolates whether
    # the *market* does work beyond the *prize*.
    # 'collaborative': decentralized like bounty_only (no trading), but when any
    # agent proves a formula the first_prover_bonus is paid to EVERY agent. With
    # prover_bonus_targets_only=True this is the "if anyone proves a target,
    # everyone gets rich" baseline.
    # 'centralized': the planner baseline — no markets, publishing, or cash;
    # proofs, conjectures, and query results become public on completion; every
    # agent receives the same team reward each step: target theorems resolved.
    control_mode: str = 'market'
    # Width of the per-formula continuous market action the policy emits (and the
    # mechanism consumes as a[phi,i]). The clearing baseline reads it as the demand
    # curve (q0, q1) so needs exactly 2; the learned mechanism may use any width.
    action_dim: int = 2
    # Resident mechanism path: cap each agent's formula-level observation (and
    # action masks) via a theorem-level context. 0 = off (every concrete formula).
    # A positive cap K gives each agent floor(K/2) theorem slots, initialized as a
    # random subset of concrete theorems and emitted as both signed formulas per
    # theorem. Active prove/conjecture/query targets refresh their last-touched
    # time; successful conjectures and query_related references insert new theorem
    # ids by evicting the least-recently touched context theorem. This fixes the
    # policy's per-agent input shape as the formula universe grows, removing
    # variable-kernel recompile storms and dropping O(F^2) attention/pointer work
    # to O(K^2). The critic sees the same per-agent context as the actor.
    obs_formula_cap: int = 0
    # Cap on the budget head's latent (exp'd into a positive budget by the policy
    # decode); bounds the budget an agent can commit in one action.
    budget_latent_cap: float = 20.0
    # ── 1j. Learned market mechanism (control_mode='learned') ──
    # The market mechanism is a pooled MLP over per-(formula, agent) states (see
    # envs/math/mechanism.py). Agents submit continuous signals and publish
    # proofs; state updates and rewards are produced by the mechanism.
    # The mechanism parameters are frozen during inner training and searched over
    # by the outer loop; `learned_mechanism_params` is a path to a saved flat
    # parameter vector (None = freshly initialized zero-output mechanism).
    learned_state_dim: int = 3
    learned_mechanism_hidden: int = 16
    learned_mechanism_layernorm: bool = False
    learned_mechanism_params: Optional[str] = None

    # ── 1k. Outer mechanism-search fitness ──
    # The outer search ranks mechanisms by a fitness that discounts each target
    # by *when* it first resolves, not the horizon-degenerate resolved fraction
    # at a fixed max_timestep (which → 1 for any actor as max_timestep → ∞).
    # Per target i with first-resolution
    # time t_i, the per-target score is exp(-t_i / tau) if resolved within
    # max_timestep else 0; the fitness is the mean over targets (and
    # rollouts/seeds). `tau` is the reporting discount's time scale.
    # Diagnostic-only here: it changes no reward or dynamics, only the metric
    # train.py reports (resolved fraction is still reported alongside).
    fitness_tau: float = 10.0

    def __post_init__(self):
        self.num_theorems = int(self.num_theorems)
        _require(self.num_theorems > 0, "num_theorems must be positive")
        self.F_size = 2 * self.num_theorems
        self.fitness_tau = float(self.fitness_tau)
        _require(self.fitness_tau > 0.0, f"fitness_tau must be positive, got {self.fitness_tau}")
        self.theorem_graph_m = float(self.theorem_graph_m)
        _require(
            self.theorem_graph_m >= 1.0,
            f"theorem_graph_m must be >= 1.0, got {self.theorem_graph_m}",
        )
        self.n_weight_levels = int(self.n_weight_levels)
        self.weight_min = float(self.weight_min)
        self.weight_max = float(self.weight_max)
        self.weight_penalty = float(self.weight_penalty)
        self.truth_gibbs_sweeps = int(self.truth_gibbs_sweeps)
        self.truth_unary_cutoff = float(self.truth_unary_cutoff)
        self.weight_unary_cutoff = float(self.weight_unary_cutoff)
        self.age_directional_weight_bias = float(self.age_directional_weight_bias)
        _require(self.n_weight_levels >= 1, f"n_weight_levels must be >= 1, got {self.n_weight_levels}")
        _require(
            0.0 <= self.weight_min <= self.weight_max <= 1.0,
            f"need 0 <= weight_min <= weight_max <= 1, got [{self.weight_min}, {self.weight_max}]",
        )
        _require(self.weight_penalty >= 0.0, f"weight_penalty must be >= 0, got {self.weight_penalty}")
        _require(
            self.truth_gibbs_sweeps >= 1,
            f"truth_gibbs_sweeps must be >= 1, got {self.truth_gibbs_sweeps}",
        )
        _require(
            self.truth_unary_cutoff >= 0.0,
            f"truth_unary_cutoff must be >= 0, got {self.truth_unary_cutoff}",
        )
        _require(
            self.weight_unary_cutoff >= 0.0,
            f"weight_unary_cutoff must be >= 0, got {self.weight_unary_cutoff}",
        )
        _require(
            self.age_directional_weight_bias >= 0.0,
            f"age_directional_weight_bias must be >= 0, got {self.age_directional_weight_bias}",
        )

        self.rho = float(self.rho)
        self.eta = float(self.eta)
        _require(self.rho > 0.0, f"rho must be > 0, got {self.rho}")
        _require(self.eta > 0.0, f"eta must be > 0, got {self.eta}")

        self.prob_initially_resolved = float(self.prob_initially_resolved)
        self.prob_initially_target = float(self.prob_initially_target)
        # One theorem is always forced resolved and a distinct one forced to be a
        # target, so each probability must cover at least that 1/num_theorems mass,
        # and the two bands together cannot exceed the whole simplex.
        min_prob = 1.0 / self.num_theorems
        _require(
            self.prob_initially_resolved >= min_prob,
            f"need prob_initially_resolved >= 1/num_theorems = {min_prob:g}, "
            f"got {self.prob_initially_resolved}",
        )
        _require(
            self.prob_initially_target >= min_prob,
            f"need prob_initially_target >= 1/num_theorems = {min_prob:g}, got {self.prob_initially_target}",
        )
        _require(
            self.prob_initially_resolved + self.prob_initially_target <= 1.0,
            "need prob_initially_resolved + prob_initially_target <= 1, "
            f"got resolved={self.prob_initially_resolved}, target={self.prob_initially_target}",
        )
        _require(
            self.negative_return_penalty >= 0.0,
            f"negative_return_penalty must be >= 0, got {self.negative_return_penalty}",
        )
        _require(self.bounty_demand >= 0.0, f"bounty_demand must be >= 0, got {self.bounty_demand}")
        self.first_prover_bonus = float(self.first_prover_bonus)
        _require(
            self.first_prover_bonus >= 0.0,
            f"first_prover_bonus must be >= 0, got {self.first_prover_bonus}",
        )
        modes = ('market', 'bounty_only', 'collaborative', 'centralized', 'learned')
        _require(
            self.control_mode in modes,
            f"control_mode must be one of {modes}, got {self.control_mode!r}",
        )
        self.action_dim = int(self.action_dim)
        _require(self.action_dim >= 1, f"action_dim must be >= 1, got {self.action_dim}")
        self.obs_formula_cap = int(self.obs_formula_cap)
        _require(
            self.obs_formula_cap >= 0,
            f"obs_formula_cap must be >= 0 (0 = off), got {self.obs_formula_cap}",
        )
        if self.control_mode == 'market':
            _require(
                self.action_dim == 2,
                f"demand-curve market needs action_dim==2 (q0,q1), got {self.action_dim}",
            )
        self.learned_state_dim = int(self.learned_state_dim)
        self.learned_mechanism_hidden = int(self.learned_mechanism_hidden)
        _require(self.learned_state_dim >= 1, "learned_state_dim must be >= 1")
        _require(self.learned_mechanism_hidden >= 1, "learned_mechanism_hidden must be >= 1")
        _require(
            0.5 <= self.query_truth_prior_correct < 1.0,
            f"query_truth_prior_correct must be in [0.5,1), got {self.query_truth_prior_correct}",
        )
        _require(self.query_weight_noise >= 0.0, "query_weight_noise must be >= 0")
        _require(self.query_num_related >= 0, "query_num_related must be >= 0")

        if self.truth_map is not None:
            self._validate_truth_map()
        if self.utility_weights is not None:
            self._validate_utility_weights()

    def _validate_truth_map(self):
        _require(
            self.truth_map.shape == (self.num_theorems,),
            f"truth_map must have shape ({self.num_theorems},), got {self.truth_map.shape}",
        )
        _require(
            bool(np.all(np.isin(self.truth_map, [0, 1]))),
            "truth_map must contain only 0/1 pair-member indicators",
        )

    def _validate_utility_weights(self):
        nodes = set(range(self.F_size))
        for (src, dst), weight in self.utility_weights.items():
            _require(src in nodes and dst in nodes, f"invalid formula weight key ({src},{dst})")
            _require(src != dst, f"self weight w({src},{dst}) is not allowed")
            _require(weight > 0.0, f"w({src},{dst}) must be positive")

    def to_json_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if not f.init:
                continue
            value = getattr(self, f.name)
            if f.name == "truth_map":
                value = None if value is None else [int(v) for v in value.tolist()]
            elif f.name == "utility_weights":
                value = None if value is None else [
                    {"src": int(src), "dst": int(dst), "weight": float(weight)}
                    for (src, dst), weight in sorted(value.items())
                ]
            out[f.name] = value
        return out

    @classmethod
    def from_json_dict(cls, data: dict) -> "MathConfig":
        data = dict(data)
        if data.get("truth_map") is not None:
            data["truth_map"] = np.asarray(data["truth_map"], dtype=np.int32)
        if data.get("utility_weights") is not None:
            data["utility_weights"] = {
                (int(item["src"]), int(item["dst"])): float(item["weight"])
                for item in data["utility_weights"]
            } or None
        return cls(**data)
