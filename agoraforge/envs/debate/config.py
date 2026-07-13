"""Configuration for vanilla and protocol-bound synthetic debate."""

from dataclasses import dataclass, fields


CONTROL_MODES = ("vanilla", "learned")


@dataclass(kw_only=True)
class DebateConfig:
    env_name = "debate"

    # Claim universe and debate horizon. One debater acts per timestep, alternating.
    num_claims: int
    n_agents: int
    max_timestep: int
    judge_claim_cap: int

    # Distribution over the judge's factor graph.
    claim_graph_m: float
    coupling_min: float
    coupling_max: float
    p_attack: float
    unary_cutoff: float
    truth_gibbs_sweeps: int
    marginal_gibbs_sweeps: int

    control_mode: str = "vanilla"
    action_dim: int = 2

    # Protocol-bound F,w,P parameterization.
    learned_state_dim: int = 3
    learned_mechanism_hidden: int = 16
    learned_predictor_hidden: int = 4
    learned_mechanism_layernorm: bool = False
    learned_mechanism_params: str | None = None

    def __post_init__(self) -> None:
        self.num_claims = int(self.num_claims)
        self.n_agents = int(self.n_agents)
        self.max_timestep = int(self.max_timestep)
        self.judge_claim_cap = int(self.judge_claim_cap)
        if self.num_claims < 2:
            raise ValueError(f"num_claims must be >= 2, got {self.num_claims}")
        if self.n_agents != 2:
            raise ValueError(
                f"the draft's vanilla/protocol-bound games require exactly 2 debaters, got {self.n_agents}"
            )
        if self.max_timestep < 1:
            raise ValueError(f"max_timestep must be >= 1, got {self.max_timestep}")
        if not 1 <= self.judge_claim_cap <= min(16, self.num_claims):
            raise ValueError(
                "judge_claim_cap must be in [1, min(16, num_claims)], "
                f"got {self.judge_claim_cap}"
            )
        # Vanilla sends its whole possible transcript to the judge. Exact
        # enumeration therefore bounds turns+target, not transcript storage.
        if self.control_mode == "vanilla" and min(self.num_claims, self.max_timestep + 1) > 16:
            raise ValueError("vanilla debate needs max_timestep + target <= 16 for the exact judge")

        self.claim_graph_m = float(self.claim_graph_m)
        self.coupling_min = float(self.coupling_min)
        self.coupling_max = float(self.coupling_max)
        self.p_attack = float(self.p_attack)
        self.unary_cutoff = float(self.unary_cutoff)
        self.truth_gibbs_sweeps = int(self.truth_gibbs_sweeps)
        self.marginal_gibbs_sweeps = int(self.marginal_gibbs_sweeps)
        if self.claim_graph_m < 1.0:
            raise ValueError(f"claim_graph_m must be >= 1, got {self.claim_graph_m}")
        if not 0.0 <= self.coupling_min <= self.coupling_max:
            raise ValueError(
                "need 0 <= coupling_min <= coupling_max, "
                f"got [{self.coupling_min}, {self.coupling_max}]"
            )
        if not 0.0 <= self.p_attack <= 1.0:
            raise ValueError(f"p_attack must be in [0,1], got {self.p_attack}")
        if self.unary_cutoff < 0.0:
            raise ValueError(f"unary_cutoff must be >= 0, got {self.unary_cutoff}")
        if self.truth_gibbs_sweeps < 1 or self.marginal_gibbs_sweeps < 1:
            raise ValueError("Gibbs sweep counts must be >= 1")

        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {self.control_mode!r}")
        self.action_dim = int(self.action_dim)
        self.learned_state_dim = int(self.learned_state_dim)
        self.learned_mechanism_hidden = int(self.learned_mechanism_hidden)
        self.learned_predictor_hidden = int(self.learned_predictor_hidden)
        if min(
            self.action_dim,
            self.learned_state_dim,
            self.learned_mechanism_hidden,
            self.learned_predictor_hidden,
        ) < 1:
            raise ValueError("action and learned protocol dimensions must be >= 1")

    def to_json_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_json_dict(cls, data: dict) -> "DebateConfig":
        return cls(**data)
