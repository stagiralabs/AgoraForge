"""Structured observation schema for debate resident models.

The resident env builds observations on-device by writing these named feature columns
directly; this module is the single source of truth for the column layout and widths.
"""

CLAIM_UNARY = 0                # judge's prior lean b_v (raw log-odds)
CLAIM_IS_TARGET = 1
CLAIM_IN_TRANSCRIPT = 2
CLAIM_REVEALED_BY_ME = 3
CLAIM_REVEALED_BY_PRO = 4      # revealed by a role=+1 agent
CLAIM_REVEALED_BY_CON = 5      # revealed by a role=-1 agent
CLAIM_REVEAL_STEP_RATIO = 6    # reveal step / max_timestep (0 if unrevealed)
CLAIM_JUDGE_SELECTED = 7       # selected for the terminal bounded judge call
CLAIM_JUDGE_P = 8              # terminal judge marginal (0 unless selected)
CLAIM_FEATURE_DIM = 9

# The agent's own per-claim mechanism state s[v, j] is appended after the shared
# columns, so the actor feature width is CLAIM_FEATURE_DIM + d_state.
def claim_feat_dim(d_state: int) -> int:
    return CLAIM_FEATURE_DIM + int(d_state)



AGENT_TIMESTEP = 0
AGENT_LAST_TURN = 1
AGENT_ROLE = 2
AGENT_SCALAR_DIM = 3

ACTOR_OBS_LEGEND = {
    "claim_features": [
        "unary_bias",
        "is_target",
        "in_transcript",
        "revealed_by_me",
        "revealed_by_pro",
        "revealed_by_con",
        "reveal_step_ratio",
        "judge_selected",
        "judge_marginal",
    ],
    "agent_scalars": [
        "timestep_ratio",
        "last_turn_before_done",
        "role",
    ],
    "claim_mask": "1 for every claim (debaters are computationally unbounded).",
    "claim_couplings": "Full signed coupling matrix J of the judge's factor graph.",
}
