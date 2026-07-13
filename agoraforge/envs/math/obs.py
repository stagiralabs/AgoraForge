"""Structured observation schema for the math env's resident models.

The resident env builds observations on-device by writing these named feature
columns directly; this module is the single source of truth for the column layout
and its derived widths.
"""

LOCAL_PROVEN = 0
PUBLIC_CONCRETE = 1
PUBLIC_PROVEN = 2
JOB_ACTIVE_TARGET = 3
JOB_TAU_REMAINING = 4
JOB_TYPE_PROVE = 5
JOB_TYPE_CONJ = 6
CUMULATIVE_PROOF = 7
CUMULATIVE_CONJ = 8
QUERY_PROB_TIME_KNOWN = 9
QUERY_PROBABILITY = 10
QUERY_TIME_RATIO = 11
QUERY_RELATED_IN_KNOWN = 12
QUERY_RELATED_OUT_KNOWN = 13
# Public target designation: 1.0 if this formula's theorem carries the external
# bounty (is a target). The centralized planner has no market, so without this
# feature it is blind to targets; exposed in every mode (a posted bounty is
# public info) so target-visibility is symmetric.
FORMULA_IS_TARGET = 14

FORMULA_FEATURE_DIM = 15
AGENT_VALUE = 0
AGENT_TIMESTEP = 1
AGENT_LAST_TURN = 2
AGENT_SCALAR_DIM = 3

# Learned/market modes append the agent's own per-formula state (s) after the
# shared math/knowledge columns, so the mechanism-mode formula feature is the full
# column set plus d_state.
LEARNED_PROVING_COLS = (
    LOCAL_PROVEN, PUBLIC_CONCRETE, PUBLIC_PROVEN, JOB_ACTIVE_TARGET,
    JOB_TAU_REMAINING, JOB_TYPE_PROVE, JOB_TYPE_CONJ, CUMULATIVE_PROOF,
    CUMULATIVE_CONJ, QUERY_PROB_TIME_KNOWN, QUERY_PROBABILITY, QUERY_TIME_RATIO,
    QUERY_RELATED_IN_KNOWN, QUERY_RELATED_OUT_KNOWN, FORMULA_IS_TARGET,
)
LEARNED_PROVING_DIM = len(LEARNED_PROVING_COLS)  # 15


def learned_formula_feat_dim(d_state: int) -> int:
    return LEARNED_PROVING_DIM + int(d_state)


# Model-slot nodes (centralized control mode). One node per math model,
# permutation-equivariant: features carry job state only, never a slot id. The
# actor decodes an action off every slot in a single forward, so no focal marker.
SLOT_JOB_ACTIVE = 0
SLOT_JOB_TAU_REMAINING = 1
SLOT_JOB_TYPE_PROVE = 2
SLOT_JOB_TYPE_CONJ = 3
SLOT_FEATURE_DIM = 4

# Slot-formula edge channels: the slot's active-job target and its cumulative
# work on that formula.
SLOT_EDGE_JOB_TARGET = 0
SLOT_EDGE_CUM_PROOF = 1
SLOT_EDGE_CUM_CONJ = 2
SLOT_EDGE_DIM = 3

ACTOR_OBS_LEGEND = {
    "formula_features": [
        "local_proven",
        "public_concrete",
        "public_proven",
        "job_active_target",
        "job_tau_remaining",
        "job_type_prove",
        "job_type_conj",
        "cumulative_proof",
        "cumulative_conj",
        "query_prob_time_known",
        "query_probability",
        "query_time_ratio",
        "query_related_in_known",
        "query_related_out_known",
        "formula_is_target",
    ],
    "agent_scalars": [
        "economic_value_ratio",
        "timestep_ratio",
        "last_turn_before_done",
    ],
    "formula_ids": "Formula id for each row in formula_features.",
    "neg_formula_ids": "Negated formula id for each row in formula_features.",
    "formula_mask": "1 for visible formula rows.",
    "query_related_edges": "Known query-related estimated weights, row source to column target.",
    "slot_features": [
        "job_active",
        "job_tau_remaining",
        "job_type_prove",
        "job_type_conj",
    ],
    "slot_formula_edges": "Per slot-formula pair: active-job target, cumulative proof, cumulative conj (centralized mode only).",
}
