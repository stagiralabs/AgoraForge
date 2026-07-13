"""Action-space vocabulary for the math environment.

The math action types and conjecture modes shared by the resident env
(``agoraforge.envs.math.env.BatchedMathEnv``), the models, and the training loop.
"""

MATH_ACTION_TYPES = ['prove', 'conj', 'query_prob_time', 'query_related']
MATH_ACTION_TYPE_TO_IDX = {name: i for i, name in enumerate(MATH_ACTION_TYPES)}
NUM_MATH_ACTION_TYPES = len(MATH_ACTION_TYPES)
# Conjecture mode-logit convention: index 0 = outward, index 1 = inward.
CONJ_MODES = ['outward', 'inward']
