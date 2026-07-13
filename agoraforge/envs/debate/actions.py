"""Action-space vocabulary for the debate environment."""

REVEAL_TYPES = ['pass', 'reveal']
REVEAL_TYPE_TO_IDX = {name: i for i, name in enumerate(REVEAL_TYPES)}
NUM_REVEAL_TYPES = len(REVEAL_TYPES)
