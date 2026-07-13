"""Model presets."""
from ml_collections import ConfigDict


def small_baseline():
    return ConfigDict({
        'n_embd': 25,
        'n_head': 5,
        'n_layer': 1,
        'init_std': 0.02,
        # None delegates the default to the environment builder; the resolved
        # checkpoint config always records dense, sparse, or auto explicitly.
        'attention_mode': None,
        'sparse_attention_min_nodes': 64,
    })
