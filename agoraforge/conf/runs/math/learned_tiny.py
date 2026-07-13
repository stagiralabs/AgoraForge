"""Tiny learned-market run: the inner evaluation for the outer mechanism search.

One inner run trains agent policies under a FIXED learned market mechanism (its
parameters set by `--config.env.learned_mechanism_params=<path>`, or fresh if
omitted) and reports the fraction of targets resolved. An outer mechanism
search calls many of these to compare candidate mechanisms.
"""
from agoraforge.envs.math import presets as math_presets
from agoraforge.conf.schema import run_config


def get_config():
    return run_config(env="math", overrides={
        "env": math_presets.small_learned(),
        "levels": [{"name": "small_learned"}],
        "schedule": [{"epoch": 0, "weights": {"small_learned": 1.0}}],
    })
