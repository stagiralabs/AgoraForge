"""Centralized-controller baseline run."""
from agoraforge.envs.math import presets as math_presets
from agoraforge.conf.schema import run_config


def get_config():
    return run_config(env="math", overrides={
        "env": math_presets.small_centralized(),
        "levels": [{"name": "small_centralized"}],
        "schedule": [{"epoch": 0, "weights": {"small_centralized": 1.0}}],
    })
