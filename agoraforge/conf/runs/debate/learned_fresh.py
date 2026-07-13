"""Fresh (no-op) learned protocol mechanism on the small_vanilla world.

The neutral point for protocol search: zero-initialized F pays nothing and P emits
0.5. Point a run at a searched theta via
``--config.env.learned_mechanism_params=<path>``.
"""

from agoraforge.envs.debate import presets as debate_presets
from agoraforge.conf.schema import run_config


def get_config():
    cfg = run_config()
    cfg.env = debate_presets.small_learned()
    cfg.levels = [{"name": "small_learned"}]
    cfg.schedule = [{"epoch": 0, "weights": {"small_learned": 1.0}}]
    return cfg
