"""Tiny CPU mechanism search -- proves the cold-start batched SHADE loop closes
end to end.

    python -m agoraforge.search --config=agoraforge/conf/runs/math/mechanism_search_smoke.py
"""
from agoraforge.conf.runs.math.mechanism_search import base_config


def get_config():
    cfg = base_config(device='cpu', online_batch_size=16384, wandb_enabled=False)
    cfg.training.online_buffer_size = 8
    s = cfg.search
    s.generations = 2
    s.population = 4
    s.inner_epochs = 3
    s.tail_epochs = 2
    s.gpus = ''                  # CPU
    return cfg
