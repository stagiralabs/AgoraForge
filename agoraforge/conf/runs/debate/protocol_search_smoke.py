"""Tiny protocol-search smoke: exercises the full search loop in minutes."""

from agoraforge.conf.runs.debate.protocol_search import base_config


def get_config():
    cfg = base_config()
    cfg.device = "cpu"
    cfg.search.population = 4
    cfg.search.generations = 2
    cfg.search.inner_epochs = 6
    cfg.search.tail_epochs = 3
    cfg.training.online_buffer_size = 16
    cfg.search.wandb_enabled = False
    return cfg
