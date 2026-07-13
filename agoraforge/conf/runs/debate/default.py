"""Vanilla debate baseline: N=32 claims, 2 debaters, zero-sum judge verdict."""

from agoraforge.conf.schema import run_config


def get_config():
    return run_config()
