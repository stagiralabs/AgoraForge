"""Run-config composition."""

import pytest

from agoraforge.conf.schema import build_env_config, run_config


def test_default_levels_build():
    cfg = run_config()
    built = {
        level["name"]: build_env_config(cfg, level=level)
        for level in cfg.levels
    }
    assert built["small_vanilla"].control_mode == "vanilla"
    assert built["small_vanilla"].num_claims == 32


def test_unknown_override_rejected():
    with pytest.raises(KeyError):
        run_config({"env.not_a_field": 1})


def test_level_override_applies():
    cfg = run_config()
    level = {"name": "n16", "env": {"num_claims": 16, "judge_claim_cap": 7}}
    built = build_env_config(cfg, level=level)
    assert built.num_claims == 16
    assert built.judge_claim_cap == 7


def test_search_configs_load():
    from agoraforge.conf.runs.debate.protocol_search import base_config
    cfg = base_config()
    assert cfg.env.control_mode == "learned"
    assert cfg.search.pbest_frac == 0.25
    built = build_env_config(cfg, level=cfg.levels[0])
    assert built.control_mode == "learned"
    assert built.truth_gibbs_sweeps == 20
    assert built.marginal_gibbs_sweeps == 20
