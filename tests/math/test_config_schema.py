import json
from dataclasses import replace

import pytest

from agoraforge.conf.schema import build_env_config, run_config
from agoraforge.training.artifacts.config import (
    CONFIG_OVERRIDES_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    write_config_artifacts,
)


def test_runtime_config_validation_raises_value_error():
    cfg = run_config(env="math")
    env_cfg = build_env_config(cfg, level=cfg.levels[0])
    with pytest.raises(ValueError, match="rho must be > 0"):
        replace(env_cfg, rho=0.0)


def test_run_config_rejects_unknown_override():
    with pytest.raises(KeyError):
        run_config({"training.not_a_field": 1}, env="math")


def test_write_config_artifacts_records_resolved_config_and_overrides(tmp_path):
    cfg = run_config({"training.online_epochs": 3}, env="math")
    levels = {
        level["name"]: build_env_config(cfg, level=level)
        for level in cfg.levels
    }

    write_config_artifacts(
        str(tmp_path),
        cfg=cfg,
        levels=levels,
        overrides={"training.online_epochs": 3},
    )

    resolved = json.loads((tmp_path / RESOLVED_CONFIG_FILENAME).read_text())
    overrides = json.loads((tmp_path / CONFIG_OVERRIDES_FILENAME).read_text())
    assert resolved["config"]["training"]["online_epochs"] == 3
    assert resolved["levels"]["small_baseline"]["num_theorems"] == cfg.env.num_theorems
    assert overrides == {"training.online_epochs": 3}
