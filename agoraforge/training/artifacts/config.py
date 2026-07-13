"""Self-describing configuration artifacts for training runs."""

from __future__ import annotations

import json
import os

RESOLVED_CONFIG_FILENAME = "resolved_config.json"
CONFIG_OVERRIDES_FILENAME = "config_overrides.json"


def serialize_config(cfg) -> dict:
    return {"env_name": cfg.env_name, **cfg.to_json_dict()}


def write_config_artifacts(run_dir: str, *, cfg, levels: dict, overrides: dict) -> None:
    resolved = {
        "config": _jsonable(cfg.to_dict()),
        "levels": {name: serialize_config(vcfg) for name, vcfg in levels.items()},
    }
    with open(os.path.join(run_dir, RESOLVED_CONFIG_FILENAME), "w") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, CONFIG_OVERRIDES_FILENAME), "w") as f:
        json.dump(_jsonable(overrides), f, indent=2, sort_keys=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
