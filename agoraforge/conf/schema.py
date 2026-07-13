"""Config construction helpers with validated overrides.

A run config = shared training/model/logging groups + the selected environment's
own groups (installed from ``envs/<name>/presets.py``). Every run config names its
env in ``cfg.env_name``; the env's presets module bridges the run config into its
env config dataclass per curriculum level.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from collections.abc import Mapping

from ml_collections import ConfigDict

from agoraforge.conf.training import logging, model, ppo


def load_run_config(config_path: str):
    """Load a run config from a Python file path or importable module name."""
    path = Path(config_path)
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_config()
    return importlib.import_module(config_path).get_config()


def select_level(run, level_name: str):
    """Select a named run-config level, defaulting to the first."""
    if not level_name:
        return run.levels[0]
    try:
        return next(level for level in run.levels if level["name"] == level_name)
    except StopIteration:
        available = [level["name"] for level in run.levels]
        raise ValueError(
            f"unknown level {level_name!r}; available: {available}"
        ) from None


def base_run_config(env_name: str = "debate") -> ConfigDict:
    from agoraforge.envs.registry import get_env
    spec = get_env(env_name)

    cfg = ConfigDict()
    cfg.env_name = env_name
    for group_name, group_cfg in spec.presets.base_groups().items():
        setattr(cfg, group_name, group_cfg)
    cfg.actor_model = model.small_baseline()
    cfg.training = ppo.default()
    # Env-specific training regime defaults layered over the shared PPO defaults.
    for key, value in getattr(spec.presets, "TRAINING_OVERRIDES", {}).items():
        cfg.training[key] = value
    cfg.logging = logging.default()

    cfg.levels = [
        {"name": spec.default_level},
    ]
    cfg.schedule = [
        {"epoch": 0, "weights": {spec.default_level: 1.0}},
    ]

    cfg.seed = 42
    cfg.results_dir = "results"
    cfg.device = "cuda"
    cfg.max_wall_clock_seconds = 0.0
    return cfg


def run_config(overrides: Mapping[str, object] | None = None, *, env: str = "debate") -> ConfigDict:
    cfg = base_run_config(env)
    apply_overrides(cfg, overrides or {})
    return cfg


def apply_overrides(cfg: ConfigDict, overrides: Mapping[str, object]) -> ConfigDict:
    for dotted_key, value in overrides.items():
        _set_dotted(cfg, dotted_key, value)
    return cfg


def build_env_config(cfg, *, level):
    """Build the env config dataclass for one curriculum level."""
    from agoraforge.envs.registry import get_env
    return get_env(cfg.env_name).build_env_config(cfg, level=level)


def validate_level_keys(level, groups) -> None:
    unknown = set(level) - {"name", *groups}
    if unknown:
        raise KeyError(
            f"Level {level['name']!r} has unknown keys {sorted(unknown)}; "
            f"per-level overrides are limited to {list(groups)}"
        )


def resolve_level_group(cfg, level, group):
    merged = copy.deepcopy(cfg[group])
    for key, value in level.get(group, {}).items():
        if key not in merged:
            raise KeyError(
                f"Level {level['name']!r} overrides unknown {group} param {key!r}"
            )
        merged[key] = value
    return merged


def _set_dotted(cfg, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise KeyError(f"invalid override key {dotted_key!r}")
    target = cfg
    for part in parts[:-1]:
        target = _child(target, part, dotted_key)
    leaf = parts[-1]
    if isinstance(target, ConfigDict):
        if leaf not in target:
            raise KeyError(f"unknown config override {dotted_key!r}")
        target[leaf] = _copy_value(value)
        return
    if isinstance(target, dict):
        if leaf not in target:
            raise KeyError(f"unknown config override {dotted_key!r}")
        target[leaf] = _copy_value(value)
        return
    raise KeyError(f"cannot override {dotted_key!r} through {type(target).__name__}")


def _child(target, part: str, dotted_key: str):
    if isinstance(target, ConfigDict):
        if part not in target:
            raise KeyError(f"unknown config override {dotted_key!r}")
        return target[part]
    if isinstance(target, dict):
        if part not in target:
            raise KeyError(f"unknown config override {dotted_key!r}")
        return target[part]
    raise KeyError(f"cannot override {dotted_key!r} through {type(target).__name__}")


def _copy_value(value):
    return copy.deepcopy(value)
