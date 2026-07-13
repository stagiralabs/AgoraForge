"""Environment specifications and lookup.

Everything generic (training loop, PPO, population evaluation, outer search,
model construction, metric reporting) reaches an environment only through its
spec. The two shipped specs are imported lazily to keep module imports acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from types import ModuleType
from typing import Callable, Mapping


def _no_derived_metrics(_: Mapping[str, float]) -> dict[str, float]:
    return {}


@dataclass(frozen=True)
class EnvSpec:
    name: str
    # The env config dataclass and the BatchedEnv implementation.
    config_cls: type
    batch_cls: type
    # envs.<name>.policy: sample_actions / to_env_actions /
    # compute_log_probs_from_staged.
    policy: ModuleType
    # envs.<name>.model: build_model_config / build_actor /
    # build_shared plus the env's encoder, head, and edge constants.
    model: ModuleType
    # The env's mechanism module (mechanism/protocol): mechanism_from_config /
    # theta_from_config / mechanism_dims_from_config / mechanism_state_dims plus
    # LEARNED_MECHANISM, the layout+init spec class the outer loop instantiates.
    mechanism: ModuleType
    # envs.<name>.presets: configuration groups and the run-config bridge.
    presets: ModuleType
    # Keys of batch.eval_metrics(), reported per level each epoch.
    metric_keys: tuple[str, ...]
    # The headline metric: per-level tracked, tail-averaged, harvested to
    # level_curve.json.
    primary_metric: str
    # Derived overall metrics computed from the epoch's overall means
    # (e.g. debate's acc_frac from acc/prior/full).
    derived_overall: Callable[[Mapping[str, float]], dict[str, float]] = (
        _no_derived_metrics
    )
    # metric key -> tensorboard/W&B scalar tag.
    writer_scalars: Mapping[str, str] = field(default_factory=dict)
    # (label, metric key, format spec) triples for the per-epoch console line.
    console_items: tuple[tuple[str, str, str], ...] = ()

    @property
    def default_level(self) -> str:
        return self.presets.DEFAULT_LEVEL

    def build_env_config(self, cfg, *, level):
        return self.presets.build_env_config(cfg, level=level)


@cache
def _registry() -> dict[str, EnvSpec]:
    from agoraforge.envs.debate.spec import SPEC as debate
    from agoraforge.envs.math.spec import SPEC as math

    specs = (debate, math)
    registry = {spec.name: spec for spec in specs}
    if len(registry) != len(specs):
        raise ValueError("environment names must be unique")
    return registry


def get_env(name: str) -> EnvSpec:
    registry = _registry()
    try:
        return registry[name]
    except KeyError:
        raise KeyError(
            f"unknown environment {name!r}; available: {sorted(registry)}"
        ) from None


def available() -> list[str]:
    return sorted(_registry())
