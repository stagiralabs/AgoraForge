"""Math EnvSpec registration."""

from __future__ import annotations

from agoraforge.envs.registry import EnvSpec

from agoraforge.envs.math import mechanism, model, policy, presets
from agoraforge.envs.math.config import MathConfig
from agoraforge.envs.math.env import BatchedMathEnv

SPEC = EnvSpec(
    name="math",
    config_cls=MathConfig,
    batch_cls=BatchedMathEnv,
    policy=policy,
    model=model,
    mechanism=mechanism,
    presets=presets,
    metric_keys=(
        "resolved_frac",
        "discounted_resolved",
    ),
    primary_metric="discounted_resolved",
    writer_scalars={
        "resolved_frac": "resolved/frac",
        "discounted_resolved": "resolved/discounted_resolved",
    },
    console_items=(
        ("resolved", "resolved_frac", ".1%"),
        ("discounted_resolved", "discounted_resolved", ".1%"),
    ),
)
