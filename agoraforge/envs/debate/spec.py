"""Debate EnvSpec registration."""

from __future__ import annotations

from agoraforge.envs.registry import EnvSpec

from agoraforge.envs.debate import model, policy, presets, protocol
from agoraforge.envs.debate.config import DebateConfig
from agoraforge.envs.debate.env import BatchedDebateEnv


def _derived_overall(overall) -> dict:
    """Protocol accuracy as a fraction of the prior->full-graph anchor span."""
    span = overall["judge_acc_full"] - overall["judge_acc_prior"]
    if abs(span) < 1e-9:
        return {"acc_frac": 0.0}
    return {"acc_frac": (overall["judge_acc"] - overall["judge_acc_prior"]) / span}


SPEC = EnvSpec(
    name="debate",
    config_cls=DebateConfig,
    batch_cls=BatchedDebateEnv,
    policy=policy,
    model=model,
    mechanism=protocol,
    presets=presets,
    metric_keys=(
        "judge_acc",
        "judge_acc_prior",
        "judge_acc_full",
    ),
    primary_metric="judge_acc",
    derived_overall=_derived_overall,
    writer_scalars={
        "judge_acc": "judge/acc",
        "judge_acc_prior": "judge/acc_prior",
        "judge_acc_full": "judge/acc_full",
        "acc_frac": "judge/acc_frac",
    },
    console_items=(
        ("judge_acc", "judge_acc", ".4f"),
        ("prior", "judge_acc_prior", ".4f"),
        ("full", "judge_acc_full", ".4f"),
        ("acc_frac", "acc_frac", ".1%"),
    ),
)
