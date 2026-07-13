"""Shared model configuration and construction contracts."""

import pytest
import torch

from agoraforge.conf.runs.math.centralized import get_config as centralized_config
from agoraforge.conf.schema import build_env_config, run_config
from agoraforge.models.factory import build_model_config
from agoraforge.models.graph_transformer import GraphTransformerConfig
from agoraforge.training.models import build_training_models


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_embd": 7, "n_head": 2}, "divisible"),
        ({"n_layer": 0}, "must be positive"),
        ({"init_std": 0.0}, "init_std must be positive"),
        ({"attention_mode": "magic"}, "attention_mode"),
        ({"sparse_attention_min_nodes": 0}, "must be positive"),
    ],
)
def test_model_config_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        GraphTransformerConfig(env_name="math", **overrides)


def test_model_config_roundtrip():
    cfg = GraphTransformerConfig(
        env_name="debate",
        n_embd=12,
        n_head=3,
        attention_mode="sparse",
        head={"nested": {"value": 4}},
    )
    assert GraphTransformerConfig.from_dict(cfg.to_dict()) == cfg


def test_environment_builders_resolve_attention_mode():
    debate_run = run_config()
    debate_env = build_env_config(debate_run, level=debate_run.levels[0])
    debate_model = build_model_config(
        debate_run.actor_model, debate_env, debate_run.decoding,
    )
    assert debate_model.attention_mode == "dense"

    math_run = run_config(env="math")
    math_env = build_env_config(math_run, level=math_run.levels[0])
    math_model = build_model_config(math_run.actor_model, math_env, math_run.decoding)
    assert math_model.attention_mode == "dense"

    centralized_run = centralized_config()
    centralized_env = build_env_config(
        centralized_run, level=centralized_run.levels[0],
    )
    centralized_model = build_model_config(
        centralized_run.actor_model, centralized_env, centralized_run.decoding,
    )
    assert centralized_model.attention_mode == "auto"


def test_training_builds_shared_actor_critic():
    run = run_config()
    env_cfg = build_env_config(run, level=run.levels[0])
    bundle = build_training_models(
        run,
        {run.levels[0]["name"]: env_cfg},
        device=torch.device("cpu"),
    )
    assert bundle.model.value_head is not None
