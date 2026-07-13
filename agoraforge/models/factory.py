"""Registry-dispatched model configuration and construction."""

from __future__ import annotations

from agoraforge.models.graph_transformer import GraphTransformerConfig


def build_model_config(model_cfg_input, env_cfg, decoding_cfg=None) -> GraphTransformerConfig:
    """Build the trunk config through the env's model module (sizes node features
    to the env's obs schema and mechanism state width)."""
    from agoraforge.envs.registry import get_env
    return get_env(env_cfg.env_name).model.build_model_config(
        model_cfg_input, env_cfg, decoding_cfg)


def build_actor(config: GraphTransformerConfig):
    from agoraforge.envs.registry import get_env
    return get_env(config.env_name).model.build_actor(config)


def build_shared(config: GraphTransformerConfig):
    from agoraforge.envs.registry import get_env
    return get_env(config.env_name).model.build_shared(config)
