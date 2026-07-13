"""Stacked population policy models."""

from __future__ import annotations

import torch
from torch.func import functional_call, stack_module_state, vmap

from agoraforge.models.graph_transformer import build_population_edge_bundle
from agoraforge.models.factory import build_model_config, build_shared


def build_population_models(
    cfg, vcfg, device, K: int, init_seed_base: int, crn: bool = False,
):
    actor_cfg = build_model_config(cfg.actor_model, vcfg, cfg.decoding)
    models = []
    for k in range(K):
        offset = 0 if crn else k
        torch.manual_seed(init_seed_base + int(offset))
        models.append(build_shared(actor_cfg).to(device))
    return models, actor_cfg


def stack_population_actor_state(models):
    """Snapshot population parameters once for a no-grad rollout.

    ``stack_module_state`` materializes every candidate parameter and buffer.  A
    rollout cannot update policy weights, so rebuilding that snapshot at every
    environment timestep is pure host work and device traffic.
    """
    return stack_module_state(models)


def shared_actor_critic_outputs_from_state(
    ref, state, obs: dict, K: int,
) -> tuple[dict, torch.Tensor]:
    """Population actor/critic forward from a pre-stacked parameter snapshot."""
    obs_v, edge_bundle = _actor_obs_for_vmap(ref, obs, K)
    params, buffers = state

    def fwd(p, b, ob, bundle=None):
        if bundle is not None:
            ob = {**ob, "_prebuilt_edge_bundle": bundle}
        logits, values = functional_call(ref, (p, b), (ob, None, "", "actor_critic"))
        logits = {k: v for k, v in logits.items() if not isinstance(v, float)}
        return logits, values

    if edge_bundle is None:
        logits, norm = vmap(fwd)(params, buffers, obs_v)
    else:
        logits, norm = vmap(fwd)(params, buffers, obs_v, edge_bundle)
    vmean = _stacked_buffer(buffers, "value_mean").reshape(K, 1)
    vstd = _stacked_buffer(buffers, "value_std").reshape(K, 1)
    return logits, norm * vstd + vmean


def _actor_obs_for_vmap(ref, obs: dict, K: int):
    graph, bias, ef = ref.encoder.prebuild_graph(obs)
    obs_v = {k: _reshape_block(v, K) for k, v in obs.items()}
    graph_v = _reshape_block(graph, K)
    bias_v = _reshape_block(bias, K)
    ef_v = _reshape_block(ef, K)
    obs_v["_prebuilt_graph"] = (graph_v, bias_v, ef_v)
    attention_mode = ref._resolve_attention_mode(graph.shape[-1])
    edge_bundle = (
        build_population_edge_bundle(graph_v, ef_v, bias_v)
        if attention_mode == "sparse" else None
    )
    return obs_v, edge_bundle


def _reshape_block(t: torch.Tensor, K: int) -> torch.Tensor:
    return t.reshape(K, t.shape[0] // K, *t.shape[1:])


def _stacked_buffer(buffers: dict, name: str) -> torch.Tensor:
    if name in buffers:
        return buffers[name]
    suffix = "." + name
    matches = [v for k, v in buffers.items() if k.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(name)
