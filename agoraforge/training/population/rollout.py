"""Population rollout: K candidate mechanisms, K independent policies.

The candidates share one batched env of ``B = K * B_base`` envs on COMMON
RANDOM NUMBERS (the env's per-step stochasticity is drawn once at the base ``B_base``
and tiled across the K blocks) -- the right variance reduction for *ranking*
protocols. The K actor/critic weight sets are stacked and the param-dependent forward
is ``vmap``-ed over them, so candidate ``k``'s block of observations is scored by
policy ``k`` in one batched forward. The graph build is param-free and its in-place
tensor ops don't vectorise under vmap, so it runs once over the full ``B*A`` batch and
only the encode/attention/head core is vmapped (the ``_prebuilt_graph`` hook in
``models.graph_transformer``).
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from agoraforge.training.device import autocast_context
from agoraforge.training.population.models import (
    shared_actor_critic_outputs_from_state,
    stack_population_actor_state,
)
from agoraforge.training.rollout import finalize_rollout_history
from agoraforge.training.ppo import float_tree


def _prof_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def rollout_population(
    cfg,
    spec,
    *,
    base_batch_size: int,
    thetas,
    models,
    action_generator,
    action_generators=None,
    seeds,
    device,
    gamma: float,
    gae_lambda: float,
    step_seed: int = 0,
    profiler=None,
    amp_dtype: str = "",
    base_instance=None,
):
    """One rollout stepping K candidate protocol mechanisms, each scored by its own
    policy.

    ``models`` are K module instances (candidate ``k`` uses index ``k``). Returns
    ``(staged_per_candidate, metrics)`` where ``staged_per_candidate`` is a length-K
    list of single-candidate PPO staging dicts (each over ``B_base`` envs) and
    ``metrics`` holds the per-candidate environment metrics.
    """
    K = len(thetas)
    sample_actions = spec.policy.sample_actions
    batch = spec.batch_cls.batched_from_config(
        cfg, base_batch_size=base_batch_size, thetas=thetas,
        device=device, seeds=seeds, step_seed=step_seed, profiler=profiler,
        **({"instance": base_instance} if base_instance is not None else {}),
    )
    T = int(cfg.max_timestep)
    A = batch.A
    block = base_batch_size * A

    # Per-candidate staging histories.
    a_obs = [[] for _ in range(K)]
    act_h = [[] for _ in range(K)]
    rew_h = [[] for _ in range(K)]
    val_h = [[] for _ in range(K)]
    episode_rewards = torch.zeros((batch.B, A), dtype=torch.float32, device=device)
    # Policy weights stay fixed for the entire rollout.  Snapshot the population
    # once instead of re-stacking every parameter on every environment timestep.
    actor_state = stack_population_actor_state(models)

    with torch.no_grad():
        for _ in range(T):
            with _prof_section(profiler, "pop_rollout_observe"):
                inputs = batch.policy_inputs()
                actor_obs = inputs.actor_obs
                masks = inputs.masks

            with _prof_section(profiler, "pop_rollout_actor_forward"):
                with autocast_context(device, amp_dtype):
                    logits, values = shared_actor_critic_outputs_from_state(
                        models[0], actor_state, actor_obs, K,
                    )
                logits = float_tree(logits)
                values = values.float()
            values_ba = values.reshape(batch.B, A)                 # block-major -> (B, A)
            # One full folded-batch copy per observation tensor replaces K slice
            # copies (same stored bytes, fewer accelerator launches).
            saved_actor_obs = {key: value.clone() for key, value in actor_obs.items()}

            with _prof_section(profiler, "pop_rollout_action_sample"):
                if action_generators is None:
                    shim = SimpleNamespace(B=batch.B, A=A, device=device, cfg=cfg)
                    full_actions, staged_full = sample_actions(
                        _flatten_logits(logits), masks, shim, action_generator
                    )
                    staged_split = _split_staged(staged_full, K)
                else:
                    # Draw once and broadcast common random numbers across all
                    # candidates; policy logits and masks still yield distinct actions.
                    shim = SimpleNamespace(B=batch.B, A=A, device=device, cfg=cfg)
                    full_actions, staged_full = sample_actions(
                        _flatten_logits(logits), masks, shim,
                        action_generators[0], crn_blocks=K,
                    )
                    staged_split = _split_staged(staged_full, K)
                full_actions = spec.policy.to_env_actions(
                    full_actions, inputs.ctx, batch,
                )
                for k in range(K):
                    sl = slice(k * block, (k + 1) * block)
                    a_obs[k].append({key: value[sl] for key, value in saved_actor_obs.items()})
                    act_h[k].append(staged_split[k])
            with _prof_section(profiler, "pop_rollout_env_step"):
                rewards = batch.step(full_actions)
                rewards = rewards.squeeze(-1)  # (B, A)

            with _prof_section(profiler, "pop_rollout_bookkeeping"):
                episode_rewards += rewards
                for k in range(K):
                    bb = slice(k * base_batch_size, (k + 1) * base_batch_size)
                    rew_h[k].append(rewards[bb].detach())
                    val_h[k].append(values_ba[bb].detach())

    if action_generators is not None and len(action_generators) > 1:
        # The sampler advances generator 0; synchronize the identical CRN streams
        # before candidate-local PPO minibatch draws.
        generator_state = action_generators[0].get_state()
        for generator in action_generators[1:]:
            generator.set_state(generator_state)

    with _prof_section(profiler, "pop_rollout_finalize"):
        staged_per_candidate = [
            finalize_rollout_history(
                a_obs[k], act_h[k], rew_h[k], val_h[k],
                gamma, gae_lambda,
            )
            for k in range(K)
        ]

    def per_cand(t):
        return t.reshape(K, base_batch_size)

    metrics = {
        key: per_cand(value.detach()).mean(dim=1)
        for key, value in batch.eval_metrics().items()
    }
    metrics["episode_rewards"] = per_cand(
        episode_rewards.mean(dim=1).detach()).mean(dim=1)
    return staged_per_candidate, metrics


def _flatten_logits(logits: dict) -> dict:
    """Flatten [K, block, ...] vmapped logits to [K*block, ...]."""
    out = {}
    for key, value in logits.items():
        if isinstance(value, dict):
            out[key] = _flatten_logits(value)
        elif torch.is_tensor(value) and value.dim() >= 2:
            out[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        else:
            out[key] = value
    return out


def _split_staged(staged: dict, K: int) -> list[dict]:
    """Split full folded-batch staged action tensors into per-candidate dicts."""
    out = [{} for _ in range(K)]
    for key, value in staged.items():
        if isinstance(value, dict):
            split = _split_staged(value, K)
            for k in range(K):
                out[k][key] = split[k]
        elif torch.is_tensor(value):
            chunks = value.reshape(K, value.shape[0] // K, *value.shape[1:])
            for k in range(K):
                out[k][key] = chunks[k]
        else:
            for k in range(K):
                out[k][key] = value
    return out
