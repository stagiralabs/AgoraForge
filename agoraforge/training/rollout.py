"""Resident-GPU rollout path, generic over the environment registry."""

from __future__ import annotations

import hashlib

import torch



def tensor_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    """Fixed-horizon generalized advantage estimation without bootstrapping."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(values[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + float(gamma) * next_value - values[t]
        last_gae = delta + float(gamma) * float(lam) * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)


def _cat_obs(history: list[dict]) -> dict:
    keys = history[0].keys()
    return {key: torch.cat([obs[key] for obs in history], dim=0).detach() for key in keys}


def _cat_action_history(history: list[dict]) -> dict:
    out = {}
    for key, value in history[0].items():
        if isinstance(value, dict):
            out[key] = {
                subkey: torch.cat([h[key][subkey] for h in history], dim=0).detach()
                for subkey in value
            }
        else:
            out[key] = torch.cat([h[key] for h in history], dim=0).detach()
    return out


def finalize_rollout_history(
    actor_obs_h,
    action_h,
    reward_h,
    value_h,
    gamma,
    gae_lambda,
):
    """Turn a resident or candidate rollout history into one PPO staging buffer."""
    rewards_t = torch.stack(reward_h, dim=0)
    values_t = torch.stack(value_h, dim=0)
    advantages, returns = tensor_gae(rewards_t, values_t, gamma, gae_lambda)
    staged = _cat_action_history(action_h)
    staged["actor_obs"] = _cat_obs(actor_obs_h)
    staged["advantages"] = advantages.detach()
    staged["returns"] = returns.detach()
    staged["actor_rows"] = int(next(iter(staged["actor_obs"].values())).shape[0])
    return staged


def _rollout_loop(batch, spec, T, model, device, gamma, gae_lambda, profiler):
    """Shared device-resident rollout loop returning staged PPO tensors + metrics."""
    policy = spec.policy
    actor_obs_h, action_h = [], []
    reward_h, value_h = [], []
    episode_rewards = torch.zeros((batch.B, batch.A), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(T):
            with profiler.section("rollout_observe"):
                inputs = batch.policy_inputs()
            with profiler.section("rollout_forward"):
                # One shared-trunk forward yields the action logits and the value.
                # Centralized is single-agent: one team value per env (B,); other
                # modes keep a per-agent value (B, A).
                logits, values = model(inputs.actor_obs, mode="actor_critic")
                values = model.denormalize_value(values)
                if not batch.centralized:
                    values = values.reshape(batch.B, batch.A)
            with profiler.section("rollout_action_sample"):
                actions, staged_actions = policy.sample_actions(logits, inputs.masks, batch)
                actions = policy.to_env_actions(actions, inputs.ctx, batch)
            with profiler.section("rollout_env_step"):
                rewards = batch.step(actions)
                rewards = rewards.squeeze(-1)  # (B, A), team-identical when centralized

            actor_obs_h.append(inputs.actor_obs)
            action_h.append(staged_actions)
            reward_h.append((rewards[:, 0] if batch.centralized else rewards).detach())
            value_h.append(values.detach())
            episode_rewards += rewards

    staged = finalize_rollout_history(
        actor_obs_h,
        action_h,
        reward_h,
        value_h,
        gamma,
        gae_lambda,
    )

    metrics = {
        "episode_rewards": episode_rewards.detach(),
    }
    metrics.update({k: v.detach() for k, v in batch.eval_metrics().items()})
    return staged, metrics


def rollout_resident_config(
    cfg,
    spec,
    *,
    batch_size: int,
    seeds,
    model,
    device,
    gamma: float,
    gae_lambda: float,
    profiler,
):
    """Run a resident rollout group without constructing Python envs."""
    batch = spec.batch_cls.from_config(
        cfg,
        batch_size=batch_size,
        device=device,
        seeds=seeds,
        profiler=profiler,
    )
    T = int(cfg.max_timestep)
    return _rollout_loop(batch, spec, T, model, device, gamma, gae_lambda, profiler)


def stable_level_offset(name: str, modulo: int = 10000) -> int:
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % modulo


def run_epoch_rollouts(
    *,
    levels: dict,
    spec,
    sampled_names,
    buffer_size: int,
    seed: int,
    epoch: int,
    model,
    device,
    gamma: float,
    gae_lambda: float,
    profiler,
) -> tuple[list[dict], dict]:
    """Group sampled curriculum levels into batched resident rollouts."""
    metrics = {
        key: {name: [] for name in levels}
        for key in ("rewards", *spec.metric_keys)
    }
    level_indices = {name: 0 for name in levels}
    groups = {}
    for name in sampled_names:
        index = level_indices[name]
        level_indices[name] += 1
        rollout_seed = (
            seed + 2000 + stable_level_offset(name) + epoch * buffer_size + index
        )
        groups.setdefault(name, []).append(rollout_seed)

    staged_buffers = []
    for name, rollout_seeds in groups.items():
        staged, resident_metrics = rollout_resident_config(
            levels[name], spec,
            batch_size=len(rollout_seeds),
            seeds=rollout_seeds,
            model=model,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            profiler=profiler,
        )
        staged_buffers.append(staged)
        metrics["rewards"][name].extend(
            resident_metrics["episode_rewards"].mean(dim=1).detach().cpu().tolist()
        )
        for key in spec.metric_keys:
            metrics[key][name].extend(resident_metrics[key].detach().cpu().tolist())
    return staged_buffers, metrics
