"""PPO update from staged rollout buffers, generic over env action spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from agoraforge.training.device import autocast_context
from agoraforge.training.profiling import NullProfiler


def float_tree(value):
    """Cast floating tensors in a nested staged tree to FP32."""
    if isinstance(value, dict):
        return {key: float_tree(item) for key, item in value.items()}
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float()
    return value


@dataclass(frozen=True)
class PPOHparams:
    online_epochs: int
    online_ppo_epochs: int
    online_batch_size: int
    clip_eps: float
    actor_lr: float
    critic_lr: float
    lr_decay_epochs: float
    lr_decay_power: float
    lr_floor_frac: float
    starting_kl_coeff: float
    ending_kl_coeff: float
    kl_decay_epochs: float
    kl_decay_power: float
    grad_norm_clip: float
    gae_lambda: float
    critic_huber_beta: float
    ppo_diagnostics: bool = True
    amp_dtype: str = ""

    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        online_epochs: int | None = None,
        ppo_diagnostics: bool = True,
        amp_dtype: str = "",
    ) -> "PPOHparams":
        t = cfg.training
        return cls(
            online_epochs=int(t.online_epochs if online_epochs is None else online_epochs),
            online_ppo_epochs=int(t.online_ppo_epochs),
            online_batch_size=int(t.online_batch_size),
            clip_eps=float(t.clip_eps),
            actor_lr=float(t.actor_lr),
            critic_lr=float(t.critic_lr),
            lr_decay_epochs=float(t.lr_decay_epochs),
            lr_decay_power=float(t.lr_decay_power),
            lr_floor_frac=float(t.lr_floor_frac),
            starting_kl_coeff=float(t.starting_kl_coeff),
            ending_kl_coeff=float(t.ending_kl_coeff),
            kl_decay_epochs=float(t.kl_decay_epochs),
            kl_decay_power=float(t.kl_decay_power),
            grad_norm_clip=float(t.grad_norm_clip),
            gae_lambda=float(t.gae_lambda),
            critic_huber_beta=float(t.critic_huber_beta),
            ppo_diagnostics=bool(ppo_diagnostics),
            amp_dtype=str(amp_dtype or ""),
        )


@dataclass(frozen=True)
class PPOMinibatchLoss:
    loss: torch.Tensor
    actor_loss: torch.Tensor
    actor_advantage_loss: torch.Tensor
    actor_kl_loss: torch.Tensor
    critic_loss: torch.Tensor
    new_kls: torch.Tensor
    log_ratios: torch.Tensor
    ratios: torch.Tensor
    values: torch.Tensor
    returns_norm: torch.Tensor


def scheduled_kl_coefficient(training_cfg, epoch: int) -> float:
    start = training_cfg.starting_kl_coeff
    floor = training_cfg.ending_kl_coeff
    decay = (1.0 + epoch / training_cfg.kl_decay_epochs) ** training_cfg.kl_decay_power
    return floor + (start - floor) / decay


def scheduled_learning_rates(training_cfg, epoch: int) -> tuple[float, float]:
    decay = (1.0 + epoch / training_cfg.lr_decay_epochs) ** training_cfg.lr_decay_power
    scale = training_cfg.lr_floor_frac + (1.0 - training_cfg.lr_floor_frac) / decay
    return training_cfg.actor_lr * scale, training_cfg.critic_lr * scale


def index_staged(staged, agent_idx, env_idx=None, n_env=None):
    """Slice every staged tensor (incl. the nested obs dicts).

    The centralized single-agent setup is grouped by env: ``actor_obs``,
    and ``returns`` carry one row per env, while the action tensors,
    ``old_log_probs`` and ``advantages`` carry one row per action head (env x model).
    A leaf is env-granular iff its leading dim equals ``n_env``; it is then sliced by
    ``env_idx``, otherwise by ``agent_idx``. With no grouping (n_env == n_agent) the
    two index sets coincide, so every other arm takes the plain per-row path.
    """
    def pick(value):
        if env_idx is not None and n_env is not None and value.shape[0] == n_env:
            return env_idx
        return agent_idx

    out = {}
    for key, value in staged.items():
        if isinstance(value, dict):
            out[key] = {k: v[pick(v)] for k, v in value.items()}
        elif torch.is_tensor(value):
            out[key] = value[pick(value)]
    return out


def set_shared_lr(optimizer, actor_lr: float, critic_lr: float) -> None:
    """Set per-role learning rates on a role-tagged shared optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = critic_lr if group.get("role") == "critic" else actor_lr


def ppo_minibatch_loss(
    model,
    logits_dict,
    values,
    minibatch,
    logprob_fn,
    *,
    group: int,
    clip_eps: float,
    kl_coeff: float,
    critic_huber_beta: float,
) -> PPOMinibatchLoss:
    """Compute the shared resident/population PPO objective for one minibatch.

    ``group`` is one for decentralized and population policies. For the centralized
    policy, its A action heads form one joint action, so their log-probabilities and
    reference KLs are summed before forming one PPO ratio per environment.
    """
    head_log_probs, head_kls = logprob_fn(logits_dict, minibatch)
    new_log_probs = head_log_probs.view(-1, group).sum(dim=1)
    old_log_probs = minibatch["old_log_probs"].view(-1, group).sum(dim=1)
    new_kls = head_kls.view(-1, group).sum(dim=1)
    advantages = minibatch["advantages"]
    log_ratios = new_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_advantage_loss = -torch.min(surr1, surr2).mean()
    actor_kl_loss = kl_coeff * new_kls.mean()
    actor_loss = actor_advantage_loss + actor_kl_loss

    returns_norm = model.normalize_value(minibatch["returns"])
    critic_per_row = F.smooth_l1_loss(
        values,
        returns_norm,
        beta=critic_huber_beta,
        reduction="none",
    )
    critic_loss = critic_per_row.mean()
    return PPOMinibatchLoss(
        loss=actor_loss + critic_loss,
        actor_loss=actor_loss,
        actor_advantage_loss=actor_advantage_loss,
        actor_kl_loss=actor_kl_loss,
        critic_loss=critic_loss,
        new_kls=new_kls,
        log_ratios=log_ratios,
        ratios=ratios,
        values=values,
        returns_norm=returns_norm,
    )


def step_ppo_optimizer(model, optimizer, loss: torch.Tensor, grad_norm_clip: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    optimizer.step()


def ppo_update_staged(
    model, optimizer,
    staged,
    training_cfg, device, epoch, logprob_fn, profiler=None,
    generator=None,
    prebuild_graph: bool = False,
    return_stats: bool = True,
    profile_prefix: str = "",
):
    """Run PPO update from an already-resident tensor buffer.

    ``model`` is a shared-trunk actor/critic: one forward per minibatch yields
    the action logits and the value, and one backward drives both losses through
    the role-tagged ``optimizer``. Rollout has already produced pre-collated
    observations, action tensors, old log-probs, advantages, and returns on
    device, so the update indexes straight into staged tensors with no host-side
    staging step.
    """
    clip_eps = training_cfg.clip_eps
    grad_norm_clip = training_cfg.grad_norm_clip
    collect_diagnostics = training_cfg.ppo_diagnostics
    if profiler is None:
        profiler = NullProfiler()

    assert epoch <= training_cfg.online_epochs
    kl_coeff = scheduled_kl_coefficient(training_cfg, epoch)
    actor_lr, critic_lr = scheduled_learning_rates(training_cfg, epoch)
    set_shared_lr(optimizer, actor_lr, critic_lr)

    n_actor_rows = int(staged["actor_rows"])
    if n_actor_rows == 0:
        if return_stats:
            return 0.0, 0.0, 0.0, _empty_ppo_stats(kl_coeff, actor_lr, critic_lr)
        return None

    # Minibatch over actor rows. For the centralized actor that is one row per env
    # (a joint forward decodes all A models at once); each env row maps to A agent
    # rows of per-head actions/log-probs, while advantages/returns/critic stay
    # per-env. group == 1 for every other arm, collapsing to the per-row path. Row
    # order is env-major (env, head): env row e expands to agent rows e*group..+g-1.
    n_agent = int(staged["old_log_probs"].shape[0])
    group = max(1, n_agent // n_actor_rows)
    env_batch = max(1, training_cfg.online_batch_size // group)
    arange_group = torch.arange(group, device=device)

    if prebuild_graph and group != 1:
        raise ValueError("prebuilt PPO graphs require one policy row per action row")
    prebuilt = None
    if prebuild_graph:
        with profiler.section(f"ppo_{profile_prefix}prebuild_graph"):
            prebuilt = model.encoder.prebuild_graph(staged["actor_obs"])

    if return_stats:
        acc_actor_loss = torch.zeros((), device=device)
        acc_actor_advantage_loss = torch.zeros((), device=device)
        acc_actor_kl_loss = torch.zeros((), device=device)
        acc_critic_loss = torch.zeros((), device=device)
        acc_kl_to_reference = torch.zeros((), device=device)
        acc_approx_kl = torch.zeros((), device=device)
        acc_clip_fraction = torch.zeros((), device=device)
        acc_ratio_mean = torch.zeros((), device=device)
        acc_critic_ev = torch.zeros((), device=device)
        acc_max_ratio = torch.zeros((), device=device)
    metric_samples = 0
    n_batches = 0

    for _ in range(training_cfg.online_ppo_epochs):
        perm = torch.randperm(n_actor_rows, device=device, generator=generator)
        for start in range(0, n_actor_rows, env_batch):
            env_idx = perm[start:start + env_batch]
            with profiler.section("ppo_index"):
                if group == 1:
                    mb = index_staged(staged, env_idx)
                else:
                    agent_idx = (env_idx.unsqueeze(1) * group + arange_group).reshape(-1)
                    mb = index_staged(staged, agent_idx, env_idx=env_idx, n_env=n_actor_rows)
                if prebuilt is not None:
                    graph, bias, edge_features = prebuilt
                    mb["actor_obs"] = {
                        **mb["actor_obs"],
                        "_prebuilt_graph": (
                            graph[env_idx], bias[env_idx], edge_features[env_idx]),
                    }

            with profiler.section(f"ppo_{profile_prefix}forward"):
                with autocast_context(device, training_cfg.amp_dtype):
                    logits_dict, values = model(
                        mb["actor_obs"], mode="actor_critic",
                        profiler=profiler, profile_prefix=f"ppo_{profile_prefix}")
                logits_dict = float_tree(logits_dict)
                values = values.float()

            with profiler.section(f"ppo_{profile_prefix}loss"):
                losses = ppo_minibatch_loss(
                    model,
                    logits_dict,
                    values,
                    mb,
                    logprob_fn,
                    group=group,
                    clip_eps=clip_eps,
                    kl_coeff=kl_coeff,
                    critic_huber_beta=training_cfg.critic_huber_beta,
                )

            if return_stats and collect_diagnostics:
                with profiler.section("ppo_metrics"), torch.no_grad():
                    approx_kl = ((losses.ratios - 1.0) - losses.log_ratios).mean()
                    clip_fraction = (torch.abs(losses.ratios - 1.0) > clip_eps).float().mean()
                    explained_variance = _explained_variance(
                        losses.values, losses.returns_norm)

            with profiler.section(f"ppo_{profile_prefix}backward"):
                step_ppo_optimizer(model, optimizer, losses.loss, grad_norm_clip)

            if return_stats:
                with profiler.section("ppo_metrics"):
                    batch_size = env_idx.shape[0]
                    acc_actor_loss = acc_actor_loss + losses.actor_loss.detach()
                    acc_actor_advantage_loss = acc_actor_advantage_loss + losses.actor_advantage_loss.detach()
                    acc_actor_kl_loss = acc_actor_kl_loss + losses.actor_kl_loss.detach()
                    acc_critic_loss = acc_critic_loss + losses.critic_loss.detach()
                    acc_kl_to_reference = acc_kl_to_reference + losses.new_kls.mean().detach()
                    if collect_diagnostics:
                        acc_approx_kl = acc_approx_kl + approx_kl * batch_size
                        acc_clip_fraction = acc_clip_fraction + clip_fraction * batch_size
                        acc_ratio_mean = acc_ratio_mean + losses.ratios.mean().detach() * batch_size
                        acc_critic_ev = acc_critic_ev + explained_variance * batch_size
                        acc_max_ratio = torch.maximum(acc_max_ratio, losses.ratios.max().detach())
                    metric_samples += batch_size
                    n_batches += 1

    if not return_stats:
        return None

    if n_batches == 0:
        return 0.0, 0.0, 0.0, _empty_ppo_stats(kl_coeff, actor_lr, critic_lr)

    (total_actor_loss, total_actor_advantage_loss, total_actor_kl_loss, total_critic_loss,
     total_kl_to_reference, total_approx_kl, total_clip_fraction, total_ratio_mean,
     total_critic_explained_variance, max_ratio) = torch.stack([
        acc_actor_loss, acc_actor_advantage_loss, acc_actor_kl_loss, acc_critic_loss,
        acc_kl_to_reference, acc_approx_kl, acc_clip_fraction, acc_ratio_mean,
        acc_critic_ev, acc_max_ratio,
    ]).cpu().tolist()

    metric_denominator = max(metric_samples, 1)
    return (
        total_actor_loss / n_batches,
        total_critic_loss / n_batches,
        total_kl_to_reference / n_batches,
        {
            "actor_advantage_loss": total_actor_advantage_loss / n_batches,
            "actor_kl_loss": total_actor_kl_loss / n_batches,
            # kl_to_reference = KL(policy‖uniform/base): the entropy regularizer
            # kl_coeff penalizes (rises as the policy sharpens). NOT the trust
            # region. approx_kl below is the PPO step KL(old‖new) the clip bounds.
            "kl_to_reference": total_kl_to_reference / n_batches,
            "kl_coeff": kl_coeff,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "approx_kl": total_approx_kl / metric_denominator if collect_diagnostics else float("nan"),
            "clip_fraction": total_clip_fraction / metric_denominator if collect_diagnostics else float("nan"),
            "ratio_mean": total_ratio_mean / metric_denominator if collect_diagnostics else float("nan"),
            "ratio_max": max_ratio if collect_diagnostics else float("nan"),
            "critic_explained_variance": (
                total_critic_explained_variance / metric_denominator if collect_diagnostics else float("nan")),
        },
    )


def _empty_ppo_stats(kl_coeff: float, actor_lr: float, critic_lr: float) -> dict:
    return {
        "actor_advantage_loss": 0.0,
        "actor_kl_loss": 0.0,
        "kl_to_reference": 0.0,
        "kl_coeff": kl_coeff,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_mean": 0.0,
        "ratio_max": 0.0,
        "critic_explained_variance": 0.0,
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return_variance = torch.var(returns, unbiased=False)
    if return_variance <= 1e-8:
        return torch.zeros((), dtype=returns.dtype, device=returns.device)
    error_variance = torch.var(returns - values, unbiased=False)
    return 1.0 - error_variance / return_variance
