"""Training metric reductions and staged-buffer helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


@dataclass(frozen=True)
class EpochMetricSummary:
    """Epoch-level means: generic fields plus the env's eval metrics.

    ``overall`` holds one pooled mean per metric key (the env's eval metrics,
    and the env's derived metrics such as debate's ``acc_frac``);
    ``level_primary_mean`` tracks the env's primary
    metric per curriculum level.
    """

    overall_return: float
    overall: dict
    n_transitions_global: float
    level_return_mean: dict
    level_primary_mean: dict


def reduce_level_means(level_values, level_names):
    """Overall mean and per-level means."""
    sums = [float(sum(level_values[name])) for name in level_names]
    counts = [float(len(level_values[name])) for name in level_names]
    per_level = {
        name: sums[i] / counts[i]
        for i, name in enumerate(level_names)
        if counts[i] > 0
    }
    total = sum(counts)
    overall = sum(sums) / total if total > 0 else 0.0
    return overall, per_level


def summarize_epoch_metrics(level_metrics, level_names, staged, spec) -> EpochMetricSummary:
    overall_return, level_return_mean = reduce_level_means(
        level_metrics["rewards"], level_names)
    overall = {}
    level_primary_mean = {}
    for key in spec.metric_keys:
        overall[key], per_level = reduce_level_means(
            level_metrics[key], level_names)
        if key == spec.primary_metric:
            level_primary_mean = per_level
    overall.update(spec.derived_overall(overall))
    n_transitions_global = float(staged["advantages"].shape[0])
    return EpochMetricSummary(
        overall_return=overall_return,
        overall=overall,
        n_transitions_global=n_transitions_global,
        level_return_mean=level_return_mean,
        level_primary_mean=level_primary_mean,
    )


def write_epoch_scalars(
    writer,
    *,
    epoch: int,
    weights,
    levels,
    summary: EpochMetricSummary,
    spec,
    actor_loss: float,
    critic_loss: float,
    loss_terms: dict,
    advantage_mean: float,
    advantage_std: float,
) -> None:
    writer.add_scalar("return/mean", summary.overall_return, epoch)
    for key, tag in spec.writer_scalars.items():
        writer.add_scalar(tag, summary.overall[key], epoch)
    writer.add_scalar("training/actor_loss", actor_loss, epoch)
    writer.add_scalar("training/actor_advantage_loss_term", loss_terms["actor_advantage_loss"], epoch)
    writer.add_scalar("training/actor_kl_loss_term", loss_terms["actor_kl_loss"], epoch)
    writer.add_scalar("training/critic_loss", critic_loss, epoch)
    writer.add_scalar("training/kl_to_reference", loss_terms["kl_to_reference"], epoch)
    writer.add_scalar("training/parameters/kl_coeff", loss_terms["kl_coeff"], epoch)
    writer.add_scalar("training/parameters/actor_lr", loss_terms["actor_lr"], epoch)
    writer.add_scalar("training/parameters/critic_lr", loss_terms["critic_lr"], epoch)
    writer.add_scalar("training/advantage_raw_mean", advantage_mean, epoch)
    writer.add_scalar("training/advantage_raw_std", advantage_std, epoch)
    for name in (
        "approx_kl", "clip_fraction", "ratio_mean", "ratio_max",
        "critic_explained_variance",
    ):
        if math.isfinite(loss_terms[name]):
            writer.add_scalar(f"training/{name}", loss_terms[name], epoch)
    writer.add_scalar("training/parameters/n_transitions", summary.n_transitions_global, epoch)

    for name, ret in summary.level_return_mean.items():
        writer.add_scalar(f"bylevel/return/by_level_{name}", ret, epoch)
    for name in levels:
        writer.add_scalar(f"bylevel/levels/weight_{name}", weights.get(name, 0.0), epoch)
    for name, value in summary.level_primary_mean.items():
        writer.add_scalar(f"bylevel/{spec.primary_metric}/by_level_{name}", value, epoch)


def local_mean_std_tensor(values: torch.Tensor):
    local = _f64(values)
    count = torch.tensor(local.numel(), dtype=local.dtype, device=local.device)
    if float(count.item()) <= 0:
        return 0.0, 0.0, 0.0
    mean = local.sum() / count
    var = torch.clamp((local * local).sum() / count - mean * mean, min=0.0)
    return float(mean.cpu().item()), float(torch.sqrt(var).cpu().item()), float(count.cpu().item())


def normalize_advantages_local(advantages: torch.Tensor):
    mean, std, count = local_mean_std_tensor(advantages)
    if count <= 0:
        return advantages
    out = ((advantages.detach() - mean) / max(std, 1e-8)).to(advantages.dtype)
    return out


def normalize_advantages_tensor(advantages: torch.Tensor):
    mean, std, count = local_mean_std_tensor(advantages)
    if count <= 0:
        return advantages, 0.0, 0.0
    normalized = ((advantages.detach() - mean) / max(std, 1e-8)).to(advantages.dtype)
    return normalized, mean, std


def mean_std_tensor(values: torch.Tensor):
    return local_mean_std_tensor(values)


def concat_staged(buffers: list[dict]) -> dict:
    out = {}
    for key, value in buffers[0].items():
        if isinstance(value, dict):
            out[key] = {
                subkey: torch.cat([buf[key][subkey] for buf in buffers], dim=0)
                for subkey in value
            }
        elif isinstance(value, int):
            out[key] = sum(buf[key] for buf in buffers)
        else:
            out[key] = torch.cat([buf[key] for buf in buffers], dim=0)
    return out
