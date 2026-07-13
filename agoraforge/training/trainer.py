"""Resident PPO epoch execution and loop control."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass

import torch

from agoraforge.training.artifacts.outputs import format_duration, harvest, save_model
from agoraforge.training.rollout import run_epoch_rollouts
from agoraforge.training.device import profile_synchronizer
from agoraforge.training.metrics import (
    concat_staged,
    mean_std_tensor,
    normalize_advantages_tensor,
    summarize_epoch_metrics,
    write_epoch_scalars,
)
from agoraforge.training.models import update_value_stats
from agoraforge.training.ppo import ppo_update_staged
from agoraforge.training.profiling import EpochProfiler
from agoraforge.training.curriculum import get_level_weights, sample_levels


_STOP_REQUESTED = False


def _request_stop(signum, frame):
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


class LoopController:
    def __init__(self, max_wall_clock_seconds: float):
        self.max_wall_clock_seconds = float(max_wall_clock_seconds)
        self.loop_start = time.perf_counter()
        self.last_epoch_duration = 0.0
        self.last_completed_epoch = -1

    def should_stop_before_epoch(self, epoch: int) -> bool:
        if _STOP_REQUESTED:
            print(f"Stop signal received; stopping after {epoch} completed epoch(s).",
                  flush=True)
            return True
        if self.max_wall_clock_seconds <= 0:
            return False
        elapsed = time.perf_counter() - self.loop_start
        if elapsed + self.last_epoch_duration <= self.max_wall_clock_seconds:
            return False
        print(f"Reached wall-clock cap ({self.max_wall_clock_seconds:.0f}s); "
              f"stopping after {epoch} completed epoch(s).", flush=True)
        return True

    def start_epoch(self) -> float:
        return time.perf_counter()

    def finish_epoch(self, epoch: int, epoch_start: float) -> None:
        self.last_completed_epoch = int(epoch)
        self.last_epoch_duration = time.perf_counter() - epoch_start

    def progress(self, epoch: int, online_epochs: int) -> tuple[float, float]:
        completed_epochs = epoch + 1
        remaining_epochs = online_epochs - epoch - 1
        elapsed = time.perf_counter() - self.loop_start
        avg_epoch_time = elapsed / max(completed_epochs, 1)
        return elapsed, avg_epoch_time * remaining_epochs


@dataclass
class EpochRunContext:
    spec: object
    targs: object
    buffer_size: int
    seed: int
    levels: dict
    schedule: object
    model: object
    optimizer: object
    model_cfg: object
    device: torch.device
    writer: object
    paths: object
    loop: object
    primary_tail_history: object
    checkpoint_every: int
    print_profile_breakdown: bool
    log_profile_to_tensorboard: bool
    profile_sync_enabled: bool
    gamma: float


def run_training_epoch(ctx: EpochRunContext, epoch: int, online_epochs: int) -> None:
    profiler = EpochProfiler(sync=profile_synchronizer(ctx.device, ctx.profile_sync_enabled))
    with profiler.section("setup"):
        ctx.model.train()

        weights = get_level_weights(ctx.schedule, epoch)
        sampled_names = sample_levels(weights, ctx.buffer_size)

    with profiler.section("rollout"):
        resident_staged_buffers, level_metrics = run_epoch_rollouts(
            levels=ctx.levels,
            spec=ctx.spec,
            sampled_names=sampled_names,
            buffer_size=ctx.buffer_size,
            seed=ctx.seed,
            epoch=epoch,
            model=ctx.model,
            device=ctx.device,
            gamma=ctx.gamma,
            gae_lambda=ctx.targs.gae_lambda,
            profiler=profiler,
        )

    with profiler.section("advantage_norm"):
        resident_staged = concat_staged(resident_staged_buffers)
        resident_staged["advantages"], advantage_mean, advantage_std = (
            normalize_advantages_tensor(resident_staged["advantages"])
        )
        return_mean, return_std, return_count = mean_std_tensor(resident_staged["returns"])
        if return_count > 0:
            update_value_stats(ctx.model, return_mean, return_std)

    with profiler.section("ppo_update"):
        actor_loss, critic_loss, kl_to_reference, loss_terms = ppo_update_staged(
            ctx.model, ctx.optimizer,
            resident_staged,
            ctx.targs, ctx.device, epoch,
            ctx.spec.policy.compute_log_probs_from_staged,
            profiler=profiler,
        )

    with profiler.section("metrics"):
        level_names = list(ctx.levels)
        metric_summary = summarize_epoch_metrics(
            level_metrics, level_names, resident_staged, ctx.spec)
        ctx.primary_tail_history.append(dict(metric_summary.level_primary_mean))

    with profiler.section("tensorboard"):
        write_epoch_scalars(
            ctx.writer,
            epoch=epoch,
            weights=weights,
            levels=ctx.levels,
            summary=metric_summary,
            spec=ctx.spec,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            loss_terms=loss_terms,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
        )

    elapsed, remaining = ctx.loop.progress(epoch, online_epochs)
    env_fields = " ".join(
        f"{label}={metric_summary.overall[key]:{fmt}}"
        for label, key, fmt in ctx.spec.console_items
    )
    print(
        f"Epoch {epoch + 1}/{online_epochs} "
        f"elapsed={format_duration(elapsed)} "
        f"remaining={format_duration(remaining)} "
        f"ret={metric_summary.overall_return:.3f} "
        + env_fields +
        f" actor_loss={actor_loss:.5f} "
        f"critic_loss={critic_loss:.5f} "
        f"kl_ref={kl_to_reference:.3f}",
        flush=True,
    )

    if ctx.log_profile_to_tensorboard:
        for name, value in profiler.scalar_items():
            ctx.writer.add_scalar(f"profile/{name}_s", value, epoch)

    if ctx.print_profile_breakdown:
        print(profiler.format_line(epoch=epoch, total_epochs=online_epochs), flush=True)

    if ctx.checkpoint_every > 0 and epoch > 0 and epoch % ctx.checkpoint_every == 0:
        save_model(ctx.model, ctx.model_cfg, epoch,
                   os.path.join(ctx.paths.save_dir, f"model_epoch_{epoch:06d}.pt"))

    harvest(ctx.paths.run_dir, ctx.paths.save_dir, ctx.model,
            ctx.model_cfg, epoch,
            ctx.primary_tail_history,
            list(ctx.levels), ctx.spec.primary_metric)
