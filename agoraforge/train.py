"""PPO training over a schedule of difficulty levels, with graph transformers.

Usage:
    agoraforge-train --name=my-run [--config=agoraforge/conf/runs/debate/default.py]
    (or python -m agoraforge.train; config defaults to default.py)
"""

from __future__ import annotations

import os

from collections import deque

import torch
from absl import app, flags
from ml_collections import ConfigDict, config_flags

from agoraforge.training.ppo import PPOHparams
from agoraforge.training.device import (
    configure_cuda_allocator, enable_determinism, seed_everything, select_device,
)
from agoraforge.training.curriculum import build_training_levels, print_schedule
from agoraforge.training.models import (
    build_training_models,
    model_parameter_counts,
)
from agoraforge.training.artifacts.outputs import harvest, output_paths, setup_output_writer
from agoraforge.training.trainer import (
    EpochRunContext, LoopController, install_signal_handlers, run_training_epoch,
)
from agoraforge.envs.registry import get_env


# ── Main ─────────────────────────────────────────────────────────────────────

_CONFIG = config_flags.DEFINE_config_file(
    'config', default=os.path.join(os.path.dirname(__file__), 'conf/runs/debate/default.py'),
    help_string='Path to the entry config (ml_collections).')

_NAME = flags.DEFINE_string('name', None, 'Output run name.', required=True)


def main(argv):
    del argv
    cfg: ConfigDict = _CONFIG.value
    config_overrides = config_flags.get_override_values(flags.FLAGS["config"])

    spec = get_env(cfg.env_name)
    training = cfg.training
    online_epochs = int(training.online_epochs)
    online_buffer_size = int(training.online_buffer_size)
    # Cadence for intermediate actor checkpoints (0 = only the final actor).
    checkpoint_every = int(training.checkpoint_every_epochs)
    seed = int(cfg.seed)
    paths = output_paths(cfg.results_dir, _NAME.value)
    # Optional wall-clock cap. 0 (the default) keeps the epoch count in full
    # control; a positive value stops the loop once that many seconds of training
    # have elapsed. Hyperparameter tuning uses this to give every trial the same
    # time budget regardless of model or problem size.
    max_wall_clock_seconds = float(cfg.max_wall_clock_seconds)

    configure_cuda_allocator()

    # Opt-in bit-reproducibility (default off, keeps the fast non-deterministic
    # path). Must precede any CUDA op, so enable it before seeding/model build.
    deterministic = bool(training.deterministic)
    if deterministic:
        enable_determinism()

    seed_everything(seed)

    install_signal_handlers()

    device = select_device(cfg.device)
    levels, schedule = build_training_levels(cfg)
    print_schedule(schedule)

    models = build_training_models(cfg, levels, device)
    model = models.model
    model_cfg = models.model_cfg
    optimizer = models.optimizer

    actor_params, critic_params = model_parameter_counts(model)
    print(f"\nModel: {actor_params:,} actor params, {critic_params:,} value-head params")
    # Seeds reproduce bit-exactly only on the same GPU model.
    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    print(f"Device: {hw}" + (" (deterministic)" if deterministic else ""))

    writer = setup_output_writer(
        cfg=cfg,
        name=_NAME.value,
        paths=paths,
        levels=levels,
        config_overrides=config_overrides,
    )

    targs = PPOHparams.from_config(
        cfg,
        online_epochs=online_epochs,
        ppo_diagnostics=bool(cfg.logging.ppo_diagnostics),
    )
    gamma = float(cfg.training.gamma)
    logging_cfg = cfg.logging
    print_profile_breakdown = bool(logging_cfg.print_profile_breakdown)
    log_profile_to_tensorboard = bool(logging_cfg.log_profile_scalars)
    profile_sync_enabled = bool(logging_cfg.profile_sync)
    if deterministic:
        print("Deterministic algorithms: enabled (bit-reproducible, slower)")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n=== Training: epochs 0..{online_epochs - 1} ===\n")

    loop = LoopController(max_wall_clock_seconds)

    # Tail-window of per-level primary-metric means for the end-of-run curve file
    # (curve runs disable W&B, so this persists the metric).
    curve_tail = 5
    primary_tail_history = deque(maxlen=curve_tail)

    epoch_ctx = EpochRunContext(
        spec=spec,
        targs=targs,
        buffer_size=online_buffer_size,
        seed=seed,
        levels=levels,
        schedule=schedule,
        model=model,
        optimizer=optimizer,
        model_cfg=model_cfg,
        device=device,
        writer=writer,
        paths=paths,
        loop=loop,
        primary_tail_history=primary_tail_history,
        checkpoint_every=checkpoint_every,
        print_profile_breakdown=print_profile_breakdown,
        log_profile_to_tensorboard=log_profile_to_tensorboard,
        profile_sync_enabled=profile_sync_enabled,
        gamma=gamma,
    )

    for epoch in range(online_epochs):
        if loop.should_stop_before_epoch(epoch):
            break
        epoch_start = loop.start_epoch()
        run_training_epoch(epoch_ctx, epoch, online_epochs)
        writer.flush()
        loop.finish_epoch(epoch, epoch_start)

    # Final harvest: same code path as the per-epoch and stop paths,
    # so normal completion, the wall-clock cap, and signals all leave identical art.
    harvest(paths.run_dir, paths.save_dir, model, model_cfg,
            loop.last_completed_epoch, primary_tail_history, list(levels),
            spec.primary_metric)
    print("\nTraining complete.")

    writer.close()


def cli():
    app.run(main)


if __name__ == '__main__':
    cli()
