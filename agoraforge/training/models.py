"""Resident PPO model and optimizer construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agoraforge.models.factory import (
    build_model_config,
    build_shared,
)


@dataclass(frozen=True)
class TrainingModelBundle:
    model: torch.nn.Module
    model_cfg: object
    optimizer: torch.optim.Optimizer


def shared_optimizer(model, training_cfg) -> torch.optim.Optimizer:
    """AdamW over a shared-trunk actor/critic with role-tagged param groups.

    The value head is the critic; everything else (trunk + action head) is the
    actor. The roles carry separate learning rates through the decay schedule.
    """
    critic_ids = {id(p) for p in model.value_head.parameters()}
    critic_params = [p for p in model.parameters() if id(p) in critic_ids]
    actor_params = [p for p in model.parameters() if id(p) not in critic_ids]
    return torch.optim.AdamW(
        [
            {"params": actor_params, "lr": float(training_cfg.actor_lr), "weight_decay": 0.1, "role": "actor"},
            {"params": critic_params, "lr": float(training_cfg.critic_lr), "weight_decay": 0.1, "role": "critic"},
        ],
    )


def build_training_models(cfg, levels, device) -> TrainingModelBundle:
    ref_cfg = list(levels.values())[0]
    model_cfg = build_model_config(cfg.actor_model, ref_cfg, cfg.decoding)
    model = build_shared(model_cfg).to(device)
    return TrainingModelBundle(
        model=model,
        model_cfg=model_cfg,
        optimizer=shared_optimizer(model, cfg.training),
    )


def model_parameter_counts(model) -> tuple[int, int]:
    """(actor, critic) parameter counts: value head is the critic role."""
    critic = sum(p.numel() for p in model.value_head.parameters())
    return sum(p.numel() for p in model.parameters()) - critic, critic


def update_value_stats(model, mean: float, std: float) -> None:
    model.update_value_stats(mean, std)
