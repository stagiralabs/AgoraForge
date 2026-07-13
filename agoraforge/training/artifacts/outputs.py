"""Training output paths and final artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agoraforge.training.artifacts.config import write_config_artifacts
from agoraforge.models.graph_transformer import GraphTransformerConfig


def atomic_torch_save(obj, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_model(
    model, model_cfg: GraphTransformerConfig, epoch: int, path: str
) -> None:
    """Write a self-describing model checkpoint atomically."""
    atomic_torch_save({
        "state_dict": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "epoch": int(epoch),
    }, path)


class WandbSummaryWriter:
    """Buffer TensorBoard-style scalars into one committed W&B row per step."""

    def __init__(self, run, tee=None):
        self._run = run
        self._tee = tee
        self._step = None
        self._row = {}

    def add_scalar(self, tag, value, global_step=None):
        if global_step != self._step:
            self.flush()
            self._step = global_step
        self._row[tag] = value
        if self._tee is not None:
            self._tee.add_scalar(tag, value, global_step)

    def flush(self):
        if self._row:
            self._run.log(self._row, step=self._step, commit=True)
            self._row = {}
        if self._tee is not None:
            self._tee.flush()

    def close(self):
        self.flush()
        if self._tee is not None:
            self._tee.close()
        self._run.finish()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def run_output_dir(results_dir, name):
    """Return results_dir/<name>, rejecting names that are not a single path segment."""
    if not name or name in (".", "..") or "/" in name or os.sep in name:
        raise ValueError(
            f"--name must be a simple run name without path separators, got {name!r}"
        )
    return os.path.join(results_dir, name)


def assert_run_dir_available(run_dir):
    """Refuse to start if this run name already has outputs."""
    if os.path.exists(run_dir):
        raise FileExistsError(
            f"Run output path already exists: {run_dir}. "
            "Pick a different --name or delete that directory."
        )


@dataclass(frozen=True)
class OutputPaths:
    run_dir: str
    save_dir: str
    tensorboard_dir: str


def output_paths(results_dir: str, name: str) -> OutputPaths:
    run_dir = run_output_dir(results_dir, name)
    return OutputPaths(
        run_dir=run_dir,
        save_dir=os.path.join(run_dir, "checkpoints"),
        tensorboard_dir=os.path.join(run_dir, "tensorboard"),
    )


def setup_output_writer(
    *,
    cfg,
    name: str,
    paths: OutputPaths,
    levels: dict,
    config_overrides: dict,
):
    assert_run_dir_available(paths.run_dir)
    print(f"Run '{name}' -> {paths.run_dir}")
    os.makedirs(paths.run_dir, exist_ok=True)
    os.makedirs(paths.save_dir, exist_ok=True)
    os.makedirs(paths.tensorboard_dir, exist_ok=True)
    write_config_artifacts(
        paths.run_dir,
        cfg=cfg,
        levels=levels,
        overrides=config_overrides,
    )

    logging_cfg = cfg.logging
    if logging_cfg.wandb_enabled:
        try:
            import wandb as wandb_lib
        except ImportError:
            raise RuntimeError(
                "W&B logging requires the optional dependency: "
                "pip install 'agoraforge[wandb]'"
            ) from None
        run = wandb_lib.init(
            project=os.environ.get("WANDB_PROJECT", "agoraforge"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=name,
            config=cfg.to_dict(),
            dir=paths.run_dir,
            mode=logging_cfg.wandb_mode,
        )
        tee = SummaryWriter(paths.tensorboard_dir) if logging_cfg.tee_tensorboard else None
        return WandbSummaryWriter(run, tee=tee)
    return SummaryWriter(paths.tensorboard_dir)


def write_level_curve(path, primary_history, level_names, metric_name):
    def tail_mean(history, name):
        vals = [d[name] for d in history if name in d]
        return float(np.mean(vals)) if vals else None

    curve = {name: {metric_name: tail_mean(primary_history, name)} for name in level_names}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"tail_epochs": len(primary_history), "levels": curve}, f, indent=2)
    os.replace(tmp, path)


def harvest(run_dir, save_dir, model, model_cfg, final_epoch,
            primary_history, level_names, metric_name):
    """Write the final checkpoint and the tail-averaged curve atomically."""
    save_model(model, model_cfg, final_epoch,
               os.path.join(save_dir, "model_final.pt"))
    write_level_curve(
        os.path.join(run_dir, "level_curve.json"),
        primary_history,
        level_names,
        metric_name,
    )
