"""Device-specific runtime helpers."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext

import numpy as np
import torch


def configure_cuda_allocator() -> None:
    # Must be set before the first CUDA op; callers do this before seeding/model build.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def enable_determinism() -> None:
    """Make CUDA training bit-reproducible run-to-run."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def select_device(requested: str) -> torch.device:
    name = str(requested).lower()
    if name == "cpu":
        return torch.device("cpu")
    if name != "cuda":
        raise ValueError(f"device must be 'cpu' or 'cuda', got {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def profile_synchronizer(device: torch.device, enabled: bool):
    if not enabled:
        return None
    if device.type == "cuda":
        return torch.cuda.synchronize
    return None


def autocast_context(device, amp_dtype: str):
    dtype_name = str(amp_dtype or "").lower()
    if torch.device(device).type != "cuda" or dtype_name in ("", "none", "fp32", "float32"):
        return nullcontext()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name in ("fp16", "float16"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unknown amp dtype {amp_dtype!r}")
