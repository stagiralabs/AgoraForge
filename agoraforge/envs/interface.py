"""The environment interface consumed by training and search.

An environment is a batched, device-resident simulator. Its action tensors are its
own (an env-specific ``TensorActions`` dataclass): training never inspects them, it
only passes them from the env's policy module back into ``step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class PolicyInputs:
    """Actor inputs for one step, without constructing unused critic features.

    ``ctx`` carries whatever the env's policy module needs to map sampled actions
    back onto the full env (e.g. a top-k node selection index); training treats it
    as opaque.
    """

    actor_obs: dict
    masks: dict
    ctx: Any = None


@runtime_checkable
class BatchedEnv(Protocol):
    B: int
    A: int
    cfg: Any
    device: torch.device
    centralized: bool

    @classmethod
    def from_config(
        cls, cfg, *, batch_size, device, seeds, profiler=None
    ) -> "BatchedEnv": ...

    @classmethod
    def batched_from_config(
        cls, cfg, *, base_batch_size, thetas, device, seeds, step_seed=0, profiler=None,
    ) -> "BatchedEnv": ...

    def policy_inputs(self) -> PolicyInputs: ...

    def step(self, actions) -> torch.Tensor: ...

    def fitness(self) -> torch.Tensor: ...

    def eval_metrics(self) -> dict: ...
