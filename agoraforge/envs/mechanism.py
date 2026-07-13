"""Shared implementation toolkit for environment learned mechanisms.

Every environment's searched mechanism is a set of per-(node, agent) learned
states ``s[v, j]`` evolved by one small map ``F``. Each evaluation of ``F`` returns
both the state update and reward. The flat parameter vector ``theta`` of that
mechanism is frozen during inner RL training and searched over by the outer loop.

Each environment implements the family as two classes built from the helpers
here:

* a **spec** module (e.g. ``MarketMechanism``): an ordinary ``nn.Module`` whose
  only jobs are parameter layout and initialization -- ``theta`` vectors are
  created from it and decoded through it, so its registration order is the
  on-disk layout;
* a **batched executor** (e.g. ``BatchedMarketMechanism``): K spec instances'
  weights stacked into buffers and applied per candidate with
  :func:`candidate_linear`. All execution -- population search and ordinary
  K=1 runs alike -- goes through the executor, so there is only one
  implementation of the mechanism math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

EPS = 1e-12

# Keep invalid neural readouts from contaminating observations or rewards. The state
# recurrence itself is bounded by construction; this clamp only sanitizes NaN/Inf from
# extreme candidate weights before they enter the rest of the environment.
FINITE_CLAMP = 1e6


def finite(t: torch.Tensor) -> torch.Tensor:
    """Map NaN/Inf to finite values and clamp magnitude; no-op for finite states."""
    return torch.nan_to_num(
        t, nan=0.0, posinf=FINITE_CLAMP, neginf=-FINITE_CLAMP
    ).clamp(-FINITE_CLAMP, FINITE_CLAMP)


@dataclass(frozen=True)
class MechanismDims:
    """Shapes and fixed scales of a learned mechanism.

    ``d_action`` is the agent signal width (the policy's continuous per-node heads);
    ``d_e`` is fixed by the environment's e-vector encoding. ``d_state`` and
    ``hidden`` are the free design knobs the outer loop can grow."""

    d_state: int = 3
    d_action: int = 2
    d_e: int = 1
    hidden: int = 16
    # Typed fixed input scaling. Event/probability columns stay raw; state and action
    # groups are divided by these constants, pooled sums additionally by sqrt(A).
    input_norm: bool = True
    state_scale: float = 10.0
    action_scale: float = 1.0
    # Bounded state update: ``(1-decay)*s + state_bound*tanh(F(x)/state_bound)``.
    state_bound: float = 10.0
    state_decay: float = 0.1

    def __post_init__(self) -> None:
        for name in ("d_state", "d_action", "d_e", "hidden"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("state_scale", "action_scale", "state_bound"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0.0 < self.state_decay <= 1.0:
            raise ValueError(f"state_decay must be in (0,1], got {self.state_decay}")


def zero_final_mlp(in_dim: int, hidden: int, out_dim: int, *, layernorm: bool) -> nn.Sequential:
    """Two-layer MLP whose final layer is zero-initialized.

    Zero-init makes a fresh map return zero in every output coordinate."""
    layers: List[nn.Module] = []
    if layernorm:
        layers.append(nn.LayerNorm(in_dim))
    fc1 = nn.Linear(in_dim, hidden)
    fc2 = nn.Linear(hidden, out_dim)
    nn.init.zeros_(fc2.weight)
    nn.init.zeros_(fc2.bias)
    layers += [fc1, nn.GELU(), fc2]
    return nn.Sequential(*layers)


def bounded_residual(
    s: torch.Tensor, delta: torch.Tensor, bound: float, decay: float
) -> torch.Tensor:
    """Leaky bounded update, near-residual for small deltas.

    From zero initialization, ``|s| < bound / decay`` for every finite timestep.
    """
    return finite((1.0 - decay) * s + bound * torch.tanh(delta / bound))


class FlatParamsMixin:
    """Flat-theta I/O for a spec module; ``named_parameters`` order is the layout."""

    def flat_params(self) -> torch.Tensor:
        return nn.utils.parameters_to_vector(self.parameters()).detach().clone()

    def load_flat_params(self, vec: torch.Tensor) -> None:
        dtype = next(self.parameters()).dtype
        nn.utils.vector_to_parameters(vec.to(dtype), self.parameters())

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def stack_params(models, getter) -> torch.Tensor:
    """Stack one parameter selected by ``getter`` across the K models."""
    return torch.stack([getter(m).detach() for m in models], dim=0)


def linears(seq: nn.Sequential) -> list[nn.Linear]:
    """The Linear layers of a Sequential in order, skipping norms/activations."""
    return [m for m in seq if isinstance(m, nn.Linear)]


def first_layernorm(seq: nn.Sequential) -> nn.LayerNorm:
    return next(m for m in seq if isinstance(m, nn.LayerNorm))


def candidate_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per-candidate affine ``x @ W_k^T + b_k`` for ``x`` [K, ..., in]."""
    y = torch.einsum("k...i,koi->k...o", x, weight)
    return y + bias.view(bias.shape[0], *([1] * (x.dim() - 2)), bias.shape[-1])


def candidate_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-candidate LayerNorm over the last dim for ``x`` [K, ..., d]."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    shape = (weight.shape[0], *([1] * (x.dim() - 2)), weight.shape[-1])
    return x * weight.view(shape) + bias.view(shape)


def scaled_input(dims: MechanismDims, groups: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Concatenate typed input groups, each divided by its scale (1.0 = raw)."""
    if not dims.input_norm:
        return torch.cat([t for t, _ in groups], dim=-1)
    return torch.cat([t if s == 1.0 else t / s for t, s in groups], dim=-1)
