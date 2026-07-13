"""Tensor action-distribution helpers shared by environment policies."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def as_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.squeeze(1) if logits.dim() == 3 else logits


def as_1d(t: torch.Tensor) -> torch.Tensor:
    """Flatten a per-row scalar head to shape (B,)."""
    return t.reshape(t.shape[0])


def gaussian_kl_to_base(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    mu_base: float,
    log_std_base: float,
) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(mu_base, sigma_base^2)), clamped to >= 0."""
    std = log_std.exp()
    inv_two_var = 0.5 * math.exp(-2.0 * log_std_base)
    kl = (log_std_base - log_std) + (std * std + (mu - mu_base) ** 2) * inv_two_var - 0.5
    return kl.clamp(min=0.0)


def as_3d_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.squeeze(1) if logits.dim() == 4 else logits


LOG_SQRT_2PI = math.log(math.sqrt(2.0 * math.pi))


def cat_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def cat_log_prob(log_probs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather log-probs at ``idx`` from a pre-computed log-softmax table."""
    return log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def cat_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    return -(log_probs.exp() * log_probs).sum(dim=-1)


GUMBEL_EPS = 1e-10


def gumbel_sample(
    logits: torch.Tensor,
    generator: torch.Generator | None = None,
    crn_blocks: int = 1,
) -> torch.Tensor:
    """Sample categories, optionally broadcasting one CRN draw over blocks."""
    if crn_blocks < 1 or logits.shape[0] % crn_blocks:
        raise ValueError(
            f"crn_blocks={crn_blocks} must divide leading size {logits.shape[0]}"
        )
    shape = (logits.shape[0] // crn_blocks, *logits.shape[1:])
    if crn_blocks == 1 and generator is None:
        u = torch.rand_like(logits)
    else:
        u = torch.rand(
            shape, dtype=logits.dtype, device=logits.device, generator=generator,
        )
    if crn_blocks > 1:
        u = u.repeat((crn_blocks,) + (1,) * (u.dim() - 1))
    u = u.clamp_(GUMBEL_EPS, 1.0 - GUMBEL_EPS)
    return (logits - torch.log(-torch.log(u))).argmax(dim=-1)


def crn_normal(
    mu: torch.Tensor,
    std: torch.Tensor,
    *,
    generator: torch.Generator | None,
    crn_blocks: int,
) -> torch.Tensor:
    """Sample N(mu, std), optionally broadcasting one CRN draw over blocks."""
    if crn_blocks == 1 and generator is None:
        return torch.normal(mu, std)
    shape = (mu.shape[0] // crn_blocks, *mu.shape[1:])
    noise = torch.randn(
        shape, dtype=mu.dtype, device=mu.device, generator=generator,
    )
    if crn_blocks > 1:
        noise = noise.repeat((crn_blocks,) + (1,) * (noise.dim() - 1))
    return mu + std * noise


def normal_log_prob(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std * std
    return -((x - mu) ** 2) / (2 * var) - std.log() - LOG_SQRT_2PI


def categorical_uniform_kl(entropy: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """KL(pi || uniform-over-valid-actions) = log(n_valid) - H."""
    kl = torch.log(n_valid.clamp(min=1.0)) - entropy
    return torch.where(n_valid > 1, kl.clamp(min=0.0), torch.zeros_like(kl))
