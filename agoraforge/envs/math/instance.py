"""Static resident-instance tensor construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agoraforge.envs.math.config import MathConfig
from agoraforge.envs.math.latent import latent_config_from_config, sample_latent_batched


@dataclass(frozen=True)
class StaticInstance:
    neg_ids: torch.Tensor
    theorem_ids: torch.Tensor
    formula_ids: torch.Tensor
    truth: torch.Tensor
    graph_weights: torch.Tensor
    query_truth: torch.Tensor
    qw_src: torch.Tensor
    qw_dst: torch.Tensor
    query_values: torch.Tensor


def build_static_instance(
    cfg: MathConfig,
    *,
    batch_size: int,
    n_agents: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> StaticInstance:
    """Build immutable graph/truth/query tensors for a resident rollout batch."""
    B = int(batch_size)
    A = int(n_agents)
    N = int(cfg.num_theorems)
    F_size = int(cfg.F_size)
    neg_ids = torch.cat(
        [
            torch.arange(N, F_size, device=device),
            torch.arange(0, N, device=device),
        ]
    ).to(torch.long)
    theorem_ids = torch.arange(F_size, device=device, dtype=torch.long) % N
    formula_ids = torch.arange(F_size, device=device, dtype=torch.long)

    truth_sign, graph_weights = _sample_truth_and_weights(
        cfg, batch_size=B, dtype=dtype, device=device, gen=gen
    )
    signs = (torch.arange(F_size, device=device) // N).view(1, F_size)
    theorem_for_formula = theorem_ids.view(1, F_size).expand(B, F_size)
    true_sign_for_formula = truth_sign.gather(1, theorem_for_formula)
    truth = (signs == true_sign_for_formula).to(dtype)

    query_truth = _sample_query_truth(
        cfg,
        truth_sign=truth_sign,
        theorem_ids=theorem_ids,
        batch_size=B,
        n_agents=A,
        dtype=dtype,
        device=device,
        gen=gen,
    )
    qw_src, qw_dst, query_values = _build_sparse_query_weights(
        cfg,
        graph_weights=graph_weights,
        batch_size=B,
        n_agents=A,
        dtype=dtype,
        device=device,
        gen=gen,
    )
    return StaticInstance(
        neg_ids=neg_ids,
        theorem_ids=theorem_ids,
        formula_ids=formula_ids,
        truth=truth,
        graph_weights=graph_weights,
        query_truth=query_truth,
        qw_src=qw_src,
        qw_dst=qw_dst,
        query_values=query_values,
    )


def _sample_truth_and_weights(
    cfg: MathConfig,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    truth_sign, weights = sample_latent_batched(
        latent_config_from_config(cfg), batch_size, device, gen
    )
    truth_sign = truth_sign.to(torch.long)
    weights = weights.to(dtype)
    if cfg.truth_map is not None:
        truth_sign = torch.as_tensor(
            cfg.truth_map, dtype=torch.long, device=device
        ).view(1, cfg.num_theorems).expand(batch_size, cfg.num_theorems)
    if cfg.utility_weights is not None:
        weights = torch.zeros((batch_size, cfg.F_size, cfg.F_size), dtype=dtype, device=device)
        for (src, dst), weight in cfg.utility_weights.items():
            weights[:, int(src), int(dst)] = float(weight)
    return truth_sign, weights


def _sample_query_truth(
    cfg: MathConfig,
    *,
    truth_sign: torch.Tensor,
    theorem_ids: torch.Tensor,
    batch_size: int,
    n_agents: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    prior = float(cfg.query_truth_prior_correct)
    N = int(cfg.num_theorems)
    F_size = int(cfg.F_size)
    true_sign = truth_sign.unsqueeze(1).expand(batch_size, n_agents, N)
    reports_correct = torch.rand((batch_size, n_agents, N), device=device, generator=gen) < prior
    p_sign_one_if_true = torch.where(
        true_sign == 1,
        torch.full((), prior, dtype=dtype, device=device),
        torch.full((), 1.0 - prior, dtype=dtype, device=device),
    )
    p_sign_one = torch.where(reports_correct, p_sign_one_if_true, 1.0 - p_sign_one_if_true)
    theorem_for_formula = theorem_ids.view(1, 1, F_size).expand(batch_size, n_agents, F_size)
    p_formula_sign_one = p_sign_one.gather(2, theorem_for_formula)
    sign = (torch.arange(F_size, device=device) // N).view(1, 1, F_size)
    probs = torch.where(sign == 1, p_formula_sign_one, 1.0 - p_formula_sign_one)
    return probs.clamp(1e-6, 1.0 - 1e-6)


def _build_sparse_query_weights(
    cfg: MathConfig,
    *,
    graph_weights: torch.Tensor,
    batch_size: int,
    n_agents: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_mask = (graph_weights != 0.0).any(dim=0)
    qw_src, qw_dst = torch.nonzero(edge_mask, as_tuple=True)
    src, dst = qw_src, qw_dst
    E = src.numel()
    base = graph_weights[:, src, dst]
    noise = torch.randn(
        (batch_size, n_agents, E), device=device, generator=gen, dtype=dtype
    )
    values = torch.minimum(
        base.unsqueeze(1) * torch.exp(noise * float(cfg.query_weight_noise)),
        torch.ones((), dtype=dtype, device=device),
    )
    values = torch.where((src == dst).view(1, 1, E), torch.zeros_like(values), values)
    return qw_src, qw_dst, values
