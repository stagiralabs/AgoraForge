"""Batched Barabási–Albert parent sampling shared by the env instance samplers."""

from __future__ import annotations

import torch


def barabasi_albert_parents(
    B: int, N: int, m: float, device, gen: torch.Generator,
    *, degree_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched preferential-attachment graph as a fixed-shape parent tensor.

    Replicates the CPU reference ``barabasi_albert_edges`` per env:
    ``m_floor = floor(m)``, ``m_frac = m - m_floor``; node ``source`` attaches to its
    ``targets`` (1..m_floor+1 of them), edges added ``(target, source)``; then the next
    ``targets`` are drawn distinct from the degree-weighted endpoint multiset.

    Returns ``(parents[B, N, K], pcount[B, N])`` where ``parents[b, s, :pcount[b,s]]`` are
    the existing nodes that node ``s`` attached to (all ``< s``), and ``K = m_floor + 1``.
    Node ``source`` therefore contributes ``pcount[b, source]`` undirected edges. This is a
    lossless, ragged-free encoding of the reference edge set (each edge has a unique
    higher endpoint = its ``source``).
    """
    m = min(max(1.0, float(m)), float(N - 1))
    m_floor = int(m)
    m_frac = m - m_floor
    K = m_floor + 1  # max parents per node

    parents = torch.zeros((B, N, K), dtype=torch.long, device=device)
    pcount = torch.zeros((B, N), dtype=torch.long, device=device)
    if N < 2:
        return parents, pcount

    # Current degree of every node, grown as we add nodes (the attachment weights).
    degree = torch.zeros((B, N), dtype=degree_dtype, device=device)
    # `targets` for the very first new node = seed nodes 0..m_floor-1, shared by all envs.
    # Stored as a (B, K) buffer with a per-env validity count.
    cur_targets = torch.zeros((B, K), dtype=torch.long, device=device)
    cur_count = torch.full((B,), m_floor, dtype=torch.long, device=device)
    for j in range(m_floor):
        cur_targets[:, j] = j

    for source in range(m_floor, N):
        # Attach `source` to its current targets: add edges (target, source).
        valid = torch.arange(K, device=device).view(1, K) < cur_count.view(B, 1)  # (B,K)
        parents[:, source, :] = torch.where(valid, cur_targets, torch.zeros_like(cur_targets))
        pcount[:, source] = cur_count
        # Update degrees: each chosen target +1, and source += number of targets.
        tgt_clamped = cur_targets.clamp(0, N - 1)
        degree.scatter_add_(1, tgt_clamped, valid.to(degree_dtype))
        degree[:, source] += cur_count.to(degree_dtype)

        if source + 1 >= N:
            break

        # Draw next count k = m_floor + (1 if u < m_frac else 0), clamped to [1, source+1].
        extra = (torch.rand((B,), device=device, generator=gen) < m_frac).to(torch.long)
        k = (m_floor + extra).clamp(1, source + 1)

        # Sample `k` DISTINCT nodes ∝ current degree (over nodes 0..source). The reference
        # draws from a degree-weighted endpoint multiset with rejection-until-distinct;
        # sampling ∝ degree without replacement is the same target distribution.
        cur_targets, cur_count = _sample_distinct_by_degree(degree, source + 1, k, K, gen)
    return parents, pcount


def _sample_distinct_by_degree(
    degree: torch.Tensor, n_active: int, k: torch.Tensor, K: int, gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw, per env, ``k[b]`` distinct nodes from ``0..n_active-1`` with prob ∝ degree.

    Uses the Gumbel-top-``K`` trick: sampling-without-replacement ∝ weight is equivalent to
    taking the top entries of ``log(weight) + Gumbel``. We take the top ``K`` and mark only
    the first ``k[b]`` as valid (the reference draws exactly ``k`` distinct).
    """
    B = degree.shape[0]
    w = degree[:, :n_active].clamp_min(1e-12)
    logits = torch.log(w)
    u = torch.rand((B, n_active), device=degree.device, generator=gen).clamp_(1e-12, 1.0)
    gumbel = -torch.log(-torch.log(u))
    keys = logits + gumbel
    topk = torch.topk(keys, k=min(K, n_active), dim=1).indices  # (B, <=K)
    out = torch.zeros((B, K), dtype=torch.long, device=degree.device)
    out[:, : topk.shape[1]] = topk
    return out, k


def adjacency_from_parents(
    parents: torch.Tensor, pcount: torch.Tensor, N: int
) -> torch.Tensor:
    """Dense symmetric (B, N, N) bool adjacency from the parent encoding."""
    B, _, K = parents.shape
    device = parents.device
    adj = torch.zeros((B, N, N), dtype=torch.bool, device=device)
    src_nodes = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
    valid = torch.arange(K, device=device).view(1, 1, K) < pcount.view(B, N, 1)
    a_lo = torch.minimum(parents, src_nodes)
    b_hi = torch.maximum(parents, src_nodes)
    flat_idx = (a_lo * N + b_hi)[valid]
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)[valid]
    adj_flat = adj.view(B, N * N)
    adj_flat[batch_idx, flat_idx] = True
    adj = adj_flat.view(B, N, N)
    adj = adj | adj.transpose(1, 2)
    adj.diagonal(dim1=1, dim2=2).fill_(False)
    return adj
