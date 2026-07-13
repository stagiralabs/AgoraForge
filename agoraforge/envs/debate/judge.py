"""Exact-enumeration judge over the transcript-induced sub-factor-graph.

The judge sees only the selected claims and reasons with the factors among them: for
``C`` slots it enumerates all ``2^C`` assignments and computes exact marginals under
the induced potentials.

Invalid (empty) slots carry zero potentials, so they decouple: their marginal is exactly
0.5 and they leave every other slot's marginal untouched. Cost is
``O(B * 2^C * C^2)`` with the quadratic term chunked over the assignment axis.
"""

from __future__ import annotations

import torch

# (C, device, dtype) -> (2^C, C) sign table with entries in {-1, +1}.
_SIGN_TABLES: dict = {}

# Element budget for the (B, chunk, C) quadratic intermediate (~256 MB fp32).
_CHUNK_ELEMENT_BUDGET = 1 << 26


def _sign_table(C: int, device, dtype) -> torch.Tensor:
    key = (C, str(device), dtype)
    table = _SIGN_TABLES.get(key)
    if table is None:
        n = 1 << C
        bits = torch.arange(n, device=device).unsqueeze(1) >> torch.arange(C, device=device)
        table = ((bits & 1).to(dtype) * 2.0 - 1.0)  # (2^C, C)
        _SIGN_TABLES[key] = table
    return table


def gather_slot_potentials(
    couplings: torch.Tensor,   # (B, N, N)
    unary: torch.Tensor,       # (B, N)
    slots: torch.Tensor,       # (B, C) long claim ids (arbitrary where invalid)
    slot_valid: torch.Tensor,  # (B, C) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Induced-subgraph potentials ``(J_S[B,C,C], b_S[B,C])``, zeroed on invalid slots."""
    B, C = slots.shape
    idx = slots.clamp(min=0)
    rows = couplings.gather(1, idx.unsqueeze(-1).expand(B, C, couplings.shape[-1]))  # (B, C, N)
    J_S = rows.gather(2, idx.unsqueeze(1).expand(B, C, C))  # (B, C, C)
    valid_f = slot_valid.to(couplings.dtype)
    J_S = J_S * valid_f.unsqueeze(1) * valid_f.unsqueeze(2)
    b_S = unary.gather(1, idx) * valid_f
    return J_S, b_S


def judge_marginals(
    couplings: torch.Tensor,          # (B, N, N)
    unary: torch.Tensor,              # (B, N)
    slots: torch.Tensor,              # (B, C)
    slot_valid: torch.Tensor,         # (B, C)
) -> torch.Tensor:
    """Exact induced-subgraph marginals per selected slot -> ``(B, C)``."""
    J_S, b_S = gather_slot_potentials(couplings, unary, slots, slot_valid)
    B, C = b_S.shape
    dtype = couplings.dtype
    signs = _sign_table(C, couplings.device, dtype)  # (2^C, C)
    n_assign = signs.shape[0]

    # log-potential per assignment: 0.5 * b_eff . s + 0.5 * s^T J s (constants cancel).
    logits = 0.5 * (b_S @ signs.T)  # (B, 2^C)

    chunk = max(1, _CHUNK_ELEMENT_BUDGET // max(1, B * C))
    for start in range(0, n_assign, chunk):
        sc = signs[start : start + chunk]  # (n_c, C)
        tmp = torch.einsum("ac,bcd->bad", sc, J_S)  # (B, n_c, C)
        logits[:, start : start + chunk] += 0.5 * (tmp * sc.unsqueeze(0)).sum(dim=-1)

    weights = torch.softmax(logits, dim=-1)  # (B, 2^C)
    x_table = (signs + 1.0) * 0.5  # (2^C, C)
    return weights @ x_table  # (B, C)
