"""CPU oracle for Barabási–Albert theorem-graph sampling.

Linking theorems with preferential attachment is stage (1) of generation — it lays
down which theorems are related, and hence which weight variables and factors exist in
the factor graph. The actual truth/weight values are drawn later (stage (2)).
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


def barabasi_albert_edges(
    num_nodes: int,
    m: float,
    rng: np.random.Generator,
) -> Set[Tuple[int, int]]:
    """Undirected scale-free graph via preferential attachment.

    Nodes are ``0..num_nodes-1``. Each new node attaches to existing nodes drawn with
    probability proportional to their current degree, yielding a power-law degree
    distribution. Returns a set of ``(lo, hi)`` edges (always ``lo < hi``).

    ``m`` is the *expected* number of edges each new node brings and may be fractional.
    A value ``m = floor + frac`` attaches ``floor + 1`` edges with probability ``frac``
    and ``floor`` otherwise, so the per-node mean is exactly ``m``.
    The average degree stays ``≈ 2m``.
    """
    num_nodes = int(num_nodes)
    if num_nodes < 2:
        return set()
    m = min(max(1.0, float(m)), float(num_nodes - 1))
    m_floor = int(np.floor(m))
    m_frac = m - m_floor

    def _draw_count(available: int) -> int:
        k = m_floor + (1 if rng.random() < m_frac else 0)
        return max(1, min(k, available))

    edges: Set[Tuple[int, int]] = set()
    targets = list(range(m_floor))   # seed nodes the first new node attaches to
    repeated_nodes: list = []        # endpoint multiset; degree-weighted draw pool
    for source in range(m_floor, num_nodes):
        for target in targets:
            edges.add((target, source))  # target < source by construction
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * len(targets))
        targets = _sample_distinct(repeated_nodes, _draw_count(source + 1), rng)
    return edges


def _sample_distinct(pool: list, k: int, rng: np.random.Generator) -> List[int]:
    """Draw ``k`` distinct values from ``pool`` (a degree-weighted multiset)."""
    pool_arr = np.asarray(pool)
    chosen: Set[int] = set()
    while len(chosen) < k:
        chosen.add(int(pool_arr[rng.integers(0, len(pool_arr))]))
    return list(chosen)
