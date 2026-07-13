"""Glue for building the CPU reference factor graph from config."""

from __future__ import annotations

import numpy as np

from tests.math.reference.barabasi_albert import barabasi_albert_edges
from tests.math.reference.factor_graph import FormulaFactorGraph
from tests.math.reference.unary_factors import sample_unary_factors


def _discrete_weight_values(cfg) -> np.ndarray:
    """Discrete weight levels in ``[0, 1]`` derived from the config."""
    return np.linspace(
        float(cfg.weight_min), float(cfg.weight_max), int(cfg.n_weight_levels)
    )


def build_factor_graph(cfg, rng: np.random.Generator) -> FormulaFactorGraph:
    """Stage (1): write down the distribution.

    Draw the Barabási–Albert theorem graph (the factor graph's *structure*), then draw a
    random unary factor for each truth and weight variable, and attach all of them as the
    factors parameterized by ``cfg``. The caller owns the rng, so every draw is
    seeded; an unseeded fallback here would silently break reproducibility.
    """
    theorem_edges = barabasi_albert_edges(
        cfg.num_theorems, m=cfg.theorem_graph_m, rng=rng
    )
    levels = _discrete_weight_values(cfg)
    truth_unary, weight_unary = sample_unary_factors(
        cfg, theorem_edges, levels, rng=rng
    )
    return FormulaFactorGraph(
        num_theorems=cfg.num_theorems,
        theorem_edges=theorem_edges,
        levels=levels,
        penalty=cfg.weight_penalty,
        truth_unary=truth_unary,
        weight_unary=weight_unary,
    )
