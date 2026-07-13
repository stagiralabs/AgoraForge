"""Shared fixtures for attention-equivalence testing.

Builds the real GraphActor / shared actor-critic and generates structurally-valid batched
observations (matching the collate output: padded formula rows, -1 padded ids,
active neg/query/slot edges) at a chosen N. Any perf optimization must leave the
forward and backward of these models numerically unchanged on this input -- the
equivalence tests in ``test_attention_equivalence.py`` enforce that.

The observation *content* is randomized rather than game-realistic: equivalence
holds for any valid input, and randomized content exercises every edge channel
and the padding/mask path. Realistic-content timing lives in the profiler runs.
"""

from __future__ import annotations

import numpy as np
import torch

import math as _math

from agoraforge.models.graph_transformer import GraphActor, GraphTransformerConfig, SharedGraphActorCritic
from agoraforge.envs.math import model as math_model
from agoraforge.envs.math.obs import (
    AGENT_SCALAR_DIM,
    FORMULA_FEATURE_DIM,
    SLOT_EDGE_DIM,
    SLOT_FEATURE_DIM,
)


def build_config(centralized: bool, n_embd: int = 9, n_head: int = 3, n_layer: int = 5) -> GraphTransformerConfig:
    """A GraphTransformerConfig matching the N=64 ``large`` preset shape (n_embd=9, 3 heads, 5 layers)."""
    # Mirror the resolved defaults from envs.math.model.build_model_config.
    attention_mode = "auto" if centralized else "dense"
    return GraphTransformerConfig(
        env_name="math",
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        attention_mode=attention_mode,
        node_feat_dim=FORMULA_FEATURE_DIM,
        agent_scalar_dim=AGENT_SCALAR_DIM,
        head={
            "centralized": centralized,
            "demand_log_std_base": _math.log(0.4216072441824628),
            "budget_log_std_base": _math.log(0.2823763678533231),
            "budget_mu_base": _math.log(9.36964023613551),
            "log_std_min": -5.0,
            "log_std_max": 2.0,
        },
    )


def _valid_counts(rng: np.random.Generator, bsz: int, max_f: int) -> np.ndarray:
    """Per-row valid formula counts in [max(1, max_f//2), max_f] so padding is exercised."""
    lo = max(1, max_f // 2)
    return rng.integers(lo, max_f + 1, size=bsz)


def make_actor_obs(
    bsz: int,
    max_f: int,
    *,
    centralized: bool = False,
    n_slots: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """A batched actor observation with valid masks and active neg/query/slot edges."""
    rng = np.random.default_rng(seed)
    counts = _valid_counts(rng, bsz, max_f)

    formula_features = rng.standard_normal((bsz, max_f, FORMULA_FEATURE_DIM)).astype(np.float32)
    formula_mask = np.zeros((bsz, max_f), dtype=np.float32)
    formula_ids = -np.ones((bsz, max_f), dtype=np.int64)
    neg_formula_ids = -np.ones((bsz, max_f), dtype=np.int64)
    agent_scalars = rng.standard_normal((bsz, AGENT_SCALAR_DIM)).astype(np.float32)
    query_related_edges = np.zeros((bsz, max_f, max_f), dtype=np.float32)

    for i, n in enumerate(counts):
        formula_mask[i, :n] = 1.0
        # Unique non-negative ids per valid row (distinct across the batch row).
        formula_ids[i, :n] = np.arange(n) + i * max_f
        # Point ~half the valid rows' negation at another valid row's id -> neg edges.
        for k in range(n):
            if rng.random() < 0.5:
                neg_formula_ids[i, k] = formula_ids[i, (k + 1) % n]
        # Sparse positive query weights inside the valid block -> query edges/bias.
        block = (rng.random((n, n)) < 0.3) * rng.random((n, n))
        query_related_edges[i, :n, :n] = block.astype(np.float32)
        # Zero the padded feature rows (the codec leaves padded rows at zero).
        formula_features[i, n:] = 0.0

    obs = {
        "formula_features": torch.from_numpy(formula_features),
        "formula_mask": torch.from_numpy(formula_mask),
        "formula_ids": torch.from_numpy(formula_ids),
        "neg_formula_ids": torch.from_numpy(neg_formula_ids),
        "agent_scalars": torch.from_numpy(agent_scalars),
        "query_related_edges": torch.from_numpy(query_related_edges),
    }

    if centralized:
        # One joint graph per env: n_slots model nodes, decoded together (no focal
        # marker). bsz is the env-row count; the actor emits bsz * n_slots actions.
        slot_features = rng.standard_normal((bsz, n_slots, SLOT_FEATURE_DIM)).astype(np.float32)
        slot_formula_edges = np.zeros((bsz, n_slots, max_f, SLOT_EDGE_DIM), dtype=np.float32)
        for i, n in enumerate(counts):
            mask = (rng.random((n_slots, n, SLOT_EDGE_DIM)) < 0.3)
            slot_formula_edges[i, :, :n] = (mask * rng.standard_normal((n_slots, n, SLOT_EDGE_DIM))).astype(np.float32)
        obs["slot_features"] = torch.from_numpy(slot_features)
        obs["slot_formula_edges"] = torch.from_numpy(slot_formula_edges)

    return {k: v.to(device) for k, v in obs.items()}


def build_actor(centralized: bool = False, seed: int = 0, device: str = "cpu") -> GraphActor:
    torch.manual_seed(seed)
    return math_model.build_actor(build_config(centralized)).to(device).eval()


def build_shared(seed: int = 0, device: str = "cpu") -> SharedGraphActorCritic:
    torch.manual_seed(seed)
    return math_model.build_shared(build_config(False)).to(device).eval()
