"""Dense/sparse attention numerical-equivalence harness.

The safety net for any speed optimization to the actor/critic: an optimized path
must leave the *full model* forward and backward numerically unchanged, and stay
deterministic given a seed. These tests run the real math models over
structurally-valid N=64 observations and assert:

  * determinism  -- same code, same input, run twice -> bit-identical;
  * equivalence  -- the alternative attention path (sparse, the standing
    optimization candidate) matches the dense default in forward and backward,
    for the market actor, the centralized actor, and the shared value head.

When a new optimization lands behind a flag, add it to ``ATTN_MODES`` / wire its
toggle here so it is held to the same bar before becoming the default.
"""

from __future__ import annotations

import torch

from tests.math.attention_fixtures import build_actor, build_shared, make_actor_obs

N64 = 65  # 1 global + 64 formulas (+ slots when centralized).


def _flatten_actor(out: dict) -> list[torch.Tensor]:
    """All tensor outputs of the actor head, in a fixed order, for comparison."""
    tensors = []
    for key in sorted(out):
        if key == "formula_logits_per_type":
            for t in sorted(out[key]):
                tensors.append(out[key][t])
        elif torch.is_tensor(out[key]):
            tensors.append(out[key])
    return tensors


def _scalar_loss(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Finite, mask-independent scalar that touches every output element so the
    # backward exercises every parameter's gradient path.
    return sum((t * t).sum() for t in tensors)


def _run_value(model, obs, mode):
    orig_model = getattr(model, "attention_mode", None)
    model.attention_mode = mode
    try:
        for p in model.parameters():
            p.grad = None
        _, values = model.forward_actor_critic(obs)
        loss = (values * values).sum()
        loss.backward()
        outs = [values.detach().clone()]
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        return outs, grads
    finally:
        model.attention_mode = orig_model


def _run(model, obs, mode, is_actor):
    orig_model = getattr(model, "attention_mode", None)
    model.attention_mode = mode
    try:
        for p in model.parameters():
            p.grad = None
        out = model(obs)
        tensors = _flatten_actor(out) if is_actor else [out]
        loss = _scalar_loss(tensors)
        loss.backward()
        outs = [t.detach().clone() for t in tensors]
        grads = [p.grad.detach().clone() for p in model.parameters()]
        return outs, grads
    finally:
        model.attention_mode = orig_model


def _assert_close(a, b, tag, tol=2e-4):
    """Scale-relative equivalence: maxdiff / max|y| < tol.

    Different attention paths sum the same terms in a different order, so they
    agree only up to float32 reordering. A scale-relative bound is the right test
    -- it is invariant to the (large, loss-amplified) gradient magnitudes while
    still catching any real divergence (a bug moves outputs by O(1) relative).
    """
    assert len(a) == len(b), tag
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape, f"{tag}[{i}] shape {x.shape} vs {y.shape}"
        if not x.numel():
            continue
        scale = y.abs().max().clamp_min(1e-6)
        rel = (x - y).abs().max() / scale
        assert rel < tol, f"{tag}[{i}] rel_maxdiff={rel.item():.2e} (scale={scale.item():.2e})"


def _assert_equal(a, b, tag):
    assert len(a) == len(b), tag
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"{tag}[{i}] not bit-identical (maxdiff {(x-y).abs().max().item()})"


# ── Determinism: same path twice must be bit-identical ──────────────────────

def test_actor_market_deterministic():
    actor = build_actor(centralized=False, seed=1)
    obs = make_actor_obs(8, N64, centralized=False, seed=2)
    o1, g1 = _run(actor, obs, "dense", is_actor=True)
    o2, g2 = _run(actor, obs, "dense", is_actor=True)
    _assert_equal(o1, o2, "actor-market fwd determinism")
    _assert_equal(g1, g2, "actor-market grad determinism")


def test_actor_centralized_deterministic():
    actor = build_actor(centralized=True, seed=1)
    obs = make_actor_obs(8, N64, centralized=True, seed=2)
    o1, g1 = _run(actor, obs, "dense", is_actor=True)
    o2, g2 = _run(actor, obs, "dense", is_actor=True)
    _assert_equal(o1, o2, "actor-centralized fwd determinism")
    _assert_equal(g1, g2, "actor-centralized grad determinism")


def test_shared_value_deterministic():
    shared = build_shared(seed=1)
    obs = make_actor_obs(8, N64, seed=2)
    o1, g1 = _run_value(shared, obs, "dense")
    o2, g2 = _run_value(shared, obs, "dense")
    _assert_equal(o1, o2, "shared value fwd determinism")
    _assert_equal(g1, g2, "shared value grad determinism")


# ── Equivalence: dense vs sparse on the full models at N=64 ─────────────────

def test_actor_market_dense_eq_sparse():
    actor = build_actor(centralized=False, seed=3)
    obs = make_actor_obs(6, N64, centralized=False, seed=4)
    dense = _run(actor, obs, "dense", is_actor=True)
    sparse = _run(actor, obs, "sparse", is_actor=True)
    _assert_close(dense[0], sparse[0], "actor-market fwd dense/sparse")
    _assert_close(dense[1], sparse[1], "actor-market grad dense/sparse")


def test_actor_centralized_dense_eq_sparse():
    actor = build_actor(centralized=True, seed=3)
    obs = make_actor_obs(6, N64, centralized=True, seed=4)
    dense = _run(actor, obs, "dense", is_actor=True)
    sparse = _run(actor, obs, "sparse", is_actor=True)
    _assert_close(dense[0], sparse[0], "actor-centralized fwd dense/sparse")
    _assert_close(dense[1], sparse[1], "actor-centralized grad dense/sparse")


def test_shared_value_dense_eq_sparse():
    shared = build_shared(seed=3)
    obs = make_actor_obs(6, N64, seed=4)
    dense = _run_value(shared, obs, "dense")
    sparse = _run_value(shared, obs, "sparse")
    _assert_close(dense[0], sparse[0], "shared value fwd dense/sparse")
    _assert_close(dense[1], sparse[1], "shared value grad dense/sparse")


def test_centralized_actor_auto_sparse_cutover():
    actor = build_actor(centralized=True, seed=5)
    actor.attention_mode = "auto"
    actor.sparse_attention_min_nodes = N64
    small = make_actor_obs(2, 12, centralized=True, seed=6)
    large = make_actor_obs(2, N64, centralized=True, seed=7)

    small_nodes = small["formula_features"].shape[1] + 1 + small["slot_features"].shape[1]
    large_nodes = large["formula_features"].shape[1] + 1 + large["slot_features"].shape[1]
    assert actor._resolve_attention_mode(small_nodes) == "dense"
    assert actor._resolve_attention_mode(large_nodes) == "sparse"


def test_centralized_actor_defaults_to_auto_when_not_forced():
    actor = build_actor(centralized=True, seed=6)
    assert actor.attention_mode == "auto"


def test_market_actor_defaults_to_dense():
    actor = build_actor(centralized=False, seed=6)
    assert actor.attention_mode == "dense"
