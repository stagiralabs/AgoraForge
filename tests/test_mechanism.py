"""Flat-theta layout for the protocol-bound debate F,w,P parameterization."""

import torch
import pytest

from agoraforge.envs.mechanism import MechanismDims, bounded_residual
from agoraforge.envs.debate.protocol import (
    BatchedDebateMechanism,
    DebateMechanism,
    DebateMechanismDims,
)


def test_bounded_recurrence_makes_state_divergence_impossible():
    bound, decay = 10.0, 0.1
    state = torch.zeros(4)
    deltas = (
        torch.full_like(state, 1e30),
        torch.full_like(state, float("inf")),
        torch.full_like(state, float("nan")),
        torch.full_like(state, -1e30),
    )
    for step in range(1_000):
        state = bounded_residual(state, deltas[step % len(deltas)], bound, decay)
        assert torch.isfinite(state).all()
        assert state.abs().max() <= bound / decay


def test_mechanism_requires_leaky_state_for_finite_global_bound():
    with pytest.raises(ValueError, match="state_decay"):
        MechanismDims(state_decay=0.0)


def test_debate_mechanism_param_count():
    dims = DebateMechanismDims(d_state=3, d_action=2, hidden=16, predictor_hidden=4)
    mechanism = DebateMechanism(dims)
    f_in = 3 * 3 + 2 + dims.d_e
    f_params = f_in * 16 + 16 + 16 * (3 + 1) + (3 + 1)
    selector_params = 3
    predictor_params = 3 * 4 + 4 + 4 + 1
    assert mechanism.n_params == f_params + selector_params + predictor_params


def test_debate_mechanism_flat_param_order():
    mechanism = DebateMechanism(DebateMechanismDims(d_state=3, d_action=2, hidden=16))
    assert [name for name, _ in mechanism.named_parameters()] == [
        "judge_w",
        "f.0.weight",
        "f.0.bias",
        "f.2.weight",
        "f.2.bias",
        "p.0.weight",
        "p.0.bias",
        "p.2.weight",
        "p.2.bias",
    ]


def test_debate_mechanism_flat_params_determine_all_outputs():
    dims = DebateMechanismDims(d_state=2, d_action=1, hidden=4, predictor_hidden=3)
    spec = DebateMechanism(dims)
    torch.manual_seed(1234)
    theta = torch.randn(spec.n_params) * 0.1
    spec.load_flat_params(theta)
    # The spec round-trips the layout; both executors must decode identically.
    first = BatchedDebateMechanism([theta], dims)
    second = BatchedDebateMechanism([spec.flat_params()], dims)

    generator = torch.Generator().manual_seed(7)
    s = torch.randn((1, 5, 6, 2, 2), generator=generator)
    a = torch.randn((1, 5, 6, 1), generator=generator)
    e = torch.randn((1, 5, 6, dims.d_e), generator=generator)
    transcript = torch.rand((1, 5, 6), generator=generator) > 0.3
    for agent in range(2):
        got = first.turn(s, a, e, agent)
        want = second.turn(s, a, e, agent)
        assert all(torch.equal(x, y) for x, y in zip(got, want))
    assert torch.equal(first.judge_query_scores(s), second.judge_query_scores(s))
    assert torch.equal(first.predict(s, transcript), second.predict(s, transcript))
