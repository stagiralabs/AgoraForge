"""Learned market mechanism (control_mode='learned').

Pins the mechanism math: leaky no-op at init (state decays, reward zero), the
target bit reaching the mechanism, flat-param round-trip, and determinism. Also
pins the closed-form ``BaselineMarketMechanism`` it must contain as exact points --
the clearing price, midpoint-executed P&L (zero-sum, hand-computed), the YES-share
resolution payout, the is_target-gated bounty curve, and the first-prover bonus.
The batched executor is the live mechanism the resident GPU env runs.
"""
import torch

from agoraforge.envs.math.mechanism import (
    BaselineMarketMechanism,
    BatchedMarketMechanism,
    MarketMechanism,
    MarketMechanismDims,
)


def _executor(spec: MarketMechanism) -> BatchedMarketMechanism:
    return BatchedMarketMechanism([spec.flat_params()], spec.dims)


def _inputs(M=3, A=2, dims=MarketMechanismDims(), seed=1):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, M, A, dims.d_state, generator=g)
    a = torch.randn(1, M, A, dims.d_action, generator=g)
    e = torch.zeros(1, M, A, dims.d_e)
    return s, a, e


def _wired_f_spec(dims, in_idx, c, C=20.0):
    """A zeroed spec whose F pays 2.0 into state coord ``c`` when input ``in_idx``
    is 1, via a sharp gelu gate."""
    spec = MarketMechanism(dims, layernorm=False)
    for p in spec.parameters():
        torch.nn.init.zeros_(p)
    fc1, fc2 = spec.f[0], spec.f[2]
    fc1.weight.data[0, in_idx] = C
    fc1.bias.data[0] = -C / 2.0
    fc2.weight.data[c, 0] = 2.0 / (C / 2.0)
    return spec


# ── mechanism math ──────────────────────────────────────────────────────

def test_zero_init_is_leaky_noop():
    """F's final layer is zero-initialized -> a fresh mechanism leaves only
    the configured state leak and pays nothing: a stable warm start."""
    dims = MarketMechanismDims(state_decay=0.2)
    torch.manual_seed(0)
    op = _executor(MarketMechanism(dims))
    s, a, e = _inputs(dims=dims)
    s_next, reward = op.step(s, a, e)
    assert torch.allclose(s_next, 0.8 * s)
    assert torch.equal(reward, torch.zeros_like(reward))


def test_f_has_one_joint_state_reward_output():
    dims = MarketMechanismDims()
    spec = MarketMechanism(dims)
    assert spec.f[-1].out_features == dims.d_state + 1
    assert not hasattr(spec, "r")


def test_typed_norm_keeps_event_bits_raw():
    """Typed input scaling must not make event gates depend on state/action scale."""
    dims = MarketMechanismDims(input_norm=True, state_scale=10.0, action_scale=2.0)
    tgt_idx = dims.d_state + dims.d_action + 4
    op = _executor(_wired_f_spec(dims, tgt_idx, c=1))
    s = torch.full((1, 2, 1, dims.d_state), 1000.0)
    a = torch.full((1, 2, 1, dims.d_action), -1000.0)
    e = torch.zeros(1, 2, 1, dims.d_e)
    e[0, 0, 0, 4] = 1.0
    with torch.no_grad():
        s_next, _ = op.step(s, a, e)
        paid = (s_next - (1.0 - dims.state_decay) * s)[..., 1]
    assert abs(float(paid[0, 0, 0]) - 2.0) < 0.05
    assert abs(float(paid[0, 1, 0])) < 0.05


def test_public_origin_reaches_mechanism():
    """originally_made_public_by_this_agent is fed after the static target bit."""
    dims = MarketMechanismDims()
    public_idx = dims.d_state + dims.d_action + 5
    op = _executor(_wired_f_spec(dims, public_idx, c=1))
    M, A = 2, 1
    s, a = torch.zeros(1, M, A, dims.d_state), torch.zeros(1, M, A, dims.d_action)
    e = torch.zeros(1, M, A, dims.d_e)
    e[0, 0, 0, 5] = 1.0
    with torch.no_grad():
        s_next, _ = op.step(s, a, e)
        paid = (s_next - s)[..., 1]
    assert abs(float(paid[0, 0, 0]) - 2.0) < 0.05
    assert abs(float(paid[0, 1, 0])) < 0.05


def test_f_reward_reads_state():
    """F can make its reward depend on the current recurrent state."""
    dims = MarketMechanismDims(input_norm=True, state_scale=10.0)
    spec = MarketMechanism(dims, layernorm=False)
    for p in spec.parameters():
        torch.nn.init.zeros_(p)
    spec.f[0].weight.data[0, 0] = 1.0
    spec.f[2].weight.data[-1, 0] = 1.0
    op = _executor(spec)
    s0 = torch.zeros(1, 1, 1, dims.d_state)
    s1 = torch.zeros(1, 1, 1, dims.d_state)
    s1[..., 0] = 10.0
    a = torch.zeros(1, 1, 1, dims.d_action)
    e = torch.zeros(1, 1, 1, dims.d_e)
    with torch.no_grad():
        _, r0 = op.step(s0, a, e)
        _, r1 = op.step(s1, a, e)
    assert r1.shape == (1, 1, 1)
    assert float(r1[0, 0, 0]) > float(r0[0, 0, 0]) + 0.5


def test_is_target_reaches_mechanism():
    """is_target is fed as the last per-agent env bit; a hand-wired F can gate the
    state update on it, so target-conditioned mechanisms (e.g. tgt) are expressible."""
    dims = MarketMechanismDims()
    tgt_idx = dims.d_state + dims.d_action + 4  # is_target within the F input
    op = _executor(_wired_f_spec(dims, tgt_idx, c=1))
    M, A = 2, 1
    s, a = torch.zeros(1, M, A, dims.d_state), torch.zeros(1, M, A, dims.d_action)
    e = torch.zeros(1, M, A, dims.d_e)
    e[0, 0, 0, 4] = 1.0  # formula 0 is a target, formula 1 is not
    with torch.no_grad():
        s_next, _ = op.step(s, a, e)
        paid = (s_next - s)[..., 1]  # change in the gated coord per formula
    assert abs(float(paid[0, 0, 0]) - 2.0) < 0.05  # target market paid
    assert abs(float(paid[0, 1, 0])) < 0.05        # non-target market did not


def test_determinism():
    torch.manual_seed(0)
    spec = MarketMechanism(MarketMechanismDims())
    with torch.no_grad():
        for p in spec.parameters():
            p.add_(0.1 * torch.randn_like(p))
    op = _executor(spec)
    ins = _inputs()
    first = op.step(*ins)
    second = op.step(*ins)
    assert all(torch.equal(x, y) for x, y in zip(first, second))


def test_flat_param_round_trip():
    torch.manual_seed(0)
    op = MarketMechanism(MarketMechanismDims())
    vec = torch.randn(op.n_params)
    op.load_flat_params(vec)
    assert torch.allclose(op.flat_params(), vec, atol=1e-6)


# ── closed-form baselines (BaselineMarketMechanism) ─────────────────────
# Exact points of the mechanism family: flat / fp3 / tgt / bounty. The state is
# s = [p_old, cash, position, marked]; net wealth V = c0 + sum_phi marked.

CASH = 10.0


def _baseline(**kw):
    return BaselineMarketMechanism(initial_cash=CASH, negative_return_penalty=0.0, **kw)


def _env(A, *, resolved_by=None, proven_true=False, theorem_resolved=False,
         is_target=False, M=1):
    e = torch.zeros(M, A, 5)
    if is_target:
        e[..., 4] = 1.0
    if proven_true:
        e[..., 1] = 1.0
    if theorem_resolved:
        e[..., 2] = 1.0
    if resolved_by is not None:
        e[0, resolved_by, 0] = 1.0
    return e


def _value(op, s):
    """Per-agent value g(V), V[j] = c0 + sum_phi w . s[phi, j]."""
    return op.value(s.sum(dim=0))


def _curves(*pairs):
    return torch.tensor([[list(p) for p in pairs]], dtype=torch.float32)  # [1, A, 2]


def test_clearing_price_is_aggregate_demand_ratio():
    op = _baseline()
    a = _curves((2, 0), (0, -1))  # Sq0=2, Sq1=-1 -> price = 2/(2-(-1)) = 2/3
    s = op(torch.zeros(1, 2, 4), a, _env(2))
    assert abs(float(s[0, 0, 0]) - 2.0 / 3.0) < 1e-5  # carried clearing price
    assert torch.allclose(s[0, :, 0], s[0, 0, 0])     # synced across agents


def test_trade_pnl_is_zero_sum_and_hand_computed():
    op = _baseline()
    s = op(torch.zeros(1, 2, 4), _curves((2, 0), (0, -1)), _env(2))
    v = _value(op, s)
    assert abs(float(v[0]) - (CASH + 2.0 / 9.0)) < 1e-5
    assert abs(float(v[1]) - (CASH - 2.0 / 9.0)) < 1e-5
    assert abs(float(v.sum()) - 2 * CASH) < 1e-5, "trading is zero-sum"


def test_resolution_pays_yes_holders():
    op = _baseline()
    a = _curves((2, 0), (0, -1))
    s = op(torch.zeros(1, 2, 4), a, _env(2))
    # Resolve TRUE (proved by agent 0); re-submit the same curves (no further trade).
    s = op(s, a, _env(2, resolved_by=0, proven_true=True, theorem_resolved=True))
    v = _value(op, s)
    # YES shares pay 1: long agent 0 gains, short agent 1 loses; still zero-sum.
    assert abs(float(v[0]) - (CASH + 4.0 / 9.0)) < 1e-5
    assert abs(float(v[1]) - (CASH - 4.0 / 9.0)) < 1e-5


def test_bounty_curve_only_on_targets():
    # Same agent curves; a target market carries the external bounty curve
    # (q1 -= bounty_demand) so it clears at a different price than a plain market.
    plain = _baseline(bounty_demand=1.0)
    a = _curves((1, 0), (1, 0))  # plain: Sq0=2, Sq1=0 -> price 1
    s_plain = plain(torch.zeros(1, 2, 4), a, _env(2, is_target=False))
    tgt = _baseline(bounty_demand=1.0)
    s_tgt = tgt(torch.zeros(1, 2, 4), a, _env(2, is_target=True))  # Sq1=-1 -> 2/3
    assert abs(float(s_plain[0, 0, 0]) - 1.0) < 1e-5
    assert float(s_tgt[0, 0, 0]) < float(s_plain[0, 0, 0]) - 1e-3


def test_first_prover_bonus_paid_to_prover_only():
    op = _baseline(bonus=3.0, trading=False)  # bounty preset: prize, no trading
    s = op(torch.zeros(1, 2, 4), torch.zeros(1, 2, 2),
           _env(2, resolved_by=1, proven_true=True, theorem_resolved=True))
    v = _value(op, s)
    assert abs(float(v[1]) - (CASH + 3.0)) < 1e-5, "prover gets the bonus"
    assert abs(float(v[0]) - CASH) < 1e-5, "non-prover gets nothing"


def test_targets_only_bonus_gating():
    for is_target, expected in [(True, CASH + 3.0), (False, CASH)]:
        op = _baseline(bonus=3.0, targets_only=True, trading=False)
        s = op(torch.zeros(1, 2, 4), torch.zeros(1, 2, 2),
               _env(2, resolved_by=0, proven_true=True, theorem_resolved=True,
                    is_target=is_target))
        assert abs(float(_value(op, s)[0]) - expected) < 1e-5


def test_collaborative_baseline_pays_every_agent_when_anyone_proves_target():
    op = _baseline(bonus=2.0, targets_only=True, team_reward=True, trading=False)
    s = op(torch.zeros(1, 3, 4), torch.zeros(1, 3, 2),
           _env(3, resolved_by=1, proven_true=True, theorem_resolved=True,
                is_target=True))
    v = _value(op, s)
    assert torch.allclose(v, torch.full((3,), CASH + 2.0), atol=1e-5)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS  {fn.__name__}")
