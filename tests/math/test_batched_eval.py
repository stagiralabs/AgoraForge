"""Population-batched mechanism/env parity (M1).

Evaluating K candidate mechanisms folded into one env batch must reproduce, per
candidate block, exactly what that mechanism produces run alone -- the whole point
of batching is throughput, not different numbers. Two tiers:

  - A K-candidate ``BatchedMarketMechanism`` == the same thetas run as K
    single-candidate executors (pins the per-candidate weight stacking).
  - A K-candidate ``BatchedMathEnv`` driven by scripted actions == K single-mechanism
    envs driven by the same actions and the same per-step RNG stream.

RNG alignment: candidates in a generation are compared on COMMON random numbers
(same env instance, same seeds), so the batched env draws the per-step
stochasticity once at the base ``B`` and shares it across the K blocks. A single
run with the same ``step_seed`` draws the identical stream, so block ``k`` matches
its standalone run bit-for-bit up to float reassociation in the mechanism matmuls.
"""

import torch
import pytest

from agoraforge.conf.schema import build_env_config
from agoraforge.conf.runs.math.centralized import get_config as cen_config
from agoraforge.envs.math.env import BatchedMathEnv, TensorActions
from agoraforge.envs.math.mechanism import BatchedMarketMechanism, MarketMechanism, MarketMechanismDims


def _dims() -> MarketMechanismDims:
    return MarketMechanismDims(d_state=3, d_action=2, hidden=16)


def _thetas(dims: MarketMechanismDims, K: int, seed: int = 0, layernorm: bool = False,
            scale: float = 0.4) -> list[torch.Tensor]:
    """K distinct, NON-identity mechanisms (perturbed so the residual F actually
    moves the state -- a zero-init fresh mechanism leaves s static and hides bugs)."""
    g = torch.Generator().manual_seed(seed)
    base = MarketMechanism(dims, layernorm=layernorm).flat_params()
    return [base + scale * torch.randn(base.shape, generator=g) for _ in range(K)]


def _step_solo(theta, dims, s, a, e, k, *, layernorm=False):
    solo = BatchedMarketMechanism([theta], dims, layernorm=layernorm)
    with torch.no_grad():
        return solo.step(s[k : k + 1], a[k : k + 1], e[k : k + 1])


def test_batched_mechanism_matches_single():
    dims = _dims()
    K, M, A = 4, 5, 3
    thetas = _thetas(dims, K, seed=1)
    batched = BatchedMarketMechanism(thetas, dims)

    g = torch.Generator().manual_seed(7)
    s = torch.randn(K, M, A, dims.d_state, generator=g)
    a = torch.randn(K, M, A, dims.d_action, generator=g)
    e = torch.randn(K, M, A, dims.d_e, generator=g)

    out, rew = batched.step(s, a, e)
    for k in range(K):
        ref, ref_r = _step_solo(thetas[k], dims, s, a, e, k)
        assert torch.allclose(out[k], ref[0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(rew[k], ref_r[0], atol=1e-5, rtol=1e-4)


def test_batched_mechanism_layernorm_matches_single():
    dims = _dims()
    K, M, A = 3, 4, 2
    thetas = _thetas(dims, K, seed=2, layernorm=True)
    batched = BatchedMarketMechanism(thetas, dims, layernorm=True)
    g = torch.Generator().manual_seed(9)
    s = torch.randn(K, M, A, dims.d_state, generator=g)
    a = torch.randn(K, M, A, dims.d_action, generator=g)
    e = torch.randn(K, M, A, dims.d_e, generator=g)
    out, _ = batched.step(s, a, e)
    for k in range(K):
        ref, _ = _step_solo(thetas[k], dims, s, a, e, k, layernorm=True)
        assert torch.allclose(out[k], ref[0], atol=1e-5, rtol=1e-4)


def test_batched_mechanism_typed_norm_reward_matches_single():
    dims = MarketMechanismDims(
        d_state=3,
        d_action=2,
        hidden=16,
        input_norm=True,
        state_scale=10.0,
        action_scale=1.0,
        state_bound=10.0,
        state_decay=0.1,
    )
    K, M, A = 4, 5, 2
    thetas = _thetas(dims, K, seed=8, scale=0.05)
    batched = BatchedMarketMechanism(thetas, dims)

    g = torch.Generator().manual_seed(11)
    s = torch.randn(K, M, A, dims.d_state, generator=g)
    a = torch.randn(K, M, A, dims.d_action, generator=g)
    e = torch.randn(K, M, A, dims.d_e, generator=g)

    out, rew = batched.step(s, a, e)
    for k in range(K):
        ref, ref_r = _step_solo(thetas[k], dims, s, a, e, k)
        assert torch.allclose(out[k], ref[0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(rew[k], ref_r[0], atol=1e-5, rtol=1e-4)


def _learned_cfg(B_base):
    cfg = cen_config()
    cfg.env.num_theorems = 6
    cfg.env.max_timestep = 20
    cfg.env.fitness_tau = 10.0
    cfg.env.prob_initially_target = 0.5
    cfg.env.prob_initially_resolved = 0.25
    cfg.env.control_mode = "learned"
    cfg.env.learned_state_dim = 3
    return build_env_config(cfg, level=cfg.levels[0])


def test_closed_form_single_arm_uses_population_constructor():
    cfg = cen_config()
    cfg.env.control_mode = "collaborative"
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    dummy = torch.zeros(1)
    env = BatchedMathEnv.batched_from_config(
        vcfg, base_batch_size=3, thetas=[dummy], device="cpu",
        seeds=[1, 2, 3], step_seed=99,
    )
    assert env.K == 1
    assert env.B == 3
    assert env._step_gen.initial_seed() == 99
    with pytest.raises(ValueError, match="exactly one arm"):
        BatchedMathEnv.batched_from_config(
            vcfg, base_batch_size=3, thetas=[dummy, dummy], device="cpu",
            seeds=[1, 2, 3], step_seed=99,
        )


def _scripted_actions(B, A, F, action_dim, g):
    """Deterministic (seeded) action batch; the env masks invalid choices, so the
    values need not be feasible -- only identical across the batched and single runs."""
    return TensorActions(
        math_type=torch.randint(-1, 4, (B, A), generator=g),
        math_formula=torch.randint(0, F, (B, A), generator=g),
        budget=torch.randn(B, A, generator=g),
        math_mode=torch.randint(0, 2, (B, A), generator=g),
        publish_statement=(torch.rand(B, A, F, generator=g) < 0.3),
        publish_proof=(torch.rand(B, A, F, generator=g) < 0.3),
        market_action=torch.randn(B, A, F, action_dim, generator=g),
    )


def _tile_actions(actions: TensorActions, K: int) -> TensorActions:
    def rep(t):
        return t.repeat((K,) + (1,) * (t.dim() - 1))
    return TensorActions(*(rep(getattr(actions, f)) for f in (
        "math_type", "math_formula", "budget", "math_mode",
        "publish_statement", "publish_proof", "market_action")))


def _single_env(vcfg, B_base, theta, seeds, step_seed):
    env = BatchedMathEnv.from_config(vcfg, batch_size=B_base, device="cpu", seeds=seeds)
    env.mechanism = BatchedMarketMechanism([theta], env.mechanism.dims).to(env.device)
    env._step_gen = torch.Generator().manual_seed(step_seed)
    return env


def test_batched_env_rollout_matches_single_runs():
    K, B_base = 3, 5
    vcfg = _learned_cfg(B_base)
    dims = MarketMechanismDims(d_state=vcfg.learned_state_dim, d_action=vcfg.action_dim,
                        hidden=vcfg.learned_mechanism_hidden)
    # Gentle perturbation -> order-1, non-chaotic dynamics (the regime the scale
    # penalty enforces in the real search); a blowing-up mechanism amplifies float
    # reassociation (einsum vs addmm) past any fixed tolerance over many steps.
    thetas = _thetas(dims, K, seed=3, scale=0.05)
    seeds = [100 + b for b in range(B_base)]
    step_seed = 4242
    T = vcfg.max_timestep
    A, F, ad = vcfg.n_agents, vcfg.F_size, vcfg.action_dim

    batched = BatchedMathEnv.batched_from_config(
        vcfg, base_batch_size=B_base, thetas=thetas, device="cpu",
        seeds=seeds, step_seed=step_seed)
    refs = [_single_env(vcfg, B_base, thetas[k], seeds, step_seed) for k in range(K)]

    act_g = torch.Generator().manual_seed(55)
    for _ in range(T):
        base_actions = _scripted_actions(B_base, A, F, ad, act_g)
        b_rewards = batched.step(_tile_actions(base_actions, K))
        for k in range(K):
            r_rewards = refs[k].step(base_actions)
            sl = slice(k * B_base, (k + 1) * B_base)
            assert torch.allclose(batched.s[sl], refs[k].s, atol=1e-5, rtol=1e-4), \
                f"state mismatch candidate {k}"
            assert torch.allclose(b_rewards[sl], r_rewards, atol=1e-5, rtol=1e-4), \
                f"reward mismatch candidate {k}"

    b_fit = batched.fitness()
    b_val = batched.economic_value()
    for k in range(K):
        sl = slice(k * B_base, (k + 1) * B_base)
        assert torch.allclose(b_fit[sl], refs[k].fitness(), atol=1e-5, rtol=1e-4)
        assert torch.allclose(b_val[sl], refs[k].economic_value(), atol=1e-5, rtol=1e-4)
