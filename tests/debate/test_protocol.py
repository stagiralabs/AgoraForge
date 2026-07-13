"""Protocol-bound F,w,P behavior and batched parameter parity."""

import torch

from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX
from agoraforge.envs.debate.env import BatchedDebateEnv, TensorActions
from agoraforge.envs.debate.protocol import (
    BatchedDebateMechanism,
    DebateMechanism,
    DebateMechanismDims,
    VanillaDebateMechanism,
    mechanism_from_config,
)


from tests.conftest import small_config

PASS = REVEAL_TYPE_TO_IDX["pass"]


def test_empty_mechanism_path_builds_fresh_noop(debate_cfg):
    debate_cfg.control_mode = "learned"
    debate_cfg.learned_mechanism_params = ""
    batch = BatchedDebateEnv.from_config(
        debate_cfg, batch_size=1, device=torch.device("cpu"), seeds=[0]
    )
    assert isinstance(batch.mechanism, BatchedDebateMechanism)
    assert batch.mechanism.K == 1


def _make(cfg, B=4, seeds=(1, 2, 3, 4)):
    return BatchedDebateEnv.from_config(cfg, batch_size=B, device=torch.device("cpu"), seeds=seeds)


def _pass_actions(batch, signals=None):
    if signals is None:
        signals = torch.zeros((batch.B, batch.A, batch.N, batch.cfg.action_dim))
    return TensorActions(
        reveal_type=torch.full((batch.B, batch.A), PASS, dtype=torch.long),
        reveal_claim=torch.full((batch.B, batch.A), -1, dtype=torch.long),
        signals=signals,
    )


def test_fresh_learned_protocol_is_noop_with_half_prediction():
    cfg = small_config(control_mode="learned", max_timestep=3)
    batch = _make(cfg)
    total = torch.zeros(batch.B, batch.A)
    for _ in range(cfg.max_timestep):
        total += batch.step(_pass_actions(batch)).squeeze(-1)
    assert batch.s.eq(0).all()
    assert total.eq(0).all()
    assert torch.allclose(batch.final_p, torch.full((batch.B,), 0.5))


def test_only_active_debater_state_and_reward_update():
    cfg = small_config(control_mode="learned", max_timestep=3)
    batch = _make(cfg)
    dims = batch.mechanism.dims
    theta = DebateMechanism(dims).flat_params()
    theta += 0.1 * torch.randn_like(theta)
    batch.mechanism = BatchedDebateMechanism([theta], dims)
    before = batch.s.clone()
    reward = batch.step(_pass_actions(batch)).squeeze(-1)
    assert not torch.allclose(batch.s[:, :, 0], before[:, :, 0])
    assert torch.equal(batch.s[:, :, 1], before[:, :, 1])
    assert reward[:, 1].eq(0).all()


def test_flat_params_roundtrip_includes_selector_and_predictor():
    dims = DebateMechanismDims(d_state=3, d_action=2)
    mechanism = DebateMechanism(dims)
    theta = mechanism.flat_params()
    mechanism.load_flat_params(theta + 0.1)
    assert torch.allclose(mechanism.flat_params(), theta + 0.1)
    names = [name for name, _ in mechanism.named_parameters()]
    assert names[0] == "judge_w"
    assert mechanism.f[-1].out_features == dims.d_state + 1
    assert not hasattr(mechanism, "r")
    assert any(name.startswith("p.") for name in names)


def test_batched_mechanism_matches_individual_F_w_and_P():
    dims = DebateMechanismDims(d_state=3, d_action=2)
    torch.manual_seed(0)
    thetas = []
    for _ in range(3):
        theta = DebateMechanism(dims).flat_params()
        thetas.append(theta + 0.05 * torch.randn_like(theta))
    solos = [BatchedDebateMechanism([theta], dims) for theta in thetas]
    batched = BatchedDebateMechanism(torch.stack(thetas), dims)

    K, B, N, A = 3, 2, 7, 2
    s = torch.randn(K, B, N, A, dims.d_state)
    a = torch.randn(K, B, N, dims.d_action)
    e = torch.randn(K, B, N, dims.d_e)
    transcript = torch.rand(K, B, N) > 0.3
    for agent in range(A):
        got_s, got_r = batched.turn(s, a, e, agent)
        for k, solo in enumerate(solos):
            want_s, want_r = solo.turn(s[k : k + 1], a[k : k + 1], e[k : k + 1], agent)
            assert torch.allclose(got_s[k], want_s[0], atol=1e-5)
            assert torch.allclose(got_r[k], want_r[0], atol=1e-5)
    got_q = batched.judge_query_scores(s)
    got_p = batched.predict(s, transcript)
    for k, solo in enumerate(solos):
        assert torch.allclose(got_q[k], solo.judge_query_scores(s[k : k + 1])[0], atol=1e-5)
        assert torch.allclose(
            got_p[k], solo.predict(s[k : k + 1], transcript[k : k + 1])[0], atol=1e-5
        )


def test_mechanism_from_config_dispatch():
    vanilla = mechanism_from_config(small_config(control_mode="vanilla"))
    learned = mechanism_from_config(small_config(control_mode="learned"))
    assert isinstance(vanilla, VanillaDebateMechanism)
    assert isinstance(learned, BatchedDebateMechanism)
