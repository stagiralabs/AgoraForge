"""Debate lifecycle: transcript, terminal judge, and full-marginal objective."""

import torch

from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX
from agoraforge.envs.debate.env import BatchedDebateEnv, TensorActions
from agoraforge.envs.debate.judge import judge_marginals
from tests.conftest import small_config

REVEAL = REVEAL_TYPE_TO_IDX["reveal"]
PASS = REVEAL_TYPE_TO_IDX["pass"]


def _make(cfg, B=4, seeds=(1, 2, 3, 4)):
    return BatchedDebateEnv.from_config(cfg, batch_size=B, device=torch.device("cpu"), seeds=seeds)


def _pass_actions(batch):
    return TensorActions(
        reveal_type=torch.full((batch.B, batch.A), PASS, dtype=torch.long),
        reveal_claim=torch.full((batch.B, batch.A), -1, dtype=torch.long),
        signals=torch.zeros((batch.B, batch.A, batch.N, batch.cfg.action_dim)),
    )


def _reveal(batch, agent, claims):
    actions = _pass_actions(batch)
    actions.reveal_type[:, agent] = REVEAL
    actions.reveal_claim[:, agent] = claims
    return actions


def _first_unrevealed(batch):
    return (~batch.in_transcript).to(torch.float32).argmax(dim=1)


def test_reset_has_target_prior_and_no_judge_feedback(debate_cfg):
    batch = _make(debate_cfg)
    assert torch.equal(batch.slots[:, 0], batch.target_idx)
    assert batch.slot_count.eq(1).all()
    b_target = batch.unary.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
    assert torch.allclose(batch.prior_p, torch.sigmoid(b_target), atol=1e-6)
    assert not batch.judge_selected.any()
    assert batch.judge_p.eq(0).all()


def test_turns_alternate_and_unpermitted_reveals_are_dropped(debate_cfg):
    batch = _make(debate_cfg)
    masks = batch.available_action_masks()
    assert masks["reveal_type_mask"][:, 0, REVEAL].eq(1).all()
    assert masks["reveal_type_mask"][:, 1, REVEAL].eq(0).all()
    claim = _first_unrevealed(batch)
    batch.step(_reveal(batch, 1, claim))
    assert batch.slot_count.eq(1).all()
    batch.step(_reveal(batch, 1, claim))
    assert batch.slot_count.eq(2).all()
    assert batch.revealed_by[:, :, 1].gather(1, claim.unsqueeze(1)).all()


def test_transcript_is_bounded_by_horizon_not_judge_budget():
    cfg = small_config(control_mode="learned", max_timestep=4, judge_claim_cap=2)
    batch = _make(cfg)
    for step in range(cfg.max_timestep):
        batch.step(_reveal(batch, step % 2, _first_unrevealed(batch)))
    assert batch.slot_count.eq(5).all()
    assert batch.in_transcript.sum(dim=1).eq(5).all()
    assert batch.judge_slot_valid.sum(dim=1).eq(2).all()
    assert batch.judge_selected.sum(dim=1).eq(2).all()


def test_learned_judge_runs_only_at_terminal_and_matches_direct_call():
    cfg = small_config(control_mode="learned", max_timestep=3, judge_claim_cap=2)
    batch = _make(cfg)
    for step in range(2):
        batch.step(_reveal(batch, step % 2, _first_unrevealed(batch)))
        assert not batch.judge_selected.any()
        assert batch.judge_p.eq(0).all()
    batch.step(_reveal(batch, 0, _first_unrevealed(batch)))
    direct = judge_marginals(
        batch.couplings, batch.unary, batch.judge_slots, batch.judge_slot_valid
    )
    observed = batch.judge_p.gather(1, batch.judge_slots)
    assert torch.allclose(observed, direct, atol=1e-6)


def test_vanilla_judges_full_transcript_and_pays_terminal_zero_sum():
    cfg = small_config(control_mode="vanilla", max_timestep=4)
    batch = _make(cfg)
    rewards = []
    for step in range(cfg.max_timestep):
        rewards.append(batch.step(_reveal(batch, step % 2, _first_unrevealed(batch))).squeeze(-1))
    assert torch.stack(rewards[:-1]).eq(0).all()
    assert torch.equal(batch.judge_slots, batch.slots)
    assert torch.equal(batch.judge_slot_valid, batch.slot_valid)
    target_p = batch.judge_p.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
    assert torch.allclose(batch.final_p, target_p)
    assert torch.allclose(rewards[-1][:, 0], target_p - 0.5, atol=1e-6)
    assert torch.allclose(rewards[-1].sum(dim=1), torch.zeros(batch.B), atol=1e-6)


def test_fitness_compares_prediction_to_full_graph_marginal(debate_cfg):
    batch = _make(debate_cfg)
    target_full = batch.p_full.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
    batch.final_p = target_full.clone()
    assert torch.allclose(batch.fitness(), torch.ones(batch.B))
    anchors = batch.anchor_metrics()
    assert torch.allclose(anchors["acc_full"], torch.ones(batch.B))
    assert torch.allclose(
        anchors["acc_prior"], 1.0 - (batch.prior_p - target_full).pow(2)
    )


def test_same_seed_same_instance(debate_cfg):
    a = _make(debate_cfg, B=2, seeds=(7, 8))
    b = _make(debate_cfg, B=2, seeds=(7, 8))
    for name in ("couplings", "unary", "truth", "p_full", "target_idx"):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_couplings_symmetric_zero_diag(debate_cfg):
    batch = _make(debate_cfg)
    assert torch.equal(batch.couplings, batch.couplings.transpose(1, 2))
    assert batch.couplings.diagonal(dim1=1, dim2=2).eq(0).all()
