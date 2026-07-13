"""Resident (GPU-tensor) path for the centralized planner.

The centralized planner runs on the resident path with shared knowledge (no per-agent visibility, no publish action)
and a reward that is the per-step increment of the outer fitness, so the
undiscounted return equals ``fitness`` (mean over targets of
``exp(-t_first / tau)``).
"""
import torch

from agoraforge.conf.schema import build_env_config
from agoraforge.conf.runs.math.centralized import get_config as cen_config
from agoraforge.envs.math.env import BatchedMathEnv, TensorActions
from agoraforge.envs.math.obs import (
    FORMULA_FEATURE_DIM,
    SLOT_EDGE_DIM,
    SLOT_FEATURE_DIM,
)


def _make_batch(*, num_theorems=12, max_timestep=50, fitness_tau=10.0, B=4, seed=0):
    cfg = cen_config()
    cfg.env.num_theorems = num_theorems
    cfg.env.max_timestep = max_timestep
    cfg.env.fitness_tau = fitness_tau
    cfg.env.prob_initially_target = 0.25
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    seeds = [seed + b for b in range(B)]
    return BatchedMathEnv.from_config(vcfg, batch_size=B, device="cpu", seeds=seeds)


def _noop_actions(batch):
    B, A, F = batch.B, batch.A, batch.F
    return TensorActions(
        math_type=torch.full((B, A), -1, dtype=torch.long),
        math_formula=torch.full((B, A), -1, dtype=torch.long),
        budget=torch.full((B, A), -1.0),
        math_mode=torch.zeros((B, A), dtype=torch.long),
        publish_statement=torch.zeros((B, A, F), dtype=torch.bool),
        publish_proof=torch.zeros((B, A, F), dtype=torch.bool),
        market_action=torch.zeros((B, A, F, batch.cfg.action_dim)),
    )


def _unresolved_targets(batch):
    is_t = batch.is_target[:, : batch.N].bool() | batch.is_target[:, batch.N :].bool()
    resolved = batch.public_resolved[:, : batch.N] | batch.public_resolved[:, batch.N :]
    return is_t & ~resolved  # (B, N)


def test_centralized_config_builds_resident_batch():
    batch = _make_batch()
    assert batch.centralized


def test_centralized_has_no_mechanism():
    batch = _make_batch()
    assert batch.centralized
    assert batch.mechanism is None
    # No market state is allocated and economic value is identically zero.
    assert not hasattr(batch, "s_a")
    assert torch.equal(batch.economic_value(), torch.zeros((batch.B, batch.A)))


def test_obs_widths_and_slot_nodes():
    batch = _make_batch()
    actor_obs = batch.actor_obs()
    B, A, F = batch.B, batch.A, batch.F
    # One joint graph per env: B rows, one slot per model.
    assert actor_obs["formula_features"].shape == (B, F, FORMULA_FEATURE_DIM)
    assert actor_obs["slot_features"].shape == (B, A, SLOT_FEATURE_DIM)
    assert actor_obs["slot_formula_edges"].shape == (B, A, F, SLOT_EDGE_DIM)


def test_joint_actor_decodes_every_slot():
    from agoraforge.models.factory import build_actor, build_model_config

    batch = _make_batch()
    run = cen_config()
    actor = build_actor(build_model_config(run.actor_model, batch.cfg, run.decoding))
    logits = actor(batch.actor_obs())
    # One forward over B env graphs yields one action row per (env, model).
    assert logits["math_type_logits"].shape[0] == batch.B * batch.A
    # The per-model formula pointer scores every formula for every model.
    assert logits["formula_logits"].shape == (batch.B * batch.A, batch.F)


def test_shared_value_one_per_env():
    from agoraforge.models.factory import build_model_config, build_shared

    batch = _make_batch()
    run = cen_config()
    shared = build_shared(build_model_config(run.actor_model, batch.cfg, run.decoding))
    _, values = shared.forward_actor_critic(batch.actor_obs())
    # One team value per env -- not one per (env, model).
    assert values.shape == (batch.B,)


def test_proof_is_shared_to_all_agents():
    batch = _make_batch()
    tgt = _unresolved_targets(batch)
    b = 0
    theorem = int(tgt[b].nonzero().flatten()[0])
    # One agent resolves the target formula; the step must publish it to all.
    batch.resolved[b, 0, theorem] = True
    batch.step(_noop_actions(batch))
    assert bool(batch.public_resolved[b, theorem])
    assert bool(batch.resolved[b, :, theorem].all())
    assert bool(batch.concrete[b, :, theorem].all())


def test_reward_equals_fitness():
    batch = _make_batch(num_theorems=12, max_timestep=20, fitness_tau=8.0, B=4, seed=3)
    tau = float(batch.cfg.fitness_tau)
    is_t = batch.is_target[:, : batch.N].bool() | batch.is_target[:, batch.N :].bool()
    n_targets = is_t.sum(dim=1).clamp_min(1).float()

    total = torch.zeros((batch.B, batch.A))
    # Resolve one fresh target per env every few steps so resolutions land at
    # known, distinct timesteps; the per-step team reward must equal the fitness
    # mass added that step.
    for step in range(1, batch.cfg.max_timestep + 1):
        if step % 4 == 0:
            tgt = _unresolved_targets(batch)
            for b in range(batch.B):
                ids = tgt[b].nonzero().flatten()
                if len(ids):
                    batch.resolved[b, 0, int(ids[0])] = True
        before = _unresolved_targets(batch).sum(dim=1)
        rewards = batch.step(_noop_actions(batch))
        rewards = rewards.squeeze(-1)
        after = _unresolved_targets(batch).sum(dim=1)
        newly = (before - after).float()
        expected = torch.exp(torch.tensor(-step / tau)) * newly / n_targets
        # Reward is identical across agents (team reward).
        assert torch.allclose(rewards[:, 0], rewards[:, 1])
        assert torch.allclose(rewards[:, 0], expected, atol=1e-6)
        total += rewards

    fitness = batch.fitness()
    assert torch.allclose(total[:, 0], fitness, atol=1e-6)
    assert (fitness > 0).any()
