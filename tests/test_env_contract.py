"""Registry contract: every registered env satisfies the BatchedEnv protocol."""

import pytest
import torch

from agoraforge.envs.interface import BatchedEnv, PolicyInputs
from agoraforge.envs.registry import available, get_env

# Minimal per-env config kwargs for a tiny contract-check instance.
_SMALL_CONFIG_KWARGS = {
    "debate": dict(
        num_claims=8,
        n_agents=2,
        max_timestep=4,
        judge_claim_cap=4,
        claim_graph_m=2.0,
        coupling_min=0.2,
        coupling_max=1.0,
        p_attack=0.3,
        unary_cutoff=3.0,
        truth_gibbs_sweeps=3,
        marginal_gibbs_sweeps=3,
    ),
    "math": dict(
        num_theorems=6,
        n_agents=2,
        max_timestep=6,
        theorem_graph_m=1.5,
        n_weight_levels=9,
        weight_min=0.0,
        weight_max=1.0,
        weight_penalty=4.0,
        truth_gibbs_sweeps=3,
        truth_unary_cutoff=2.0,
        weight_unary_cutoff=2.0,
        initial_cash=1.0,
        negative_return_penalty=1.0,
        prob_initially_resolved=0.2,
        prob_initially_target=0.2,
        rho=0.4,
        eta=2.0,
        horizon_H=1000,
        query_truth_prior_correct=0.8,
        query_weight_noise=0.1,
        query_num_related=5,
        bounty_demand=10.0,
    ),
}


def _small_config(spec):
    return spec.config_cls(**_SMALL_CONFIG_KWARGS[spec.name])


@pytest.mark.parametrize("name", available())
def test_env_satisfies_contract(name):
    spec = get_env(name)
    cfg = _small_config(spec)
    assert cfg.env_name == name

    B = 3
    batch = spec.batch_cls.from_config(
        cfg, batch_size=B, device=torch.device("cpu"), seeds=list(range(B)))
    assert isinstance(batch, BatchedEnv)
    assert batch.B == B and batch.A >= 1
    assert batch.s.dim() == 4 and batch.s.shape[0] == B

    inputs = batch.policy_inputs()
    assert isinstance(inputs, PolicyInputs)
    for group in (inputs.actor_obs, inputs.masks):
        assert isinstance(group, dict) and group

    actions, staged = spec.policy.sample_actions(
        _uniform_logits(spec, inputs, batch), inputs.masks, batch)
    actions = spec.policy.to_env_actions(actions, inputs.ctx, batch)
    rewards = batch.step(actions)
    assert rewards.shape[:2] == (B, batch.A)
    assert "old_log_probs" in staged

    fitness = batch.fitness()
    assert fitness.shape == (B,) and torch.isfinite(fitness).all()
    metrics = batch.eval_metrics()
    assert set(spec.metric_keys) <= set(metrics)
    assert spec.primary_metric in metrics
    for key in spec.metric_keys:
        assert metrics[key].shape == (B,)


def _uniform_logits(spec, inputs, batch):
    """Neutral logits shaped like the env's action head output, via the real model."""
    from agoraforge.models.factory import build_actor, build_model_config
    from agoraforge.conf.schema import base_run_config

    run = base_run_config(batch.cfg.env_name)
    torch.manual_seed(0)
    actor = build_actor(build_model_config(run.actor_model, batch.cfg, run.decoding)).eval()
    with torch.no_grad():
        return actor(inputs.actor_obs)
