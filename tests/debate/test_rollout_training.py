"""Rollout + PPO integration: sampler/recompute consistency, staged shapes, update step."""

import torch

from agoraforge.conf.schema import base_run_config
from agoraforge.envs.registry import get_env
from agoraforge.envs.debate.env import BatchedDebateEnv
from agoraforge.envs.debate.policy import compute_log_probs_from_staged, sample_actions
from agoraforge.models.factory import build_model_config, build_shared
from agoraforge.training.models import shared_optimizer
from agoraforge.training.ppo import PPOHparams, ppo_update_staged
from agoraforge.training.profiling import NullProfiler
from agoraforge.training.rollout import rollout_resident_config
SPEC = get_env("debate")


def _model(cfg_env):
    run = base_run_config()
    mc = build_model_config(run.actor_model, cfg_env, run.decoding)
    torch.manual_seed(0)
    return build_shared(mc).eval(), run


def test_recomputed_log_probs_match_sampled(debate_cfg):
    model, _ = _model(debate_cfg)
    batch = BatchedDebateEnv.from_config(
        debate_cfg, batch_size=3, device=torch.device('cpu'), seeds=(5, 6, 7))
    obs = batch.actor_obs()
    masks = batch.available_action_masks()
    with torch.no_grad():
        logits = model(obs)
        _, staged = sample_actions(logits, masks, batch)
        recomputed, kls = compute_log_probs_from_staged(logits, staged)
    assert torch.allclose(recomputed, staged['old_log_probs'], atol=1e-5)
    assert torch.isfinite(kls).all()


def test_rollout_and_ppo_update(debate_cfg):
    model, run = _model(debate_cfg)
    staged, metrics = rollout_resident_config(
        debate_cfg, SPEC, batch_size=4, seeds=(1, 2, 3, 4),
        model=model, device=torch.device('cpu'),
        gamma=0.99, gae_lambda=0.95, profiler=NullProfiler(),
    )
    T, B, A = debate_cfg.max_timestep, 4, debate_cfg.n_agents
    rows = T * B * A
    assert staged['old_log_probs'].shape == (rows,)
    assert staged['advantages'].shape == (rows,)
    assert staged['actor_obs']['claim_features'].shape[0] == rows
    for key in ('judge_acc', 'judge_acc_prior', 'judge_acc_full'):
        assert metrics[key].shape == (B,)
        assert metrics[key].ge(0).all() and metrics[key].le(1).all()

    run.training.online_batch_size = 64
    targs = PPOHparams.from_config(run, online_epochs=2, ppo_diagnostics=True)
    model.train()
    opt = shared_optimizer(model, run.training)
    actor_loss, critic_loss, kl_ref, terms = ppo_update_staged(
        model, opt, staged, targs, torch.device('cpu'), 0,
        compute_log_probs_from_staged)
    for v in (actor_loss, critic_loss, kl_ref):
        assert torch.isfinite(torch.tensor(v))
