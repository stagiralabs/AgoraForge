"""Population-batched inner-training smoke + isolation.

``evaluate_population`` trains K candidate mechanisms' policies in one batched run:
one env batch of ``B = K * B_base`` on common random numbers, K stacked shared
actor/critic policies vmapped, K independent PPO updates.

The production path samples actions once over the folded K batch for throughput, so
exact K=1-vs-K policy-RNG parity is not a contract. These tests pin the cheaper
invariants: finite per-candidate outputs and CRN clone/chunk equality.
Batched-versus-solo environment tests separately pin candidate isolation.
"""

import torch

from agoraforge.conf.schema import build_env_config
from agoraforge.conf.runs.math.learned_tiny import get_config
from agoraforge.training.population.evaluation import evaluate_population
from agoraforge.envs.math.env import BatchedMathEnv
from agoraforge.envs.math.mechanism import theta_from_config

def _cfg():
    cfg = get_config()
    cfg.env.num_theorems = 8
    cfg.env.n_agents = 2
    cfg.env.obs_formula_cap = 6
    cfg.env.max_timestep = 8
    cfg.env.fitness_tau = 10.0
    cfg.env.prob_initially_target = 0.5
    cfg.env.prob_initially_resolved = 0.25
    cfg.training.online_batch_size = 4096
    return cfg


def _thetas(vcfg, K, seed=3, scale=0.06):
    base = theta_from_config(vcfg)
    g = torch.Generator().manual_seed(seed)
    return [base + scale * torch.randn(base.shape, generator=g) for _ in range(K)]


def _kw(cfg, vcfg, **over):
    base = dict(cfg=cfg, vcfg=vcfg, base_batch_size=4, inner_epochs=3, tail_epochs=2,
                seed=11, device="cpu")
    base.update(over)
    return base


def test_population_returns_finite_per_candidate_metrics():
    cfg = _cfg()
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    K = 3
    thetas = _thetas(vcfg, K)
    info = evaluate_population(thetas, policy_seed_base=100, **_kw(cfg, vcfg))
    for value in info["discounted_resolved"]:
        assert torch.isfinite(torch.tensor(value))
    for key in ("discounted_resolved", "resolved_frac"):
        assert len(info[key]) == K
        assert torch.isfinite(torch.tensor(info[key])).all()


def test_crn_clones_identical_and_chunk_invariant():
    """Under CRN every candidate sees the same policy init and RNG streams, so a
    cloned theta scores identically to its twin in the same fold, and scores
    are invariant to how the population is chunked."""
    cfg = _cfg()
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    t = _thetas(vcfg, 3)
    bank = [t[0], t[0].clone(), t[1], t[2]]

    obj = evaluate_population(bank, policy_seed_base=100, crn=True, **_kw(cfg, vcfg))["discounted_resolved"]
    assert obj[0] == obj[1], "CRN clones in one fold must score identically"

    chunked = evaluate_population(bank, policy_seed_base=100, crn=True, chunk_size=2,
                                  **_kw(cfg, vcfg))["discounted_resolved"]
    assert chunked == obj, "CRN scores must not depend on chunking"

    plain = evaluate_population(bank, policy_seed_base=100, **_kw(cfg, vcfg))["discounted_resolved"]
    assert plain[0] != plain[1], "without CRN the clone pair should differ (seed lottery)"


def test_population_uses_capped_environment_observation_contract(monkeypatch):
    """Search must not bypass top-k selection and train on the full theorem graph."""
    cfg = _cfg()
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    observed_widths = []
    original_policy_inputs = BatchedMathEnv.policy_inputs

    def checked_policy_inputs(batch):
        inputs = original_policy_inputs(batch)
        observed_widths.append(inputs.actor_obs["formula_features"].shape[1])
        return inputs

    monkeypatch.setattr(BatchedMathEnv, "policy_inputs", checked_policy_inputs)
    evaluate_population(
        _thetas(vcfg, 2),
        cfg=cfg,
        vcfg=vcfg,
        base_batch_size=4,
        inner_epochs=1,
        tail_epochs=1,
        seed=11,
        device="cpu",
        policy_seed_base=100,
        crn=True,
    )
    assert observed_widths
    assert set(observed_widths) == {cfg.env.obs_formula_cap}
