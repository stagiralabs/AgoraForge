"""Resident-path fitness matches the reference scalar objective.

The outer mechanism search ranks mechanisms by the horizon-robust fitness
``mean_i exp(-t_first/tau)``; here we pin that the resident batched path
(``BatchedMathEnv.target_resolution_step`` / ``fitness``) reproduces the
reference scalar objective. The four regimes the metric must distinguish:

  - nothing resolved          -> fitness == 0
  - a single target resolved late, all others early -> fitness drops well below the
    resolved fraction (the horizon-degeneracy fix)
  - H-invariance: stamping the same resolution step at horizon H and 2H gives
    the same fitness (the metric depends on *when*, not the horizon length)
  - the resident per-env fitness equals the scalar objective on the same per-target
    resolution-step vector.
"""
import math

import torch

from agoraforge.conf.runs.math.centralized import get_config as cen_config
from agoraforge.conf.schema import build_env_config
from agoraforge.envs.math.env import BatchedMathEnv


def _scalar_fitness(times, tau):
    vals = [math.exp(-float(t) / tau) for t in times if t is not None]
    return sum(vals) / len(times) if times else 0.0


def _make_batch(num_theorems, max_timestep, fitness_tau, *, B=4, seed=0):
    cfg = cen_config()
    cfg.env.num_theorems = num_theorems
    cfg.env.max_timestep = max_timestep
    cfg.env.fitness_tau = fitness_tau
    cfg.env.prob_initially_target = 0.25
    cfg.env.control_mode = "learned"
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    seeds = [seed + b for b in range(B)]
    return BatchedMathEnv.from_config(vcfg, batch_size=B, device="cpu", seeds=seeds)


def _target_theorems(batch):
    """(B, list[int]) target theorem ids per env (either-sign is_target)."""
    is_t = batch.is_target[:, : batch.N].bool() | batch.is_target[:, batch.N :].bool()
    return [is_t[b].nonzero().flatten().tolist() for b in range(batch.B)]


def _resolve(batch, b, theorem_id):
    """Force-resolve theorem ``theorem_id`` in env ``b`` (mirror a publication)."""
    batch.public_resolved[b, theorem_id] = True


def test_nothing_resolved_rate_zero():
    batch = _make_batch(12, 60, 10.0)
    # Targets begin unresolved (prob_initially_resolved=0): no steps -> fitness 0.
    fitness = batch.fitness()
    assert torch.allclose(fitness, torch.zeros_like(fitness))


def test_all_unresolved_after_horizon_rate_zero():
    batch = _make_batch(12, 30, 10.0)
    for _ in range(30):
        batch.timestep += 1
        batch._record_target_resolutions()  # nothing resolved -> stays sentinel
    fitness = batch.fitness()
    assert torch.allclose(fitness, torch.zeros_like(fitness))


def test_late_target_low_rate_high_frac():
    tau, H = 10.0, 60
    batch = _make_batch(16, H, tau)
    targets = _target_theorems(batch)
    # Pick an env with >= 2 targets so "one late" is meaningful.
    b = next(i for i, t in enumerate(targets) if len(t) >= 2)
    tids = targets[b]
    for step in range(1, H + 1):
        batch.timestep[b] = step
        if step == 1:
            for tid in tids[:-1]:
                _resolve(batch, b, tid)  # all but one resolve early
        if step == H:
            _resolve(batch, b, tids[-1])  # last resolves at the horizon
        batch._record_target_resolutions()
    fitness = batch.fitness()[b].item()
    frac = (batch.resolved_target_count()[b].item() / len(tids))
    assert frac == 1.0  # everything resolved...
    # ...yet the late one drags the fitness below what an all-early resolve gives.
    all_early = (len(tids) * math.exp(-1.0 / tau)) / len(tids)
    assert fitness < all_early
    assert fitness < frac  # the degeneracy the fitness objective fixes


def test_rate_h_invariant():
    """Same resolution step at H and 2H -> same fitness (depends on when, not H)."""
    tau, resolve_step = 10.0, 5
    fitnesses = {}
    for H in (30, 60):
        batch = _make_batch(16, H, tau, seed=7)
        targets = _target_theorems(batch)
        for step in range(1, H + 1):
            batch.timestep[:] = step
            if step == resolve_step:
                for b, tids in enumerate(targets):
                    for tid in tids:
                        _resolve(batch, b, tid)
            batch._record_target_resolutions()
        fitnesses[H] = batch.fitness()
    assert torch.allclose(fitnesses[30], fitnesses[60], atol=1e-6)


def test_resident_matches_scalar_fitness():
    """Per-env resident fitness equals the scalar objective on the same step vector."""
    tau, H = 10.0, 60
    batch = _make_batch(16, H, tau, seed=3)
    targets = _target_theorems(batch)
    # Resolve each env's targets at a deterministic, env-varying schedule.
    for step in range(1, H + 1):
        batch.timestep[:] = step
        for b, tids in enumerate(targets):
            for k, tid in enumerate(tids):
                if step == (b + k) % H + 1:
                    _resolve(batch, b, tid)
        batch._record_target_resolutions()

    resident_fitness = batch.fitness()
    for b, tids in enumerate(targets):
        steps = batch.target_resolution_step[b]
        # Build the reference per-target time list (None when at the sentinel).
        times = [
            None if steps[tid].item() >= batch._fitness_sentinel else int(steps[tid].item())
            for tid in tids
        ]
        expected = _scalar_fitness(times, tau)
        assert math.isclose(resident_fitness[b].item(), expected, rel_tol=1e-6, abs_tol=1e-6)
