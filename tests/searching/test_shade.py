"""SHADE on known functions plus the search-harness interface contract.

Pins the DE sampler against closed-form landscapes (a maximizer must drive its
deployable point -- the best member -- onto the optimum) and checks the exact
ask()/tell() shapes, elitism, and pickle/seed determinism that search.py relies on.
"""
import pickle

import numpy as np
import pytest

from agoraforge.searching.shade import SHADE


def _optimize(objective, strategy, generations):
    """Run SHADE (maximizing) and return the final deployable point."""
    for _ in range(generations):
        pop = strategy.ask()
        strategy.tell(np.array([objective(x) for x in pop]))
    return strategy.incumbent, strategy


def _shade(x0, *, sigma=0.3, population=16, seed=0, **kw):
    return SHADE(np.asarray(x0, dtype=float), sigma=sigma,
                  population=population, seed=seed, **kw)


def test_sphere_converges_to_optimum():
    """Maximizing -||x - t||^2 drives the best member onto the target."""
    target = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
    mean, _ = _optimize(lambda x: -np.sum((x - target) ** 2),
                        _shade(np.zeros(5), sigma=0.5, population=20), generations=250)
    assert np.linalg.norm(mean - target) < 1e-2


def test_ill_conditioned_quadratic():
    """A 100:1 anisotropic quadratic: the difference-vector step must adapt to scale."""
    n = 6
    scale = np.geomspace(1.0, 1e2, n)
    target = np.linspace(-1, 1, n)
    mean, _ = _optimize(lambda x: -np.sum(scale * (x - target) ** 2),
                        _shade(np.zeros(n), sigma=0.3, population=24), generations=400)
    assert np.linalg.norm(mean - target) < 5e-2


def test_interface_shapes():
    """gen 0 asks N; every generation after asks 2N; diagnostics have the right types."""
    strategy = _shade(np.zeros(8), population=10)
    first = strategy.ask()
    assert first.shape == (10, 8)
    strategy.tell(np.random.default_rng(0).standard_normal(10))
    second = strategy.ask()
    assert second.shape == (20, 8)
    strategy.tell(np.random.default_rng(1).standard_normal(20))
    assert strategy.incumbent.shape == (8,)
    assert isinstance(strategy.pbest_count, int) and strategy.pbest_count >= 1
    assert isinstance(strategy.sigma, float)
    assert strategy.axis_ratio >= 1.0
    assert strategy.coordinate_variances().shape == (8,)
    keys = strategy.internals()
    for k in ("shade/population", "shade/mean_f", "shade/mean_cr",
              "shade/incumbent_fitness", "shade/success_rate",
              "shade/archive_size", "shade/population_diversity"):
        assert k in keys


def test_population_must_be_at_least_three():
    with pytest.raises(ValueError):
        SHADE(np.zeros(4), sigma=0.1, population=2)


def test_greedy_elitism_is_monotone():
    """On a static (noiseless) objective the best fitness never decreases."""
    target = np.array([0.3, -0.7, 1.2, 0.1])
    strategy = _shade(np.zeros(4), sigma=0.4, population=16, seed=3)
    bests = []
    for _ in range(60):
        pop = strategy.ask()
        strategy.tell(np.array([-np.sum((x - target) ** 2) for x in pop]))
        bests.append(strategy.incumbent_fitness)
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(bests, bests[1:]))


def test_incumbent_uses_current_generation_scores():
    """A lucky score from an earlier seed must not freeze the deployable point."""
    strategy = _shade(np.zeros(3), population=4, seed=3)
    strategy.ask()
    strategy.tell([100.0, 0.0, 0.0, 0.0])

    candidates = strategy.ask()
    scores = np.arange(len(candidates), dtype=float) - 20.0
    strategy.tell(scores)

    current_best = int(np.argmax(strategy.fit))
    assert strategy.incumbent_fitness == strategy.fit[current_best]
    assert strategy.incumbent_fitness < 100.0
    assert np.array_equal(strategy.incumbent, strategy.pop[current_best])


def test_pickle_roundtrip():
    """An unpickled sampler continues the exact candidate stream (RNG state included)."""
    rng = np.random.default_rng(5)
    strategy = _shade(np.zeros(6), population=12, seed=1)
    for _ in range(3):
        pop = strategy.ask()
        strategy.tell(rng.standard_normal(len(pop)))
    clone = pickle.loads(pickle.dumps(strategy))
    assert np.array_equal(strategy.ask(), clone.ask())


def test_deterministic_for_seed():
    """Same seed + same objective stream => identical candidates."""
    a = _shade(np.zeros(9), population=14, seed=7)
    b = _shade(np.zeros(9), population=14, seed=7)
    rng = np.random.default_rng(2)
    for _ in range(5):
        pa, pb = a.ask(), b.ask()
        assert np.array_equal(pa, pb)
        obj = rng.standard_normal(len(pa))
        a.tell(obj)
        b.tell(obj)
