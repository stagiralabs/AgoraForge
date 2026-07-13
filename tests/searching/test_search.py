"""Search-driver and portable-dispatch tests."""

import pytest
import torch

from agoraforge.conf.search.defaults import default as search_defaults
from agoraforge.envs.debate.protocol import DebateMechanism, DebateMechanismDims
from agoraforge.search import build_strategy, fresh_theta
from agoraforge.searching.worker import gpu_queues


def test_fresh_theta_does_not_advance_global_rng():
    torch.manual_seed(91)
    expected = torch.rand(4)
    torch.manual_seed(91)
    fresh_theta(DebateMechanism, DebateMechanismDims(), seed=3, perturb=0.1)
    actual = torch.rand(4)
    assert torch.equal(actual, expected)


def test_search_defaults_name_principled_probes():
    assert list(search_defaults().probes) == ["incumbent", "zero_reward"]


def test_build_strategy_is_fixed_population_shade():
    cfg = search_defaults()
    strategy = build_strategy(cfg, torch.zeros(7))
    assert strategy.n == cfg.population
    assert strategy.ask().shape == (cfg.population, 7)


def test_gpu_queues_are_serial_per_gpu_and_allow_partial_wave():
    queues = gpu_queues(candidate_count=13, chunk_size=3, gpus=["0", "1"])
    assert queues == [
        [(0, 0, 3), (2, 6, 9), (4, 12, 13)],
        [(1, 3, 6), (3, 9, 12)],
    ]
    covered = [i for queue in queues for _, start, end in queue for i in range(start, end)]
    assert sorted(covered) == list(range(13))


@pytest.mark.parametrize("chunk_size,gpus", [(0, ["0"]), (2, [])])
def test_gpu_queues_validate_dispatch(chunk_size, gpus):
    with pytest.raises(ValueError):
        gpu_queues(8, chunk_size, gpus)
