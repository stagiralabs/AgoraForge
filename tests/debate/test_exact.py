"""Exact low-treewidth debate inference checks."""

import itertools

import torch

from agoraforge.envs.debate.exact import (
    eliminate_log_partition,
    exact_truth_and_marginals,
    min_fill_order,
)


def _brute(couplings, unary):
    states = torch.tensor(list(itertools.product((0.0, 1.0), repeat=len(unary))))
    spins = 2.0 * states - 1.0
    log_weight = states @ unary + torch.einsum("bi,ij,bj->b", spins, couplings, spins) * 0.5
    probability = torch.softmax(log_weight, dim=0)
    return torch.logsumexp(log_weight, dim=0), (probability[:, None] * states).sum(dim=0)


def test_exact_elimination_matches_enumeration():
    generator = torch.Generator().manual_seed(4)
    n = 7
    raw = torch.randn(n, n, generator=generator) * 0.3
    adjacency = torch.rand(n, n, generator=generator) < 0.3
    couplings = torch.triu(raw * adjacency, diagonal=1)
    couplings = couplings + couplings.T
    unary = torch.randn(n, generator=generator)

    differentiable = unary.clone().requires_grad_(True)
    log_z, _, _ = eliminate_log_partition(couplings, differentiable)
    (marginals,) = torch.autograd.grad(log_z, differentiable)
    expected_z, expected_marginals = _brute(couplings, unary)
    assert torch.allclose(log_z, expected_z, atol=2e-6, rtol=0)
    assert torch.allclose(marginals, expected_marginals, atol=2e-6, rtol=0)


def test_exact_sampler_has_correct_marginals():
    couplings = torch.tensor([
        [0.0, 0.7, 0.0],
        [0.7, 0.0, -0.4],
        [0.0, -0.4, 0.0],
    ])
    unary = torch.tensor([[0.3, -0.2, 0.8]])
    _, expected = _brute(couplings, unary[0])
    generator = torch.Generator().manual_seed(8)
    samples = [
        exact_truth_and_marginals(couplings[None], unary, generator)[0][0]
        for _ in range(4000)
    ]
    empirical = torch.stack(samples).float().mean(dim=0)
    assert torch.allclose(empirical, expected, atol=0.035, rtol=0)


def test_min_fill_reports_tree_width_one():
    adjacency = torch.zeros(8, 8, dtype=torch.bool)
    for i in range(7):
        adjacency[i, i + 1] = adjacency[i + 1, i] = True
    _, width = min_fill_order(adjacency)
    assert width == 1
