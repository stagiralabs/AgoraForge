"""Exact-enumeration judge vs an independent brute-force reference."""

import itertools
import math

import pytest
import torch

from agoraforge.envs.debate.judge import judge_marginals


def brute_force_marginals(J, b, claims):
    """Reference marginals over the induced subgraph, via raw enumeration.

    ``claims`` lists the selected claim ids.
    """
    n = len(claims)
    weights = []
    assignments = list(itertools.product([0, 1], repeat=n))
    for x in assignments:
        logw = 0.0
        for i, v in enumerate(claims):
            logw += float(b[v]) * x[i]
        for i, u in enumerate(claims):
            for j, v in enumerate(claims):
                if i < j:
                    logw += float(J[u, v]) * (2 * x[i] - 1) * (2 * x[j] - 1)
        weights.append(logw)
    m = max(weights)
    ws = [math.exp(w - m) for w in weights]
    Z = sum(ws)
    marg = []
    for i in range(n):
        marg.append(sum(w for w, x in zip(ws, assignments) if x[i] == 1) / Z)
    return marg


def _random_graph(N, gen):
    J = torch.randn((N, N), generator=gen) * 0.8
    J = torch.triu(J, diagonal=1)
    J = J + J.T
    # Sparsify roughly half the edges.
    mask = torch.rand((N, N), generator=gen) < 0.5
    mask = torch.triu(mask, diagonal=1)
    mask = mask | mask.T
    J = J * mask
    b = torch.randn(N, generator=gen)
    return J, b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force_full_transcript(seed):
    gen = torch.Generator().manual_seed(seed)
    N, C = 6, 6
    J, b = _random_graph(N, gen)
    slots = torch.arange(C).unsqueeze(0)
    valid = torch.ones((1, C), dtype=torch.bool)
    got = judge_marginals(J.unsqueeze(0), b.unsqueeze(0), slots, valid)[0]
    want = brute_force_marginals(J, b, list(range(C)))
    assert torch.allclose(got, torch.tensor(want, dtype=got.dtype), atol=1e-5)


@pytest.mark.parametrize("seed", [3, 4])
def test_matches_brute_force_induced_subgraph(seed):
    gen = torch.Generator().manual_seed(seed)
    N = 8
    J, b = _random_graph(N, gen)
    claims = [5, 1, 7]
    slots = torch.tensor([claims + [0]])  # one trailing invalid slot
    valid = torch.tensor([[True, True, True, False]])
    got = judge_marginals(J.unsqueeze(0), b.unsqueeze(0), slots, valid)[0]
    want = brute_force_marginals(J, b, claims)
    assert torch.allclose(got[:3], torch.tensor(want, dtype=got.dtype), atol=1e-5)
    # Invalid slot decouples: exactly 0.5.
    assert got[3].item() == pytest.approx(0.5, abs=1e-6)


def test_invalid_slots_do_not_perturb_valid_marginals():
    gen = torch.Generator().manual_seed(9)
    N = 8
    J, b = _random_graph(N, gen)
    claims = [2, 6]
    tight = judge_marginals(
        J.unsqueeze(0), b.unsqueeze(0),
        torch.tensor([claims]), torch.tensor([[True, True]]),
    )[0]
    padded = judge_marginals(
        J.unsqueeze(0), b.unsqueeze(0),
        torch.tensor([claims + [2, 2]]), torch.tensor([[True, True, False, False]]),
    )[0]
    assert torch.allclose(tight, padded[:2], atol=1e-6)


def test_batched_envs_independent():
    gen = torch.Generator().manual_seed(21)
    N = 6
    J1, b1 = _random_graph(N, gen)
    J2, b2 = _random_graph(N, gen)
    slots = torch.tensor([[0, 3], [0, 3]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    got = judge_marginals(torch.stack([J1, J2]), torch.stack([b1, b2]), slots, valid)
    solo1 = judge_marginals(J1.unsqueeze(0), b1.unsqueeze(0), slots[:1], valid[:1])[0]
    solo2 = judge_marginals(J2.unsqueeze(0), b2.unsqueeze(0), slots[1:], valid[1:])[0]
    assert torch.allclose(got[0], solo1, atol=1e-6)
    assert torch.allclose(got[1], solo2, atol=1e-6)
