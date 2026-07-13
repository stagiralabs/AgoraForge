"""Exact low-treewidth inference for the binary debate claim graph.

This is an experimental alternative to Gibbs.  Production BA(m=2) graphs have
small min-fill elimination width (median 5, p99 7 in a 512-graph probe), so exact
variable elimination is potentially practical despite having 32 binary claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EliminationConditional:
    variable: int
    remaining: tuple[int, ...]
    # Axes are (variable, *remaining), each binary.
    log_prob: torch.Tensor


def min_fill_order(adjacency: torch.Tensor) -> tuple[list[int], int]:
    """Deterministic min-fill order and induced width for one boolean graph."""
    n = int(adjacency.shape[0])
    neighbors = [
        {int(j) for j in torch.nonzero(adjacency[i], as_tuple=False).flatten().tolist()}
        for i in range(n)
    ]
    active = set(range(n))
    order: list[int] = []
    width = 0
    while active:
        def score(v: int):
            ns = sorted(neighbors[v] & active)
            fill = sum(
                int(b not in neighbors[a])
                for pos, a in enumerate(ns)
                for b in ns[pos + 1:]
            )
            return fill, len(ns), v

        variable = min(active, key=score)
        ns = list(neighbors[variable] & active)
        width = max(width, len(ns))
        for pos, a in enumerate(ns):
            for b in ns[pos + 1:]:
                neighbors[a].add(b)
                neighbors[b].add(a)
        active.remove(variable)
        order.append(variable)
    return order, width


def _align_factor(
    values: torch.Tensor, scope: tuple[int, ...], union: tuple[int, ...],
) -> torch.Tensor:
    shape = [1] * len(union)
    for variable in scope:
        shape[union.index(variable)] = 2
    return values.reshape(shape)


def eliminate_log_partition(
    couplings: torch.Tensor,
    unary: torch.Tensor,
    order: list[int] | None = None,
) -> tuple[torch.Tensor, list[EliminationConditional], int]:
    """Return exact log partition and reverse-sampling conditionals for one graph."""
    n = int(unary.shape[0])
    adjacency = couplings.ne(0)
    if order is None:
        order, width = min_fill_order(adjacency)
    else:
        width = 0

    factors: list[tuple[tuple[int, ...], torch.Tensor]] = [
        ((v,), torch.stack((unary[v] * 0.0, unary[v]))) for v in range(n)
    ]
    spin_pair = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0]], dtype=unary.dtype, device=unary.device,
    )
    for i in range(n):
        for j in range(i + 1, n):
            if bool(adjacency[i, j]):
                factors.append(((i, j), couplings[i, j] * spin_pair))

    conditionals: list[EliminationConditional] = []
    for variable in order:
        selected = [factor for factor in factors if variable in factor[0]]
        factors = [factor for factor in factors if variable not in factor[0]]
        union = tuple(sorted({v for scope, _ in selected for v in scope}))
        joint = sum(_align_factor(values, scope, union) for scope, values in selected)
        axis = union.index(variable)
        reduced = torch.logsumexp(joint, dim=axis)
        remaining = tuple(v for v in union if v != variable)
        log_conditional = joint - reduced.unsqueeze(axis)
        conditionals.append(EliminationConditional(
            variable=variable,
            remaining=remaining,
            log_prob=log_conditional.movedim(axis, 0),
        ))
        factors.append((remaining, reduced))

    log_z = sum(values for _, values in factors)
    return log_z, conditionals, width


def sample_from_elimination(
    conditionals: list[EliminationConditional],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one exact joint sample by traversing elimination conditionals backward."""
    n = len(conditionals)
    sample = torch.zeros(n, dtype=torch.long, device=conditionals[0].log_prob.device)
    for conditional in reversed(conditionals):
        index = (slice(None), *(int(sample[v]) for v in conditional.remaining))
        probability_one = conditional.log_prob[index].exp()[1]
        draw = torch.rand((), device=sample.device, generator=generator)
        sample[conditional.variable] = (draw < probability_one).to(torch.long)
    return sample


def exact_truth_and_marginals(
    couplings: torch.Tensor,
    unary: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Exact samples and marginals for a batch, currently using a per-graph CPU loop."""
    truths = []
    marginals = []
    widths = []
    for graph_couplings, graph_unary in zip(couplings, unary):
        differentiable_unary = graph_unary.detach().requires_grad_(True)
        log_z, conditionals, width = eliminate_log_partition(
            graph_couplings, differentiable_unary,
        )
        (p_full,) = torch.autograd.grad(log_z, differentiable_unary)
        truths.append(sample_from_elimination(conditionals, generator=generator))
        marginals.append(p_full.detach())
        widths.append(width)
    return torch.stack(truths), torch.stack(marginals), widths
