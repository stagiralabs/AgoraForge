"""On-device, batched sampling of claim-graph instances (BA + signed Ising + Gibbs).

An instance is the judge's factor graph over N binary claims:

1. **Barabási–Albert structure** — sequential node addition, each new claim attaching
   to ``m``-ish existing claims drawn ∝ current degree.
2. **Signed couplings** — each BA edge gets an Ising coupling ``J_uv`` with magnitude
   ``U[coupling_min, coupling_max]`` and sign negative (attack) with probability
   ``p_attack``; each claim gets a unary bias ``b_v ~ U[-cutoff, +cutoff]``.
   Log-potentials: ``b_v x_v`` per claim and ``J_uv s_u s_v`` per edge (``s = 2x - 1``).
3. **Gibbs** — ``truth_gibbs_sweeps`` burn-in sweeps, then ``marginal_gibbs_sweeps``
   sweeps whose conditional probabilities are averaged into a Rao-Blackwellized
   full-graph marginal estimate ``p_full``; the final state is the truth sample.

All batched over the env dimension ``B`` with no per-env Python loop.
"""

from __future__ import annotations

import torch

from agoraforge.envs.debate.config import DebateConfig
from agoraforge.envs.graph_sampling import adjacency_from_parents, barabasi_albert_parents


def sample_claim_graphs(
    cfg: DebateConfig, B: int, device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw ``(couplings[B,N,N], unary[B,N], truth[B,N], p_full[B,N])`` on device.

    ``couplings`` is symmetric with zero diagonal; ``truth`` is a 0/1 sample from the
    joint; ``p_full`` is the Rao-Blackwellized marginal estimate ``P(x_v = 1 | graph)``.
    """
    couplings, unary = sample_claim_graph_parameters(cfg, B, device, gen)

    truth, p_full = _gibbs_truth_and_marginals(
        couplings, unary, cfg.truth_gibbs_sweeps, cfg.marginal_gibbs_sweeps, gen
    )
    return couplings, unary, truth, p_full


def sample_claim_graph_parameters(
    cfg: DebateConfig, B: int, device, gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw graph structure and potentials without running target inference."""
    N = int(cfg.num_claims)
    parents, pcount = barabasi_albert_parents(B, N, cfg.claim_graph_m, device, gen)
    adj = adjacency_from_parents(parents, pcount, N)

    magnitude = cfg.coupling_min + (cfg.coupling_max - cfg.coupling_min) * torch.rand(
        (B, N, N), device=device, generator=gen
    )
    sign = torch.where(
        torch.rand((B, N, N), device=device, generator=gen) < cfg.p_attack, -1.0, 1.0
    )
    upper = torch.triu(magnitude * sign, diagonal=1)
    couplings = (upper + upper.transpose(1, 2)) * adj.to(torch.float32)

    unary = (torch.rand((B, N), device=device, generator=gen) * 2.0 - 1.0) * cfg.unary_cutoff

    return couplings, unary


def _gibbs_truth_and_marginals(
    couplings: torch.Tensor,   # (B, N, N)
    unary: torch.Tensor,       # (B, N)
    burn_in: int,
    marginal_sweeps: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential-scan Gibbs over the Ising claims, batched over envs.

    Conditional: ``logit P(x_v=1 | rest) = b_v + 2 * sum_u J_uv s_u``. Updates within a
    sweep are sequential per claim (each env carries its own neighbour states), so the
    joint is sampled correctly; a fixed visit order relies on enough sweeps
    (distributional, not per-seed, equivalence — same convention as the math sampler).
    """
    B, N = unary.shape
    device = unary.device
    x = (torch.rand((B, N), device=device, generator=gen) < 0.5).to(torch.float32)
    p_sum = torch.zeros((B, N), dtype=torch.float32, device=device)

    for sweep in range(burn_in + marginal_sweeps):
        record = sweep >= burn_in
        for v in range(N):
            s = 2.0 * x - 1.0
            logit = unary[:, v] + 2.0 * (couplings[:, v, :] * s).sum(dim=-1)
            p1 = torch.sigmoid(logit)
            r = torch.rand((B,), device=device, generator=gen)
            x[:, v] = (r < p1).to(torch.float32)
            if record:
                p_sum[:, v] += p1
    p_full = (p_sum / float(marginal_sweeps)).clamp(1e-4, 1.0 - 1e-4)
    return x.to(torch.long), p_full
