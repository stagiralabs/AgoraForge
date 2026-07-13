"""Tensor instance generator matches the CPU reference distributionally.

These run on the CPU device — the kernels are device-agnostic, so CPU exercises the same
math as CUDA. Bars are set above the Monte-Carlo noise floor of the reference itself.
"""
import numpy as np
import torch

import agoraforge.envs.math.latent as G
from agoraforge.envs.math.latent import LatentConfig
from tests.math.reference.barabasi_albert import barabasi_albert_edges
from tests.math.reference.factor_graph import weight_variable_ids
from tests.math.reference.sampling import build_factor_graph
from tests.math.reference.unary_factors import sample_unary_factors


def _cfg(N):
    return LatentConfig(
        num_theorems=N, theorem_graph_m=1.5, n_weight_levels=9, weight_min=0.0,
        weight_max=1.0, weight_penalty=4.0, truth_gibbs_sweeps=5,
        truth_unary_cutoff=2.0, weight_unary_cutoff=2.0,
        age_directional_weight_bias=0.0,
    )


def _adj_from_parents(parents, pcount, N):
    B = parents.shape[0]
    adj = torch.zeros((B, N, N), dtype=torch.bool)
    for b in range(B):
        for s in range(N):
            for j in range(int(pcount[b, s])):
                p = int(parents[b, s, j])
                adj[b, s, p] = adj[b, p, s] = True
    return adj


def test_ba_graph_stats_match_oracle():
    N, B = 64, 256
    # oracle adjacency
    oadj = np.zeros((B, N, N), dtype=bool)
    for b in range(B):
        rng = np.random.default_rng(1000 + b)
        for a, c in barabasi_albert_edges(N, 1.5, rng):
            oadj[b, a, c] = oadj[b, c, a] = True
    oadj = torch.from_numpy(oadj)
    gen = torch.Generator().manual_seed(7)
    parents, pcount = G.barabasi_albert_parents(B, N, 1.5, torch.device("cpu"), gen)
    gadj = _adj_from_parents(parents, pcount, N)

    def stats(adj):
        deg = adj.sum(-1).double()
        edges = adj.triu(1).sum(dim=(1, 2)).double().mean()
        A = adj.double()
        tri = torch.einsum("bij,bjk,bki->bi", A, A, A)
        poss = deg * (deg - 1)
        cc = torch.where(poss > 0, tri / poss, torch.zeros_like(tri))
        m = poss > 0
        return edges.item(), deg.mean().item(), ((cc * m).sum(1) / m.sum(1).clamp_min(1)).mean().item()

    oe, od, occ = stats(oadj)
    ge, gd, gcc = stats(gadj)
    assert abs(oe - ge) < 2.0, (oe, ge)            # edge count within noise
    assert abs(od - gd) < 0.05, (od, gd)           # mean degree
    assert abs(occ - gcc) < 0.02, (occ, gcc)       # clustering coefficient


def test_age_directional_weight_bias_tilts_by_theorem_order():
    cfg = _cfg(4)
    cfg.weight_unary_cutoff = 0.0
    cfg.age_directional_weight_bias = 3.0
    levels = np.linspace(0.0, 1.0, cfg.n_weight_levels)
    _, weight_unary = sample_unary_factors(
        cfg, {(0, 3)}, levels, rng=np.random.default_rng(123)
    )

    for src, dst in weight_variable_ids({(0, 3)}, cfg.num_theorems):
        t_src, t_dst = src % cfg.num_theorems, dst % cfg.num_theorems
        expected = 3.0 if t_src < t_dst else -3.0
        np.testing.assert_allclose(weight_unary[(src, dst)], expected * levels)


def test_gibbs_truth_marginals_match_oracle():
    """Gibbs port on a SHARED MRF: marginals + pairwise correlations near the noise floor."""
    N = 12
    cfg = _cfg(N)
    fg = build_factor_graph(cfg, rng=np.random.default_rng(42))
    edges = fg.theorem_edges

    M = 6000
    or_truths = np.zeros((M, N), dtype=np.int64)
    rng2 = np.random.default_rng(123)
    for i in range(M):
        or_truths[i] = fg.sample_truths(rng2, sweeps=5)
    or_marg, or_corr = or_truths.mean(0), np.corrcoef(or_truths.T)

    levels = torch.linspace(0, 1, 9, dtype=torch.float64)
    adj = torch.zeros((1, N, N), dtype=torch.bool)
    for a, b in edges:
        adj[0, a, b] = adj[0, b, a] = True
    tb = torch.tensor(fg.truth_unary[:, 1], dtype=torch.float64).view(1, N)
    wb = torch.zeros((1, N, N, 8), dtype=torch.float64)
    specs = [(d, ss, sd) for d in (0, 1) for ss in (0, 1) for sd in (0, 1)]
    for (a, b) in edges:
        for wi, (d, s_src, s_dst) in enumerate(specs):
            t_src, t_dst = (a, b) if d == 0 else (b, a)
            pot = fg.weight_unary.get((t_src + s_src * N, t_dst + s_dst * N))
            if pot is not None:
                wb[0, a, b, wi] = pot[-1] / float(levels[-1])
    elp = G._edge_truth_logpot(wb, levels, 4.0, N)

    # marginalization is exact math, not Monte Carlo
    for (a, b) in edges:
        assert np.abs(fg._edge_logpot[(a, b)] - elp[0, a, b].numpy()).max() < 1e-10

    gen = torch.Generator().manual_seed(999)
    gt = G._gibbs_truths(
        adj.expand(M, N, N).contiguous(), elp.expand(M, N, N, 2, 2).contiguous(),
        tb.expand(M, N).contiguous(), 5, M, N, torch.device("cpu"), gen,
    ).numpy()
    g_marg, g_corr = gt.mean(0), np.corrcoef(gt.T)
    off = ~np.eye(N, dtype=bool)
    # noise floor at M=6000 is ~0.5/sqrt(M)=0.006 for marginals; allow generous slack
    assert np.abs(or_marg - g_marg).max() < 0.03, np.abs(or_marg - g_marg).max()
    assert np.abs(or_corr - g_corr)[off].max() < 0.08, np.abs(or_corr - g_corr)[off].max()


def test_determinism():
    cfg = _cfg(32)
    outs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(2024)
        outs.append(G.sample_latent_batched(cfg, 16, torch.device("cpu"), gen))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])
