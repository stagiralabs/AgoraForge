"""On-device, batched generation of formula-graph latents (BA + unary + Gibbs MRF).

This is the GPU twin of the CPU reference factor-graph sampler. It
draws the same distribution as the reference numpy sampler, but for all ``B`` envs at
once with fixed-shape tensor kernels and no per-env Python loop or host round-trip.

Three stages, all batched over the env dimension ``B``:

1. **Barabási–Albert structure** — a sequential node-addition loop (``N`` tiny kernels),
   each new node attaching to ``m``-ish existing nodes drawn ∝ current degree. The graph
   is stored as a fixed-shape per-node parent tensor ``(B, N, 2)`` plus a count, the
   ragged-free analogue of the reference edge *set*.
2. **Unary factors** — per-theorem truth bias and per-weight-variable weight bias, both
   ``Uniform[-cutoff, +cutoff]``, as plain tensor draws.
3. **Gibbs truth MRF** — the per-edge weight-marginalized 2×2 truth potential, then
   ``sweeps`` Gibbs sweeps over the truth variables, sequential within a sweep (matching
   the reference update order) but batched across all ``B`` envs.

The result is ``(truth_sign[B, N], weights[B, F, F])`` ready for the rollout, identical
in distribution to stacking ``B`` independent reference draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agoraforge.envs.graph_sampling import adjacency_from_parents, barabasi_albert_parents


@dataclass
class LatentConfig:
    num_theorems: int
    theorem_graph_m: float
    n_weight_levels: int
    weight_min: float
    weight_max: float
    weight_penalty: float
    truth_gibbs_sweeps: int
    truth_unary_cutoff: float
    weight_unary_cutoff: float
    age_directional_weight_bias: float


def latent_config_from_config(cfg) -> LatentConfig:
    return LatentConfig(
        num_theorems=int(cfg.num_theorems),
        theorem_graph_m=float(cfg.theorem_graph_m),
        n_weight_levels=int(cfg.n_weight_levels),
        weight_min=float(cfg.weight_min),
        weight_max=float(cfg.weight_max),
        weight_penalty=float(cfg.weight_penalty),
        truth_gibbs_sweeps=int(cfg.truth_gibbs_sweeps),
        truth_unary_cutoff=float(cfg.truth_unary_cutoff),
        weight_unary_cutoff=float(cfg.weight_unary_cutoff),
        age_directional_weight_bias=float(cfg.age_directional_weight_bias),
    )


def _edge_truth_logpot(
    weight_bias_edge: torch.Tensor, levels: torch.Tensor, penalty: float, n: int
) -> torch.Tensor:
    """Marginalized 2×2 truth log-potential ``log M(T_a, T_b)`` per (batched) edge.

    Mirrors ``FormulaFactorGraph._build_edge_logpot``: for each of the edge's 8 weight
    variables and each truth assignment ``(t_a, t_b)``, contribute
    ``logsumexp_w(-penalty*w*[bad] + bias_W * w)``, where ``bad`` = src-true ∧ dst-false.

    ``weight_bias_edge`` has shape ``(..., 8)`` giving the per-weight-variable bias in the
    canonical ordering of ``weight_variable_ids`` for a single edge ``(a, b)`` with
    ``a < b``. Returns ``(..., 2, 2)``.
    """
    device = weight_bias_edge.device
    L = levels.numel()
    levels = levels.to(torch.float64)
    # Enumerate the 8 weight variables of edge (a,b) in weight_variable_ids order:
    #   for (t_src, t_dst) in [(a,b),(b,a)]: for s_src in (0,1): for s_dst in (0,1)
    # encode whether src endpoint is 'a' (dir 0) and the two signs.
    specs = []  # (dir, s_src, s_dst): dir=0 -> src=a,dst=b ; dir=1 -> src=b,dst=a
    for d in (0, 1):
        for s_src in (0, 1):
            for s_dst in (0, 1):
                specs.append((d, s_src, s_dst))

    lead = weight_bias_edge.shape[:-1]
    logpot = torch.zeros((*lead, 2, 2), dtype=torch.float64, device=device)
    for t_a in (0, 1):
        for t_b in (0, 1):
            total = torch.zeros(lead, dtype=torch.float64, device=device)
            for w_idx, (d, s_src, s_dst) in enumerate(specs):
                t_src_truth = t_a if d == 0 else t_b
                t_dst_truth = t_b if d == 0 else t_a
                src_true = (s_src == t_src_truth)
                dst_false = (s_dst != t_dst_truth)
                bad = src_true and dst_false
                base = (-penalty * levels) if bad else torch.zeros(L, dtype=torch.float64, device=device)
                bias = weight_bias_edge[..., w_idx]  # (lead,)
                # logits over levels: base + bias * levels  -> (lead, L)
                logits = base.view(*([1] * len(lead)), L) + bias.unsqueeze(-1) * levels.view(
                    *([1] * len(lead)), L
                )
                total = total + torch.logsumexp(logits, dim=-1)
            logpot[..., t_a, t_b] = total
    return logpot


def sample_latent_batched(
    cfg: LatentConfig,
    B: int,
    device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``(truth_sign[B, N], weights[B, F, F])`` fully on device, batched over envs.

    Distributional twin of stacking ``B`` independent CPU reference draws.
    """
    N = int(cfg.num_theorems)
    F = 2 * N
    m = float(cfg.theorem_graph_m)
    penalty = float(cfg.weight_penalty)
    L = int(cfg.n_weight_levels)
    levels = torch.linspace(
        float(cfg.weight_min), float(cfg.weight_max), L, dtype=torch.float64, device=device
    )
    truth_cut = float(cfg.truth_unary_cutoff)
    weight_cut = float(cfg.weight_unary_cutoff)
    age_bias = float(cfg.age_directional_weight_bias)
    sweeps = max(1, int(cfg.truth_gibbs_sweeps))

    # ── Stage 1: BA structure ──────────────────────────────────────────────
    parents, pcount = barabasi_albert_parents(
        B, N, m, device, gen, degree_dtype=torch.float64
    )
    adj = adjacency_from_parents(parents, pcount, N)

    # ── Stage 2: unary factors ─────────────────────────────────────────────
    # Truth bias_t ~ U[-c, c]; truth_unary row t = [0, bias_t].
    if truth_cut > 0.0:
        truth_bias = (torch.rand((B, N), device=device, generator=gen, dtype=torch.float64) * 2 - 1) * truth_cut
    else:
        truth_bias = torch.zeros((B, N), dtype=torch.float64, device=device)

    # Per-edge weight bias: 8 weight variables per undirected edge. We need a per-ordered
    # edge-pair (a<b) bias tensor of shape (B, N, N, 8); only edges present are used.
    if weight_cut > 0.0:
        weight_bias = (
            torch.rand((B, N, N, 8), device=device, generator=gen, dtype=torch.float64) * 2 - 1
        ) * weight_cut
    else:
        weight_bias = torch.zeros((B, N, N, 8), dtype=torch.float64, device=device)
    if age_bias > 0.0:
        # For canonical theorem edge (a<b), the first four specs are a->b and the last
        # four are b->a. The BA generator creates lower ids earlier, so this encourages
        # earlier->later formula weights and discourages later->earlier formula weights.
        direction = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            dtype=torch.float64,
            device=device,
        )
        weight_bias = weight_bias + age_bias * direction.view(1, 1, 1, 8)

    # ── Edge truth potentials log M(T_a, T_b): (B, N, N, 2, 2) for a<b edges ──
    edge_logpot = _edge_truth_logpot(weight_bias, levels, penalty, N)  # (B,N,N,2,2)

    # ── Stage 3: Gibbs over truths ─────────────────────────────────────────
    truth_sign = _gibbs_truths(
        adj, edge_logpot, truth_bias, sweeps, B, N, device, gen
    )

    # ── Stage 4: conditional weights → dense (B, F, F) ─────────────────────
    weights = _sample_weights(
        adj, weight_bias, truth_sign, levels, penalty, N, F, device, gen
    )
    return truth_sign, weights


def _gibbs_truths(
    adj: torch.Tensor,
    edge_logpot: torch.Tensor,
    truth_bias: torch.Tensor,
    sweeps: int,
    B: int,
    N: int,
    device,
    gen: torch.Generator,
) -> torch.Tensor:
    """Batched Gibbs over the weight-marginalized truth MRF.

    For each theorem ``t``, ``log P(T_t=v | rest) = truth_unary[t,v] + sum_{u~t} M_edge``,
    where the edge potential is indexed by the neighbour's current truth. Updates within a
    sweep are sequential per theorem (matching the reference), but every env updates the
    same theorem in lockstep — the per-env coupling is preserved because each env carries
    its own neighbour truths. The reference visits theorems in a random per-sweep
    permutation; permutation order does not change the stationary distribution, so we use a
    fixed order and rely on enough sweeps (distributional, not per-seed, equivalence).
    """
    # Symmetric edge potential indexed (t, u): for t<u use edge_logpot[t,u][v, T_u];
    # for t>u use edge_logpot[u,t][T_u, v]. Precompute a (B,N,N,2,2) tensor giving, for the
    # ORDERED pair (center=t, neighbour=u), pot_oriented[t,u][v, T_u].
    pot = edge_logpot  # (B,N,N,2,2), defined for a<b
    pot_T = edge_logpot.transpose(1, 2).transpose(-1, -2)  # (B,N,N,2,2): [u,t] -> swap roles
    # For center=t neighbour=u with t<u: oriented = pot[t,u]  (v=dim -2, T_u=dim -1)  ✓
    # For center=t neighbour=u with t>u: edge stored at (u,t) as [T_u, v]; we want [v,T_u]
    #   so transpose last two dims of pot[u,t]  == pot_T[t,u]. Combine by triangular mask.
    tri_upper = torch.arange(N, device=device).view(N, 1) < torch.arange(N, device=device).view(1, N)
    oriented = torch.where(
        tri_upper.view(1, N, N, 1, 1), pot, pot_T
    )  # (B,N,N,2,2): center t, neighbour u -> [v, T_u]

    truths = (torch.rand((B, N), device=device, generator=gen) < 0.5).to(torch.long)
    truth_unary = torch.stack([torch.zeros_like(truth_bias), truth_bias], dim=-1)  # (B,N,2)
    adj_f = adj.to(torch.float64)  # (B,N,N)

    for _ in range(sweeps):
        for t in range(N):
            # neighbours' current truths -> index oriented[:, t, u, :, T_u]
            Tu = truths  # (B, N)
            # oriented[:, t] : (B, N_u, 2, 2). Gather over T_u along last dim.
            ori_t = oriented[:, t]  # (B, N, 2, 2): neighbour u -> [v, T_u]
            # pick T_u: -> (B, N, 2) over v
            pot_vu = ori_t.gather(
                -1, Tu.view(B, N, 1, 1).expand(B, N, 2, 1)
            ).squeeze(-1)  # (B, N, 2)
            # mask to actual neighbours and sum
            contrib = (pot_vu * adj_f[:, t].unsqueeze(-1)).sum(dim=1)  # (B, 2)
            log_p = truth_unary[:, t] + contrib  # (B, 2)
            log_p = log_p - log_p.max(dim=-1, keepdim=True).values
            p = torch.exp(log_p)
            p1 = p[:, 1] / p.sum(dim=-1)
            r = torch.rand((B,), device=device, generator=gen)
            truths[:, t] = (r < p1).to(torch.long)
    return truths


def _sample_weights(
    adj: torch.Tensor,
    weight_bias: torch.Tensor,
    truth_sign: torch.Tensor,
    levels: torch.Tensor,
    penalty: float,
    N: int,
    F: int,
    device,
    gen: torch.Generator,
) -> torch.Tensor:
    """Sample every weight variable given truths and scatter into a dense ``(B, F, F)``.

    Mirrors ``FormulaFactorGraph.sample_weights``: for each present undirected edge (a<b),
    its 8 weight variables ``(t_src+s_src*N -> t_dst+s_dst*N)`` get
    ``p(w) ∝ exp(-penalty*w*[bad] + bias_W*w)`` with ``bad`` = src-true ∧ dst-false.
    """
    B = adj.shape[0]
    L = levels.numel()
    weights = torch.zeros((B, F, F), dtype=torch.float32, device=device)
    levels64 = levels.to(torch.float64)

    # specs in weight_variable_ids order (dir, s_src, s_dst)
    specs = []
    for d in (0, 1):
        for s_src in (0, 1):
            for s_dst in (0, 1):
                specs.append((d, s_src, s_dst))

    # present edges (a<b): boolean upper-triangular adjacency
    tri = torch.triu(torch.ones((N, N), dtype=torch.bool, device=device), diagonal=1)
    edge_mask = adj & tri.view(1, N, N)  # (B, N, N)
    bidx, aidx, bnode = edge_mask.nonzero(as_tuple=True)  # each (E,)
    if bidx.numel() == 0:
        return weights
    a = aidx
    b = bnode
    Ta = truth_sign[bidx, a]  # (E,)
    Tb = truth_sign[bidx, b]

    for w_idx, (d, s_src, s_dst) in enumerate(specs):
        # endpoints by direction
        if d == 0:
            t_src, t_dst, T_src, T_dst = a, b, Ta, Tb
        else:
            t_src, t_dst, T_src, T_dst = b, a, Tb, Ta
        bad = (s_src == T_src) & (s_dst != T_dst)  # (E,) bool
        bias = weight_bias[bidx, a, b, w_idx]  # (E,) float64
        base = torch.where(
            bad.unsqueeze(-1),
            (-penalty * levels64).view(1, L),
            torch.zeros((1, L), dtype=torch.float64, device=device),
        )  # (E, L)
        logits = base + bias.unsqueeze(-1) * levels64.view(1, L)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.exp(logits)
        p = p / p.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(p.to(torch.float32), 1, generator=gen).squeeze(-1)  # (E,)
        w_val = levels[idx].to(torch.float32)  # (E,)
        src_phi = t_src + s_src * N
        dst_phi = t_dst + s_dst * N
        weights[bidx, src_phi, dst_phi] = w_val
    return weights
