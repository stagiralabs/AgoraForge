"""FormulaFactorGraph: the *distribution* over formula graphs (generation stage 1).

A ``FormulaFactorGraph`` is a written-down distribution; sampling from it (stage 2)
produces a concrete :class:`~tests.math.reference.graph.FormulaGraph`.

Variables
---------
* **Truth** ``T_t in {0, 1}`` for each theorem ``t``. Formula ``phi`` with theorem
  ``t`` and sign ``s`` is *true* iff ``s == T_t``.
* **Weight** ``W_{A->B} in levels`` for each ordered pair of formulas ``(A, B)`` whose
  theorems are linked (by the Barabási–Albert structure) and distinct. There are
  exactly 8 per undirected theorem edge ``{a, b}``: 2 (source sign) x 2 (target sign)
  x 2 (direction).

Factors
-------
*Pairwise* — one per weight variable, over ``(W_{A->B}, T_{t_A}, T_{t_B})``::

    log psi(w, T_{t_A}, T_{t_B}) = -penalty * w   if (A true and B false) else 0

i.e. a high A->B weight when A is true and B is false is penalized (a high weight acts
like the implication ``A => B``, which a true-A/false-B world violates).

*Unary* — optionally one per variable (see :mod:`tests.math.reference.unary_factors`):
``log phi_t(T_t)`` on each truth and ``log phi_{A->B}(w)`` on each weight, supplied as
generic log-potential arrays. With no unary factors the truth prior is uniform and the
weight prior is uniform over ``levels`` (the pairwise-only model).

Joint::

    P(T, W) ∝ prod_t phi_t(T_t)
              * prod_{A->B} phi_{A->B}(W_{A->B}) psi(W_{A->B}, T_{t_A}, T_{t_B})
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np


def weight_variable_ids(
    theorem_edges: Iterable[Tuple[int, int]], num_theorems: int
) -> Iterator[Tuple[int, int]]:
    """Yield the ``(src_phi, dst_phi)`` id of every weight variable in the graph.

    8 per undirected theorem edge ``{a, b}``: both directions × both source signs × both
    target signs. The yielded ids are the canonical formula ids (``t + sign * n``), so the
    ordering of each input edge pair does not matter. This is the single source of truth
    for *which* weight variables exist — both factor construction and sampling iterate it.
    """
    n = int(num_theorems)
    for a, b in theorem_edges:
        for t_src, t_dst in ((a, b), (b, a)):
            for s_src in (0, 1):
                for s_dst in (0, 1):
                    yield (t_src + s_src * n, t_dst + s_dst * n)


def _logsumexp(arr: np.ndarray) -> float:
    """Numerically stable ``log sum exp`` over a 1-D array."""
    m = float(np.max(arr))
    return m + float(np.log(np.sum(np.exp(arr - m))))


class FormulaFactorGraph:
    def __init__(
        self,
        num_theorems: int,
        theorem_edges: Iterable[Tuple[int, int]],
        levels: np.ndarray,
        penalty: float,
        truth_unary: Optional[np.ndarray] = None,
        weight_unary: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    ):
        self.num_theorems = int(num_theorems)
        # Undirected theorem edges (a < b); the *structure* of the distribution.
        self.theorem_edges = sorted(
            {(int(min(a, b)), int(max(a, b))) for a, b in theorem_edges}
        )
        self.levels = np.asarray(levels, dtype=np.float64)
        assert self.levels.ndim == 1 and self.levels.size >= 1, "levels must be non-empty 1-D"
        assert np.all((self.levels >= 0.0) & (self.levels <= 1.0)), "levels must lie in [0, 1]"
        self.penalty = float(penalty)
        assert self.penalty >= 0.0, "penalty must be >= 0"
        self._zero_level = np.zeros(self.levels.size, dtype=np.float64)

        # ── Unary factors (default: none ⇒ uniform truth / weight priors) ──
        # Truth unary: per-theorem log-potential over {0, 1}.
        if truth_unary is None:
            self.truth_unary = np.zeros((self.num_theorems, 2), dtype=np.float64)
        else:
            self.truth_unary = np.asarray(truth_unary, dtype=np.float64)
            assert self.truth_unary.shape == (self.num_theorems, 2), \
                f"truth_unary must have shape ({self.num_theorems}, 2), got {self.truth_unary.shape}"
        # Weight unary: per-weight-variable log-potential over `levels`. Stored sparsely;
        # absent keys contribute the zero potential.
        self.weight_unary: Dict[Tuple[int, int], np.ndarray] = {}
        if weight_unary is not None:
            for (src, dst), pot in weight_unary.items():
                pot = np.asarray(pot, dtype=np.float64)
                assert pot.shape == (self.levels.size,), \
                    f"weight_unary[{(src, dst)}] must have shape ({self.levels.size},), got {pot.shape}"
                self.weight_unary[(int(src), int(dst))] = pot

        # Conditional weight-level distributions P(w | config) when there is NO unary
        # factor on the weight, precomputed once (the common fast path in sample_weights):
        #   bad config  -> P(w) ∝ exp(-penalty * w)
        #   good config -> uniform
        bad_unnorm = np.exp(-self.penalty * self.levels)
        self._p_weight_bad = bad_unnorm / bad_unnorm.sum()
        self._p_weight_good = np.full(self.levels.size, 1.0 / self.levels.size)

        # Pairwise truth potential log M(T_a, T_b) per theorem edge, obtained by
        # marginalizing that edge's 8 weight factors (pairwise penalty × any weight unary).
        # Per-edge because weight unary factors differ across weight variables.
        self._edge_logpot: Dict[Tuple[int, int], np.ndarray] = {
            (a, b): self._build_edge_logpot(a, b) for (a, b) in self.theorem_edges
        }

        # Adjacency over theorems, for Gibbs over the truth MRF.
        self._neighbors: Dict[int, list] = {t: [] for t in range(self.num_theorems)}
        for a, b in self.theorem_edges:
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)

    # ── stage-1 introspection: the marginalized truth potential ──

    def _build_edge_logpot(self, a: int, b: int) -> np.ndarray:
        """log M(T_a, T_b): marginalize the 8 weight factors on theorem edge {a, b}.

        For each weight variable W on this edge and each truth assignment (T_a, T_b), the
        weight contributes ``log sum_w exp(log psi(w) + log phi_unary(w))`` where the
        pairwise ``log psi`` is ``-penalty * w`` for the "source true ∧ target false"
        configuration and 0 otherwise. Summing these over the edge's 8 weight variables
        gives the pairwise truth log-potential.

        With no unary factors this reduces to the constant potential of the pairwise-only
        model (each truth assignment yields the same bad/good factor count per edge), so
        the truth marginal is uniform; unary weight factors break that symmetry generically
        and the Gibbs sampler picks the asymmetry up without special-casing.
        """
        n = self.num_theorems
        logpot = np.zeros((2, 2), dtype=np.float64)
        for t_a in (0, 1):
            for t_b in (0, 1):
                truth = {a: t_a, b: t_b}
                total = 0.0
                for src_phi, dst_phi in weight_variable_ids([(a, b)], n):
                    s_src, s_dst = src_phi // n, dst_phi // n
                    t_src, t_dst = src_phi % n, dst_phi % n
                    src_true = (s_src == truth[t_src])
                    dst_false = (s_dst != truth[t_dst])
                    bad = src_true and dst_false
                    base = (-self.penalty * self.levels) if bad else self._zero_level
                    unary = self.weight_unary.get((src_phi, dst_phi), self._zero_level)
                    total += _logsumexp(base + unary)
                logpot[t_a, t_b] = total
        return logpot

    # ── stage-2: drawing a sample ──
    #
    # ⚠ The two-step decomposition below (sample truths with weights marginalized out,
    # then sample weights conditionally) is VALID ONLY because each weight variable
    # appears in exactly one pairwise factor plus its own unary factor — both functions of
    # that weight alone given the truths — so weights stay conditionally independent given
    # the truths. If the factor graph couples weights to each other, this breaks and a
    # general sampler (full Gibbs over all variables, or belief propagation) is required.

    def sample(
        self, rng: np.random.Generator, sweeps: int
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        truths = self.sample_truths(rng, sweeps)
        weights = self.sample_weights(truths, rng)
        return truths, weights

    def sample_truths(self, rng: np.random.Generator, sweeps: int) -> np.ndarray:
        """Gibbs-sample theorem truths from the weight-marginalized truth MRF.

        Each theorem's conditional combines its unary truth potential with the marginalized
        pairwise potential of every incident edge. With no unary factors the conditionals
        are uniform (independent fair coins); the generic loop handles either case.
        """
        n = self.num_theorems
        truths = rng.integers(0, 2, size=n).astype(np.int32)
        for _ in range(max(1, int(sweeps))):
            for t in rng.permutation(n):
                # log P(T_t = v | rest) up to a constant: unary + incident-edge potentials.
                log_p = self.truth_unary[t].astype(np.float64).copy()  # over v in {0, 1}
                for u in self._neighbors[t]:
                    if t < u:
                        log_p += self._edge_logpot[(t, u)][:, truths[u]]
                    else:
                        log_p += self._edge_logpot[(u, t)][truths[u], :]
                log_p -= log_p.max()
                p = np.exp(log_p)
                p /= p.sum()
                truths[t] = int(rng.random() < p[1])
        return truths

    def sample_weights(
        self, truths: np.ndarray, rng: np.random.Generator
    ) -> Dict[Tuple[int, int], float]:
        """Sample every weight variable independently given the truth assignment."""
        n = self.num_theorems
        weights: Dict[Tuple[int, int], float] = {}
        for src_phi, dst_phi in weight_variable_ids(self.theorem_edges, n):
            s_src, s_dst = src_phi // n, dst_phi // n
            t_src, t_dst = src_phi % n, dst_phi % n
            src_true = (s_src == int(truths[t_src]))
            dst_false = (s_dst != int(truths[t_dst]))
            bad = src_true and dst_false
            unary = self.weight_unary.get((src_phi, dst_phi))
            if unary is None:
                # Fast path: no unary factor ⇒ use the precomputed conditional.
                p = self._p_weight_bad if bad else self._p_weight_good
            else:
                base = (-self.penalty * self.levels) if bad else self._zero_level
                logits = base + unary
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
            idx = int(rng.choice(self.levels.size, p=p))
            weights[(src_phi, dst_phi)] = float(self.levels[idx])
        return weights
