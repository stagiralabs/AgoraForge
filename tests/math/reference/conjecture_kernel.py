"""ConjectureKernel: stochastic conjecture proposal model for the math environment.

Implements a memoryless (geometric) per-step conjecture model:
    p_conj^alpha(L, s) = 1 - (1 + m^alpha(L, s))^(-eta)

Where q is supplied by the agent's query model as directional related weights:
    m^alpha(L, s) = sum_{psi in G(L)} q(psi | L, s)

This is the same power-law form as the proof kernel. With no opportunity mass
(m = 0) the success probability is exactly 0.

The per-step success probability is constant across attempts (memoryless): it
does not depend on how long the agent has already been conjecturing, and each
call corresponds to a single unit time step.

Proposal distribution: when a step succeeds, a ghost is sampled in proportion to
its directional weight q (the same q that drives the opportunity mass, which is
exactly the normaliser). Ghosts with non-positive weight cannot be proposed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from tests.math.reference.library import Library


class ConjectureKernel:
    """Conjecture proposal kernel parameterized by MathConfig."""

    def __init__(self, eta: float):
        self.eta = float(eta)

    @classmethod
    def from_config(cls, cfg) -> ConjectureKernel:
        return cls(eta=cfg.eta)

    def _opportunity_mass(
        self,
        library: Library,
        weights: np.ndarray,
    ) -> float:
        """m^alpha(L, s) from caller-supplied directional weights."""
        ghosts = set(range(library.F_size)) - library.concrete
        mass = 0.0
        for psi in ghosts:
            q = float(weights[psi])
            if q > 0:
                mass += q
        return mass

    def success_probability(
        self,
        library: Library,
        weights: np.ndarray,
    ) -> float:
        """p_conj^alpha(L, s) = 1 - (1 + m^alpha)^(-eta).

        Constant per-step success probability; memoryless, so it does not
        depend on how long the agent has already been conjecturing.
        """
        m = self._opportunity_mass(library, weights)
        return 1.0 - (1.0 + m) ** (-self.eta)

    def proposal_probabilities(
        self,
        library: Library,
        weights: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        """Ghost ids and their proposal probabilities, proportional to weight q.

        Ghosts with non-positive weight get zero mass; if every ghost weight is
        non-positive the distribution falls back to uniform.
        """
        ghosts = sorted(set(range(library.F_size)) - library.concrete)
        if not ghosts:
            return [], np.zeros(0)
        w = np.array([max(0.0, float(weights[psi])) for psi in ghosts])
        total = w.sum()
        probs = w / total if total > 0 else np.full(len(ghosts), 1.0 / len(ghosts))
        return ghosts, probs

    def sample_proposal(
        self,
        library: Library,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[int]:
        """Sample one ghost formula in proportion to its directional weight."""
        ghosts, probs = self.proposal_probabilities(library, weights)
        if not ghosts:
            return None
        return int(ghosts[rng.choice(len(ghosts), p=probs)])

    def sample(
        self,
        library: Library,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[bool, Optional[int]]:
        """Sample one memoryless conjecture step from query-model related weights."""
        p = self.success_probability(library, weights)
        success = bool(rng.random() < p)
        if success:
            proposal = self.sample_proposal(library, weights, rng)
            return True, proposal
        return False, None
