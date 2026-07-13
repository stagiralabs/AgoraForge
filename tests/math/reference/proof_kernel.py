"""ProofKernel: stochastic proof completion model for the math environment.

Implements a memoryless (geometric) per-step proof-completion model:
    p_proof([Gamma |- phi]) = g([Gamma |- phi]) * (1 - (1 + s_mass)^(-rho))

Where:
    g([Gamma |- phi]) = 1[T(phi)=1]                  (feasibility gate)
    s_mass([Gamma |- phi]) = sum_{psi in Gamma} w(psi, phi)

This is the closed form of 1 - exp(-rho * log(1 + s_mass)): the exponential
and logarithm cancel, leaving a power law in the support mass. Note that with
no supporting lemmas (s_mass = 0) the success probability is exactly 0 -- a
proof needs at least some resolved support to make progress.

The per-step success probability is constant across attempts: it does not
depend on how long the agent has already been trying to prove phi (the model
is memoryless), and each call corresponds to a single unit time step.
"""

from __future__ import annotations

import numpy as np

from tests.math.reference.graph import FormulaGraph
from tests.math.reference.library import Library


class ProofKernel:
    """Proof completion kernel parameterized by MathConfig."""

    def __init__(self, rho: float):
        self.rho = float(rho)

    @classmethod
    def from_config(cls, cfg) -> ProofKernel:
        return cls(rho=cfg.rho)

    def _feasibility_gate(self, graph: FormulaGraph, library: Library, phi: int) -> bool:
        """g([Gamma |- phi]) = 1[T(phi)=1]."""
        return graph.is_true(phi)

    def _support_mass(self, graph: FormulaGraph, library: Library, phi: int) -> float:
        """s_mass = sum_{psi in Gamma} w(psi, phi)."""
        s_mass = 0.0
        for psi in library.resolved_formulas():
            w = graph.get_weight(psi, phi)
            if w > 0:
                s_mass += w
        return s_mass

    def success_probability(
        self,
        graph: FormulaGraph,
        library: Library,
        phi: int,
    ) -> float:
        """p_proof([Gamma |- phi]) = g * (1 - (1 + s_mass)^(-rho)).

        Constant per-step success probability; memoryless, so it does not
        depend on how long the agent has already been trying to prove phi.
        """
        if not self._feasibility_gate(graph, library, phi):
            return 0.0
        s_mass = self._support_mass(graph, library, phi)
        return 1.0 - (1.0 + s_mass) ** (-self.rho)

    def sample(
        self,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        rng: np.random.Generator,
    ) -> bool:
        """Sample a single-step proof outcome."""
        p = self.success_probability(graph, library, phi)
        return bool(rng.random() < p)
