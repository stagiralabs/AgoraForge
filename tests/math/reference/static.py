"""Fixed noisy-graph query model for proof success prediction.

Each agent receives a private noisy copy of the latent proof structure at the
start of a rollout. The copy contains formula truth beliefs, utility weights,
and the agent's proof-rate parameters. It is frozen for the entire rollout;
proof attempts and publications update only the model's known concrete/resolved
state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tests.math.reference.library import Library


class StaticQueryModel:
    """Frozen CPU oracle for probability-and-time queries."""

    def __init__(
        self,
        *,
        F_size: int,
        horizon_H: int,
        truth_prob: np.ndarray,
        weights: np.ndarray,
        rho: float,
    ):
        self.F_size = int(F_size)
        self.horizon_H = int(horizon_H)
        self.truth_prob = truth_prob.astype(np.float64, copy=True)
        self.weights = weights.astype(np.float64, copy=True)
        self.rho = float(rho)
        self.concrete_formulas: set[int] = set()
        self.resolved_formulas: set[int] = set()

    def neg(self, phi: int) -> int:
        half_F = self.F_size // 2
        return phi + half_F if phi < half_F else phi - half_F

    def update(self, new_library: Library) -> None:
        """Synchronize concrete and resolved state from the current library."""
        concrete_formulas: set[int] = set()
        for phi in new_library.concrete:
            self._validate_formula(phi)
            phi = int(phi)
            concrete_formulas.add(phi)
            concrete_formulas.add(self.neg(phi))

        resolved_formulas: set[int] = set()
        for phi in new_library.resolved:
            self._validate_formula(phi)
            phi = int(phi)
            resolved_formulas.add(phi)
            concrete_formulas.add(phi)
            concrete_formulas.add(self.neg(phi))

        self.concrete_formulas = concrete_formulas
        self.resolved_formulas = resolved_formulas

    def prob_and_time(
        self,
        concrete_target_formula: int,
    ) -> Tuple[float, float]:
        self._validate_formula(concrete_target_formula)
        concrete_target_formula = int(concrete_target_formula)
        resolved = set(self.resolved_formulas)
        probability = self._truth_prob_with_hypotheticals(
            concrete_target_formula,
            resolved,
            None,
        )
        if concrete_target_formula in resolved:
            return 1.0, 0.0

        expected_time = self._expected_time_from_state(
            concrete_target_formula,
            resolved,
            hypothetical_true={concrete_target_formula},
        )
        return float(probability), float(expected_time)

    def _validate_formula(self, phi: int) -> None:
        if not 0 <= int(phi) < self.F_size:
            raise ValueError(f"formula id {phi} outside [0, {self.F_size})")

    def _truth_prob_with_hypotheticals(
        self,
        phi: int,
        resolved: set[int],
        hypothetical_true: Optional[set[int]],
    ) -> float:
        if phi in resolved:
            return 1.0
        if self.neg(phi) in resolved:
            return 0.0
        if hypothetical_true:
            if phi in hypothetical_true:
                return 1.0
            if self.neg(phi) in hypothetical_true:
                return 0.0
        return float(self.truth_prob[phi])

    def _pi_from_state(
        self,
        phi: int,
        resolved: set[int],
        hypothetical_true: Optional[set[int]] = None,
    ) -> float:
        truth_prob = self._truth_prob_with_hypotheticals(phi, resolved, hypothetical_true)
        return float(np.clip(truth_prob, 0.0, 1.0))

    def _success_prob_from_state(
        self,
        phi: int,
        resolved: set[int],
        hypothetical_true: Optional[set[int]] = None,
    ) -> float:
        """p = pi * (1 - (1 + s_mass)^(-rho)), with s_mass = sum_psi w(psi, phi)."""
        s_mass = 0.0
        for psi in resolved:
            w = self.weights[psi, phi]
            if w > 0.0:
                s_mass += float(w)
        pi = self._pi_from_state(phi, resolved, hypothetical_true)
        p = pi * (1.0 - (1.0 + s_mass) ** (-self.rho))
        return float(np.clip(p, 0.0, 1.0))

    def _expected_time_from_state(
        self,
        phi: int,
        resolved: set[int],
        hypothetical_true: Optional[set[int]] = None,
    ) -> float:
        p = self._success_prob_from_state(phi, resolved, hypothetical_true)
        # Memoryless per-step success probability => geometric completion time.
        # Report E[T | T <= horizon_H], which stays bounded by the horizon.
        if p <= 1e-12:
            return float(self.horizon_H)
        expected_remaining = 0.0
        prev = 0.0
        for t in range(1, self.horizon_H + 1):
            cdf = 1.0 - (1.0 - p) ** t
            expected_remaining += (cdf - prev) * t
            prev = cdf

        return float(expected_remaining / prev) if prev > 1e-12 else float(self.horizon_H)
