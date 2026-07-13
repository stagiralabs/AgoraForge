"""CPU reference FormulaGraph for a theorem-level math-env instance.

This is the *materialized sample* — a single concrete formula graph. The distribution
it is drawn from, and the sampler that draws it, live in ``factor_graph.py`` /
``sampling.py``.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np


class FormulaGraph:
    """Immutable latent structure of a math-env instance."""

    def __init__(
        self,
        num_theorems: int,
        truth_map: np.ndarray,
        utility_weights: Dict[Tuple[int, int], float],
    ):
        self.num_theorems = int(num_theorems)
        self.F_size = 2 * self.num_theorems
        self.half_F = self.num_theorems

        self.theorem_truth_map = truth_map.copy()
        self.truth_map = np.zeros(self.F_size, dtype=np.int32)
        # A weight of exactly 0 denotes an absent edge: get_weight() returns 0.0 for any
        # missing key, so storing it would be redundant. Drop it to keep the dict sparse.
        self.utility_weights: Dict[Tuple[int, int], float] = {
            (int(src), int(dst)): float(weight)
            for (src, dst), weight in utility_weights.items()
            if float(weight) > 0.0
        }

        for theorem_id in range(self.num_theorems):
            true_phi = self.true_formula(theorem_id)
            false_phi = self.neg(true_phi)
            self.truth_map[true_phi] = 1
            self.truth_map[false_phi] = 0

        self.true_formulas: Set[int] = {
            self.true_formula(theorem_id) for theorem_id in range(self.num_theorems)
        }

        self._validate_weights()

    def formula_from_pair(self, sign: int, theorem_id: int) -> int:
        return theorem_id + sign * self.num_theorems

    def pair_sign(self, phi: int) -> int:
        return 0 if phi < self.num_theorems else 1

    def theorem_id(self, phi: int) -> int:
        return phi % self.num_theorems

    def neg(self, i: int) -> int:
        theorem_id = self.theorem_id(i)
        return self.formula_from_pair(1 - self.pair_sign(i), theorem_id)

    def true_formula(self, theorem_id: int) -> int:
        sign = int(self.theorem_truth_map[theorem_id])
        return self.formula_from_pair(sign, theorem_id)

    def is_true(self, phi: int) -> bool:
        return self.truth_map[phi] == 1

    def get_weight(self, psi: int, phi: int) -> float:
        return self.utility_weights.get((psi, phi), 0.0)

    def _validate_weights(self):
        nodes = set(range(self.F_size))
        for (psi, phi), weight in self.utility_weights.items():
            assert psi in nodes and phi in nodes, f"invalid formula weight key ({psi},{phi})"
            assert psi != phi, f"self weight w({psi},{phi}) is not allowed"
            assert weight > 0.0, f"w({psi},{phi}) must be positive"
