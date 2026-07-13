"""CPU reference per-agent knowledge state for the math environment.

Each agent maintains a private library with:
- concrete set C: formulas whose truth value is known
- resolved set RS: formulas that have been proved
"""

from __future__ import annotations


class Library:
    """Agent knowledge state: concrete set C and resolved set RS."""

    def __init__(self, F_size: int):
        self.F_size = F_size
        self.half_F = F_size // 2
        self.concrete: set[int] = set()
        self.resolved: set[int] = set()

    def is_valid(self) -> bool:
        if self.half_F < 0:
            return False
        if self.F_size != 2 * self.half_F:
            return False
        for formula in self.concrete:
            if formula < 0 or formula >= self.F_size:
                return False
            if self.neg(formula) not in self.concrete:
                return False
        for formula in self.resolved:
            if formula < 0 or formula >= self.F_size:
                return False
            if formula not in self.concrete:
                return False
            if self.neg(formula) in self.resolved:
                return False
        return True

    def neg(self, i: int) -> int:
        return i + self.half_F if i < self.half_F else i - self.half_F

    def _validate_formula(self, phi: int) -> int:
        phi = int(phi)
        if phi < 0 or phi >= self.F_size:
            raise ValueError(f"formula index {phi} out of range [0, {self.F_size})")
        return phi

    def add_concrete(self, phi: int) -> None:
        """Add phi and its negation to the concrete set (negation pairing)."""
        phi = self._validate_formula(phi)
        self.concrete.add(phi)
        self.concrete.add(self.neg(phi))
        assert self.is_valid(), "Library invariant violation after add_concrete"

    def add_resolved(self, phi: int) -> None:
        """Add phi to the resolved set.

        Enforces: phi and neg(phi) cannot both be resolved.
        """
        phi = self._validate_formula(phi)
        neg_phi = self.neg(phi)
        assert neg_phi not in self.resolved, \
            f"Cannot resolve {phi}: negation {neg_phi} already resolved"
        self.resolved.add(phi)
        self.add_concrete(phi)
        assert self.is_valid(), "Library invariant violation after add_resolved"

    def resolved_formulas(self) -> set[int]:
        """Return the set of resolved formula indices."""
        return set(self.resolved)
