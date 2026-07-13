"""SHADE: success-history adaptive Differential Evolution, maximizing.

A population-of-elites search with a compact ``ask()/tell()`` interface. SHADE
keeps N persistent members and perturbs each locally by a
*scaled difference between other members* (``current-to-pbest/1``): the step
auto-aligns to the population's own geometry, which suits narrow, separately-oriented
ridges where a single-mean update stalls. Greedy one-to-one replacement (a child
replaces its parent only if at least as good) makes each slot elitist. F and CR
self-tune from a memory of historically successful values (SHADE).

**Parent re-evaluation.** The inner fitness is noisy and the env seed changes every
generation, so a parent's stored score is not comparable to a child scored under a
different seed. Each generation therefore re-evaluates the parents alongside the
trials (``ask()`` returns ``[pop; trials]`` = 2N vectors) and the greedy comparison
is made between parent and child scored under the *same* seed. Common random numbers
(``cfg.search.crn``) further sharpen that within-generation comparison.

The deployable point is exposed as ``incumbent``: the best member in the most
recently evaluated population, not a population average. The whole object is
pickled every generation for checkpoint/resume, so all state is plain numpy / lists / a
``np.random.Generator``.
"""

from __future__ import annotations

import numpy as np


class SHADE:
    """SHADE (current-to-pbest/1 + archive), maximizing, drop-in for ``search.py``.

    ``ask()`` returns the initial population on the first call (N vectors), then
    ``[parents; trials]`` (2N vectors) every generation after. ``tell(scores)``
    consumes exactly those, doing greedy per-slot replacement and the SHADE memory
    update. N stays constant; the paper's linear population-size reduction is not
    part of this implementation.
    """

    def __init__(
        self,
        initial: np.ndarray,
        sigma: float,
        population: int,
        seed: int = 0,
        pbest_frac: float = 0.25,
        memory_size: int = 6,
        archive_factor: float = 1.0,
    ):
        self.d = d = int(initial.size)
        self.lam = self.n = N = int(population)
        if N < 3:
            raise ValueError("population must be >= 3")
        self.sigma0 = float(sigma)
        self.pbest_frac = float(pbest_frac)
        self.H = int(memory_size)
        self.archive_factor = float(archive_factor)
        if self.sigma0 < 0:
            raise ValueError("sigma must be non-negative")
        if not 0 < self.pbest_frac <= 1:
            raise ValueError("pbest_frac must be in (0, 1]")
        if self.H < 1:
            raise ValueError("memory_size must be >= 1")
        if self.archive_factor < 0:
            raise ValueError("archive_factor must be non-negative")
        self.rng = np.random.default_rng(seed)

        m0 = initial.astype(np.float64).copy()
        self.pop = np.repeat(m0[None, :], N, axis=0)
        self.pop[1:] += self.sigma0 * self.rng.standard_normal((N - 1, d))
        self.fit = None                       # None until the initial population is scored
        self.M_F = np.full(self.H, 0.5)
        self.M_CR = np.full(self.H, 0.5)
        self.memory_pos = 0
        self.archive: list[np.ndarray] = []
        self.gen = 0
        self._incumbent = m0.copy()
        self.incumbent_fitness: float | None = None

        # ask/tell handoff state (pending == "init" | "trials" | None)
        self._pending = None
        self._trials = None
        self._trial_F = None
        self._trial_CR = None
        # last-generation diagnostics
        self._success_rate = 0.0
        self._mean_F = float(self.M_F.mean())
        self._mean_CR = float(self.M_CR.mean())

    # ------------------------------------------------------------------ ask/tell
    def ask(self) -> np.ndarray:
        if self.fit is None:
            self._pending = "init"
            return self.pop.copy()
        self._build_trials()
        self._pending = "trials"
        return np.concatenate([self.pop, self._trials], axis=0)

    def _build_trials(self) -> None:
        N, d, rng = self.n, self.d, self.rng
        order = np.argsort(-self.fit)                     # descending fitness
        n_pbest = max(2, round(self.pbest_frac * N))
        pbest_idx = order[:n_pbest]

        trials = np.empty((N, d))
        F = np.empty(N)
        CR = np.empty(N)
        for i in range(N):
            r = int(rng.integers(self.H))
            f = -1.0
            while f <= 0.0:                                # Cauchy, resample non-positive
                f = self.M_F[r] + 0.1 * rng.standard_cauchy()
            F[i] = min(f, 1.0)
            CR[i] = float(np.clip(rng.normal(self.M_CR[r], 0.1), 0.0, 1.0))

            x_pbest = self.pop[rng.choice(pbest_idx)]
            r1 = int(rng.integers(N))
            while r1 == i:
                r1 = int(rng.integers(N))
            # x_r2 from pop \ {i, r1} union archive
            pool = [self.pop[j] for j in range(N) if j != i and j != r1]
            pool.extend(self.archive)
            x_r2 = pool[int(rng.integers(len(pool)))]

            v = self.pop[i] + F[i] * (x_pbest - self.pop[i]) + F[i] * (self.pop[r1] - x_r2)
            jrand = int(rng.integers(d))
            cross = rng.random(d) < CR[i]
            cross[jrand] = True
            trials[i] = np.where(cross, v, self.pop[i])
        self._trials, self._trial_F, self._trial_CR = trials, F, CR

    def tell(self, scores) -> None:
        obj = np.asarray(scores, dtype=np.float64)
        expected = self.n if self._pending == "init" else 2 * self.n
        if obj.shape != (expected,):
            raise ValueError(f"expected {expected} scores, got shape {obj.shape}")
        if not np.isfinite(obj).all():
            raise ValueError("scores must be finite")
        if self._pending == "init":
            self.fit = obj.copy()
            self._pending = None
        elif self._pending == "trials":
            self._tell_trials(obj)
        else:
            raise RuntimeError("tell() called without a matching ask()")
        b = int(np.argmax(self.fit))
        # Every generation uses a new evaluation seed, so scores are comparable
        # within a generation but not against historical maxima.  The incumbent
        # is therefore the best currently evaluated population member.
        self.incumbent_fitness = float(self.fit[b])
        self._incumbent = self.pop[b].copy()
        self.gen += 1

    def _tell_trials(self, obj: np.ndarray) -> None:
        N = self.n
        parent_obj, trial_obj = obj[:N], obj[N:]
        win = trial_obj >= parent_obj
        s_F, s_CR, s_w = [], [], []
        for i in range(N):
            if win[i]:
                self.archive.append(self.pop[i].copy())
                self.pop[i] = self._trials[i]
                self.fit[i] = trial_obj[i]
                s_F.append(self._trial_F[i])
                s_CR.append(self._trial_CR[i])
                s_w.append(trial_obj[i] - parent_obj[i])
            else:
                self.fit[i] = parent_obj[i]               # refresh stale parent score

        if s_F:
            w = np.asarray(s_w) + 1e-30
            f = np.asarray(s_F)
            cr = np.asarray(s_CR)
            self.M_F[self.memory_pos] = float((w * f * f).sum() / (w * f).sum())   # Lehmer
            self.M_CR[self.memory_pos] = float((w * cr).sum() / w.sum())
            self.memory_pos = (self.memory_pos + 1) % self.H
        self._trim_archive()
        self._success_rate = float(win.mean())
        self._mean_F = float(self.M_F.mean())
        self._mean_CR = float(self.M_CR.mean())
        self._pending = None

    def _trim_archive(self) -> None:
        cap = round(self.archive_factor * self.n)
        while len(self.archive) > cap:
            self.archive.pop(int(self.rng.integers(len(self.archive))))

    # ------------------------------------------------------------- diagnostics
    @property
    def incumbent(self) -> np.ndarray:
        return self._incumbent.copy()

    @property
    def pbest_count(self) -> int:
        return max(2, round(self.pbest_frac * self.n))

    def _coord_std(self) -> np.ndarray:
        return self.pop.std(axis=0)

    @property
    def sigma(self) -> float:
        return float(self._coord_std().mean())

    @property
    def axis_ratio(self) -> float:
        s = self._coord_std()
        lo = float(s.min())
        return float(s.max() / lo) if lo > 0 else 1.0

    def coordinate_variances(self) -> np.ndarray:
        return np.maximum(self.pop.var(axis=0), 1e-20)

    def internals(self) -> dict:
        return {
            "shade/population": float(self.n),
            "shade/mean_f": self._mean_F,
            "shade/mean_cr": self._mean_CR,
            "shade/incumbent_fitness": self.incumbent_fitness,
            "shade/success_rate": self._success_rate,
            "shade/archive_size": float(len(self.archive)),
            "shade/population_diversity": float(self._coord_std().mean()),
        }
