"""Random unary factors on the truth and weight variables (generation stage 1).

Alongside the Barabási–Albert *structure* (which pairwise factors exist), we draw a
single-variable ("unary") factor for each truth and each weight variable. These tilt the
distribution toward particular truths/weights independently of the implication penalty —
e.g. "theorem t is unlikely to be false" or "the weight ¬φ_a → ¬φ_b prefers to be high".

Functional form (both linear in the variable, biases drawn uniformly per build):

* **Truth** ``T_t ∈ {0, 1}``::

      log φ_t(T_t) = bias_t * T_t,   bias_t ~ Uniform[-truth_cutoff, +truth_cutoff]

  ``bias_t > 0`` favours ``T_t = 1`` (the sign-1 pair member true); ``bias_t < 0`` favours
  ``T_t = 0``. ``|bias_t|`` is the log-odds magnitude of the preference.

* **Weight** ``W_{A→B} ∈ levels`` (levels ⊂ [0, 1])::

      log φ_{A→B}(w) = bias_{A→B} * w,  bias_{A→B} ~ Uniform[-weight_cutoff, +weight_cutoff]

  ``bias > 0`` tilts toward high weight, ``bias < 0`` toward low. The tilt at ``w = 1`` is
  ``bias``, directly comparable to the pairwise ``weight_penalty``.

  A deterministic age-direction bias may also be added: lower theorem id to higher
  theorem id gets ``+age_directional_weight_bias * w``; reverse direction gets the
  negative. In the BA generator, lower theorem id means earlier theorem.

A cutoff of 0 disables that random family; ``age_directional_weight_bias=0`` disables
the deterministic tilt. All biases at 0 recover the pairwise-only model exactly. The
factor graph stores these as generic log-potential arrays, so the sampler stays correct
for any unary shape — this module just picks the linear-bias form.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from tests.math.reference.factor_graph import weight_variable_ids


def sample_unary_factors(
    cfg,
    theorem_edges: Iterable[Tuple[int, int]],
    levels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """Draw a unary factor for every truth and every weight variable.

    Returns ``(truth_unary, weight_unary)`` ready to hand to ``FormulaFactorGraph``:

    * ``truth_unary``: ``(num_theorems, 2)`` log-potentials, row ``t`` = ``[0, bias_t]``.
    * ``weight_unary``: ``{(src_phi, dst_phi): bias * levels}`` over every weight variable.
    """
    n = int(cfg.num_theorems)
    levels = np.asarray(levels, dtype=np.float64)
    truth_cutoff = float(cfg.truth_unary_cutoff)
    weight_cutoff = float(cfg.weight_unary_cutoff)
    age_bias = float(getattr(cfg, "age_directional_weight_bias", 0.0))

    truth_unary = np.zeros((n, 2), dtype=np.float64)
    if truth_cutoff > 0.0:
        # log φ_t(T_t) = bias_t * T_t  ⇒  [φ_t(0), φ_t(1)] = [0, bias_t].
        truth_unary[:, 1] = rng.uniform(-truth_cutoff, truth_cutoff, size=n)

    weight_unary: Dict[Tuple[int, int], np.ndarray] = {}
    if weight_cutoff > 0.0 or age_bias > 0.0:
        keys = list(weight_variable_ids(theorem_edges, n))
        random_biases = (
            rng.uniform(-weight_cutoff, weight_cutoff, size=len(keys))
            if weight_cutoff > 0.0
            else np.zeros(len(keys), dtype=np.float64)
        )
        for key, random_bias in zip(keys, random_biases):
            src_phi, dst_phi = key
            t_src, t_dst = src_phi % n, dst_phi % n
            directional_bias = 0.0
            if age_bias > 0.0:
                directional_bias = age_bias if t_src < t_dst else -age_bias
            # log φ_{A→B}(w) = bias * w, evaluated on the discrete levels.
            weight_unary[key] = (float(random_bias) + directional_bias) * levels

    return truth_unary, weight_unary
