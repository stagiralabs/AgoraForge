"""N-scaling protocol for mechanism-search runs.

One place that defines how a run at problem size N is configured, so a run at
any N follows the same rule.

The rule, holding the *regime* fixed as N grows:
  - n_agents ≈ N / 32    -- a small team relative to the graph, so each agent
                            owns a large slice and coordination stays hard. At
                            the N=64 floor this is 2 agents, 4 at N=128, etc.
  - target density FIXED -- prob_initially_target = 1/64, so the target count
                            grows with N: one target at N=64, two at N=128,
                            four at N=256.
  - one resolved seed    -- prob_initially_resolved = 1/N gives exactly one
                            initially-resolved theorem at any N.
  - per-agent obs window FIXED -- a constant per-agent formula window
                            (obs_formula_cap) so the observation/model size
                            does not grow with N.

The fitness horizon is FIXED at all N: ``fitness_tau`` = 100 and
``max_timestep`` = 75, independent of N and of the planner-median resolution
time. A large, N-invariant tau rewards far-off multi-hop targets and keeps the
learning signal comparable as N grows; we deliberately do NOT median-calibrate
per N (that collapses the horizon onto whatever the planner already finds easy).
"""
from __future__ import annotations

# Fixed fitness horizon, used at every N. tau=100 rewards far-off,
# multi-hop targets; max_timestep=75 caps the episode. Neither depends on N --
# do not median-calibrate.
FITNESS_TAU = 100.0
MAX_TIMESTEP = 75
OBS_FORMULA_CAP = 16
THEOREMS_PER_AGENT = 32.0
TARGET_PROB = 1.0 / 64.0
MIN_THEOREMS = 64


def scale_env_overrides(num_theorems: int) -> dict:
    """train.py `--config.env.*` overrides for a run at N=``num_theorems``.

    ``fitness_tau`` and ``max_timestep`` are the fixed constants
    ``FITNESS_TAU``/``MAX_TIMESTEP`` at every N. Returns a dict of {env-key: value};
    the caller renders it to ``--config.env.<key>=<value>`` flags.
    """
    n = int(num_theorems)
    if n < MIN_THEOREMS:
        raise ValueError(f"num_theorems must be >= {MIN_THEOREMS}, got {n}")
    n_agents = max(1, round(n / THEOREMS_PER_AGENT))

    return {
        "num_theorems": n,
        "n_agents": int(n_agents),
        # Fixed target density: target count grows as N/64 (>=1 in the env).
        "prob_initially_target": TARGET_PROB,
        # Exactly one initially-resolved theorem at any N.
        "prob_initially_resolved": 1.0 / float(n),
        # Fixed at all N (see module note); never per-N calibrated.
        "fitness_tau": FITNESS_TAU,
        "max_timestep": MAX_TIMESTEP,
        # Fixed per-agent observation window. K=16 is the validated resident cap
        # from the N=64/N=256 GPU path; keep it constant as N grows.
        "obs_formula_cap": OBS_FORMULA_CAP,
    }
