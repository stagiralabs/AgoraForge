"""Curriculum construction, schedule interpolation, and level allocation."""

from __future__ import annotations

import numpy as np

from agoraforge.conf.schema import build_env_config


def build_training_levels(cfg):
    levels = {
        level["name"]: build_env_config(cfg, level=level)
        for level in cfg.levels
    }
    return levels, cfg.schedule


def print_schedule(schedule) -> None:
    print(f"\nSchedule ({len(schedule)} points):")
    for point in schedule:
        active = [f"{name}={weight:.0%}" for name, weight in point["weights"].items()]
        print(f"  epoch {point['epoch']}: {', '.join(active)}")


def get_level_weights(schedule, epoch):
    """Return normalized weights by interpolating between schedule points.

    Each schedule entry specifies the target mixture at an epoch. Before the
    first point, the first mixture is used directly. Between consecutive points,
    weights are linearly interpolated. After the final point, the final mixture
    is maintained.
    """

    # Collect all level names across points in config order. The deterministic
    # sampler below relies on stable ordering for tie-breaks.
    all_levels = []
    seen_levels = set()
    for point in schedule:
        for name in point['weights'].keys():
            if name not in seen_levels:
                seen_levels.add(name)
                all_levels.append(name)

    def _normalize(weights):
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    if epoch <= schedule[0]['epoch']:
        w = schedule[0]['weights']
        return _normalize({name: w.get(name, 0.0) for name in all_levels})

    for i in range(len(schedule) - 1):
        start = schedule[i]['epoch']
        end = schedule[i + 1]['epoch']
        if epoch <= end:
            alpha = (epoch - start) / (end - start)
            w_before = schedule[i]['weights']
            w_after = schedule[i + 1]['weights']
            raw = {}
            for name in all_levels:
                raw[name] = (1.0 - alpha) * w_before.get(name, 0.0) + alpha * w_after.get(name, 0.0)
            return _normalize(raw)

    w = schedule[-1]['weights']
    return _normalize({name: w.get(name, 0.0) for name in all_levels})


def sample_levels(weights, n):
    """Allocate n level names deterministically according to weights.

    Counts are computed by flooring each exact weighted allocation, then
    assigning any leftover rollouts to the levels with the largest fractional
    remainders. For example, weights {easy: 0.4, hard: 0.6} and n=10 produces
    exactly 4 easy rollouts and 6 hard rollouts.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return []

    names = list(weights.keys())
    total_weight = sum(max(float(weights[name]), 0.0) for name in names)
    if total_weight <= 0:
        raise ValueError("Cannot allocate levels with non-positive total weight")

    exact_counts = {
        name: (max(float(weights[name]), 0.0) / total_weight) * n
        for name in names
    }
    counts = {name: int(np.floor(exact_counts[name])) for name in names}
    remaining = n - sum(counts.values())

    remainders = sorted(
        names,
        key=lambda name: (-(exact_counts[name] - counts[name]), names.index(name)),
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    return [name for name in names for _ in range(counts[name])]
