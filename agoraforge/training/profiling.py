"""Small helpers for structured training-loop profiling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextlib import nullcontext
import time


class NullProfiler:
    def section(self, name: str):
        del name
        return nullcontext()


class EpochProfiler:
    """Accumulates named wall-clock timings for one epoch.

    ``sync`` is an optional callable run at each section's entry and exit. On an
    async accelerator (CUDA) the device queues work without blocking, so
    naive wall-clock timing charges a section for whatever later op forces a
    sync. Passing ``torch.cuda.synchronize`` drains the queue at both boundaries,
    so each section is charged only its own device work -- at the cost of the
    pipelining the syncs prevent, hence opt-in for diagnostic runs only.
    """

    def __init__(self, sync=None):
        self._sync = sync
        self._start = time.perf_counter()
        self._durations = defaultdict(float)

    @contextmanager
    def section(self, name: str):
        if self._sync is not None:
            self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync is not None:
                self._sync()
            self._durations[name] += time.perf_counter() - start

    def elapsed(self, name: str) -> float:
        return float(self._durations.get(name, 0.0))

    def total(self) -> float:
        return time.perf_counter() - self._start

    def scalar_items(self) -> list[tuple[str, float]]:
        items = sorted(self._durations.items())
        items.append(("total", self.total()))
        return [(name, float(value)) for name, value in items]

    def format_line(self, *, epoch: int, total_epochs: int) -> str:
        total = self.total()
        parts = [f"profile epoch={epoch + 1}/{total_epochs}", f"total_s={total:.2f}"]
        parts.extend(
            f"{name}_s={value:.2f}"
            for name, value in sorted(self._durations.items())
            if value > 0.0
        )
        return " ".join(parts)
