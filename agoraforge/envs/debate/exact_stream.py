"""Asynchronous CPU production of exact debate instances for GPU evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch

from agoraforge.envs.debate.exact import exact_truth_and_marginals
from agoraforge.envs.debate.latent import sample_claim_graph_parameters


def _produce_chunk(payload):
    cfg, count, seed = payload
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    couplings, unary = sample_claim_graph_parameters(
        cfg, count, torch.device("cpu"), generator,
    )
    truth, p_full, _ = exact_truth_and_marginals(couplings, unary, generator)
    target_idx = torch.randint(
        0, int(cfg.num_claims), (count,), generator=generator,
    )
    return {
        "couplings": couplings.numpy(),
        "unary": unary.numpy(),
        "truth": truth.numpy(),
        "p_full": p_full.numpy(),
        "target_idx": target_idx.numpy(),
    }


class ExactDebateInstanceStream:
    """Two-batch lookahead stream backed by a persistent spawned process pool."""

    def __init__(self, cfg, batch_size: int, epochs: int, seed: int, workers: int = 16):
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.workers = min(int(workers), self.batch_size)
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
        )
        self._submitted = 0
        self._yielded = 0
        self._pending = []
        self.wait_seconds = 0.0
        self._submit_next()
        self._submit_next()

    def _submit_next(self):
        if self._submitted >= self.epochs:
            return
        epoch = self._submitted
        counts = [self.batch_size // self.workers] * self.workers
        for k in range(self.batch_size % self.workers):
            counts[k] += 1
        futures = [
            self.executor.submit(
                _produce_chunk,
                (self.cfg, count, self.seed + epoch * 1_000_003 + k * 10_007),
            )
            for k, count in enumerate(counts)
            if count
        ]
        self._pending.append(futures)
        self._submitted += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded >= self.epochs:
            raise StopIteration
        started = time.perf_counter()
        outputs = [future.result() for future in self._pending.pop(0)]
        self.wait_seconds += time.perf_counter() - started
        self._yielded += 1
        self._submit_next()
        return {
            key: torch.from_numpy(np.concatenate([output[key] for output in outputs]))
            for key in outputs[0]
        }

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
