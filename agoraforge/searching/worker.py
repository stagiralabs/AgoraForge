"""Portable multi-GPU dispatch and subprocess worker for search evaluation.

One worker scores a chunk of candidate mechanisms with the in-process batched
evaluator. Every eval knob is read from the run config (``cfg.search`` /
``cfg.training``); only the per-chunk candidate file, output path, env seed, and
policy-seed offset are passed on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from agoraforge.conf.schema import build_env_config, load_run_config
from agoraforge.training.population.evaluation import evaluate_population
from agoraforge.training.device import select_device


def gpu_queues(
    candidate_count: int, chunk_size: int, gpus: list[str]
) -> list[list[tuple[int, int, int]]]:
    """Assign chunks to per-GPU queues, including a final partial chunk.

    A queue is executed serially, so a GPU never receives overlapping workers.
    Separate queues run concurrently. This works for both N-candidate initial
    generations and the later 2N parent-plus-trial generations without imposing a
    divisibility constraint on either population size or probes.
    """
    if chunk_size <= 0:
        raise ValueError("sharded evaluation requires chunk_size > 0")
    if not gpus:
        raise ValueError("sharded evaluation requires at least one GPU")
    queues: list[list[tuple[int, int, int]]] = [[] for _ in gpus]
    for chunk_id, start in enumerate(range(0, candidate_count, chunk_size)):
        end = min(start + chunk_size, candidate_count)
        queues[chunk_id % len(gpus)].append((chunk_id, start, end))
    return queues


def evaluate_population_sharded(
    candidates,
    *,
    seed: int,
    policy_seed_base: int,
    out_dir: Path,
    gen: int,
    config_path: str,
    gpus: list[str],
    chunk_size: int,
    threads_per_worker: int,
    crn: bool = False,
) -> dict:
    """Score candidate chunks concurrently across GPUs, serially within each GPU."""
    queues = gpu_queues(len(candidates), chunk_size, gpus)
    work_dir = Path(out_dir).resolve() / "inner" / f"g{gen:03d}-chunks"
    work_dir.mkdir(parents=True, exist_ok=True)

    def run_chunk(gpu: str, job: tuple[int, int, int]):
        chunk_id, start, end = job
        stem = f"chunk{chunk_id:03d}_{start:03d}_{end:03d}"
        cand_path = work_dir / f"{stem}.npy"
        result_path = work_dir / f"{stem}.json"
        log_path = work_dir / f"{stem}.log"
        np.save(cand_path, np.stack(candidates[start:end]).astype(np.float32))
        env = dict(os.environ)
        for var in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = str(threads_per_worker)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        chunk_policy_seed = policy_seed_base if crn else policy_seed_base + start
        cmd = [
            sys.executable, "-m", "agoraforge.searching.worker",
            "--config", config_path,
            "--candidates", str(cand_path),
            "--out", str(result_path),
            "--seed", str(seed),
            "--policy-seed-base", str(chunk_policy_seed),
        ]
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0 or not result_path.exists():
            tail = log_path.read_text(errors="replace").splitlines()[-60:]
            raise RuntimeError(
                f"search chunk {chunk_id} [{start}:{end}] failed on GPU {gpu} "
                f"(exit {proc.returncode}); log={log_path}\n" + "\n".join(tail)
            )
        return start, end, json.loads(result_path.read_text())

    def run_queue(gpu: str, jobs: list[tuple[int, int, int]]):
        return [run_chunk(gpu, job) for job in jobs]

    active = [(gpu, queue) for gpu, queue in zip(gpus, queues) if queue]
    info: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = [pool.submit(run_queue, gpu, queue) for gpu, queue in active]
        for future in futures:
            for start, end, payload in future.result():
                for key, values in payload.items():
                    info.setdefault(key, [0.0] * len(candidates))[start:end] = values
    return info


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--policy-seed-base", type=int, required=True)
    args = p.parse_args(argv)

    cfg = load_run_config(args.config)
    s = cfg.search
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    candidates = np.load(args.candidates).astype(np.float32)
    thetas = [torch.from_numpy(candidates[k]) for k in range(candidates.shape[0])]
    exact_workers = int(s.get("exact_stream_workers", 0))
    stream = None
    if exact_workers:
        if vcfg.env_name != "debate":
            raise ValueError("exact instance stream is debate-only")
        from agoraforge.envs.debate.exact_stream import ExactDebateInstanceStream
        stream = ExactDebateInstanceStream(
            vcfg,
            int(cfg.training.online_buffer_size),
            int(s.inner_epochs),
            args.seed,
            workers=exact_workers,
        )
    try:
        info = evaluate_population(
            thetas,
            cfg=cfg,
            vcfg=vcfg,
            base_batch_size=int(cfg.training.online_buffer_size),
            inner_epochs=int(s.inner_epochs),
            tail_epochs=int(s.tail_epochs),
            seed=args.seed,
            device=select_device(cfg.device),
            policy_seed_base=args.policy_seed_base,
            chunk_size=0,
            amp_dtype=str(s.amp_dtype),
            crn=bool(s.crn),
            instances=stream,
        )
    finally:
        if stream is not None:
            stream.close()
    Path(args.out).write_text(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
