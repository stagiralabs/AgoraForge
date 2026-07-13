"""Outer loop and public CLI for black-box search over learned mechanisms.

The searched mechanism has parameters ``theta`` (debate protocols or math market
mechanisms). A ``theta`` produces a fitness after agents train under the mechanism.
SHADE optimizes this non-differentiable, bi-level objective.

Config-driven: the whole run is one config file. ``cfg.search`` holds the outer-loop
settings and ``cfg.env`` the inner learned world / mechanism shape (see
``agoraforge/conf/runs/debate/protocol_search.py``). The only command-line inputs are
which config to run and where to write outputs:

    python -m agoraforge.search --config=agoraforge/conf/runs/debate/protocol_search.py --out <dir>

The smoke is just another config (``agoraforge/conf/runs/debate/protocol_search_smoke.py``).

Each generation's whole population is folded into a single env batch with K stacked
policies trained cold. The maximized objective is the environment's primary metric.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from agoraforge.conf.schema import build_env_config, load_run_config
from agoraforge.envs.mechanism import MechanismDims
from agoraforge.envs.registry import get_env
from agoraforge.searching.shade import SHADE
from agoraforge.searching.worker import evaluate_population_sharded
from agoraforge.training.device import select_device


def _robust_stats(values: np.ndarray, prefix: str) -> dict:
    """Median, p75, and top-quartile mean of a per-candidate vector."""
    values = np.asarray(values, dtype=np.float64)
    top_k = max(1, int(np.ceil(0.25 * values.size)))
    top25 = np.sort(values)[-top_k:]
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_top25_mean": float(top25.mean()),
    }


def _param_slices(mechanism_cls, dims: MechanismDims, layernorm: bool) -> list[tuple[str, slice]]:
    """Named flat-theta slices in ``flat_params`` order (shared trunk params once)."""
    mechanism = mechanism_cls(dims, layernorm=layernorm)
    slices, start = [], 0
    for name, p in mechanism.named_parameters():
        slices.append((name, slice(start, start + p.numel())))
        start += p.numel()
    return slices


def fresh_theta(mechanism_cls, dims: MechanismDims, seed: int = 0, layernorm: bool = False,
                perturb: float = 0.0) -> torch.Tensor:
    """Flat params of a fresh mechanism. With ``perturb`` > 0, add Gaussian noise of
    that std to every parameter so the residual map F is no longer the zero
    (identity) update."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        theta = mechanism_cls(dims, layernorm=layernorm).flat_params()
        if perturb > 0:
            theta = theta + perturb * torch.randn_like(theta)
    return theta


def _cleanup_inner(inner_dir: Path) -> None:
    """Drop a finished generation's inner-run ephemera."""
    shutil.rmtree(inner_dir, ignore_errors=True)


def policy_seed_base_for_generation(gen: int) -> int:
    """Policy RNG draw for a generation, shared across candidates under CRN.

    CRN reduces within-generation ranking noise by giving every candidate the same
    policy RNG stream. The generation offset keeps held-out probes from measuring
    one fixed policy initialization/action stream for the entire search.
    """
    return 777 + 10_000 * int(gen)


def evaluate_population_generation(
    cfg,
    candidates,
    *,
    seed: int,
    policy_seed_base: int,
    inner_epochs: int,
    tail_epochs: int,
    chunk_size: int = 0,
    amp_dtype: str = "",
    crn: bool = False,
) -> dict:
    """Score a whole generation in one batched in-process run.

    Trains all ``len(candidates)`` mechanisms' policies folded into a single env batch
    (B = K * base_batch on common random numbers) with K stacked policies vmapped and
    K independent PPO updates. Returns the tail-averaged primary metric per candidate
    plus the environment's raw evaluation metrics.

    The inner buffer width (``base_batch``) comes from ``cfg.training.online_buffer_size``
    -- the per-candidate env count the subprocess path would use for one run.
    """
    from agoraforge.training.population.evaluation import evaluate_population

    vcfg = build_env_config(cfg, level=cfg.levels[0])
    device = select_device(cfg.device)
    thetas = [torch.from_numpy(c.astype(np.float32)) for c in candidates]
    info = evaluate_population(
        thetas,
        cfg=cfg,
        vcfg=vcfg,
        base_batch_size=int(cfg.training.online_buffer_size),
        inner_epochs=inner_epochs,
        tail_epochs=tail_epochs,
        seed=seed,
        device=device,
        policy_seed_base=policy_seed_base,
        chunk_size=chunk_size,
        amp_dtype=amp_dtype,
        crn=crn,
    )
    return info


def build_strategy(s, theta: torch.Tensor):
    """Construct the repository's single supported search strategy."""
    strategy = SHADE(
        theta.numpy(), sigma=s.sigma, population=s.population, seed=s.seed,
        pbest_frac=float(s.pbest_frac), memory_size=int(s.memory_size),
        archive_factor=float(s.archive_factor),
    )
    print(f"SHADE: N={strategy.n}, pbest={strategy.pbest_count}/{strategy.n}, "
          f"H={strategy.H}, sigma0={strategy.sigma0:.3g}", flush=True)
    return strategy


def search(cfg, out_dir: Path, config_path: str) -> None:
    """Run the mechanism search defined by ``cfg``.

    The full strategy state is checkpointed to ``search_state.pt`` after every
    generation, and a run whose ``out_dir`` already holds one resumes from it —
    relaunching with the same ``--out`` continues the same search (cluster time
    limits chunk one logical run into several jobs). Per-generation env seeds are
    derived from the generation index, so a resumed run scores the exact
    generations the uninterrupted run would have."""
    s = cfg.search
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "search_state.pt"
    config_snapshot_path = out_dir / "search_config.json"
    config_snapshot = json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
    state = None
    if state_path.exists():
        if not config_snapshot_path.exists():
            raise ValueError(f"cannot resume without {config_snapshot_path}")
        if json.loads(config_snapshot_path.read_text()) != json.loads(config_snapshot):
            raise ValueError("search config differs from the saved resume config")
        state = torch.load(state_path, weights_only=False)
        print(f"resuming from {state_path} at gen {state['gen'] + 1}", flush=True)
    else:
        config_snapshot_path.write_text(config_snapshot + "\n")

    wandb_run = None
    if bool(s.wandb_enabled):
        try:
            import wandb
        except ImportError:
            raise RuntimeError(
                "W&B logging requires the optional dependency: "
                "pip install 'agoraforge[wandb]'"
            ) from None

        wandb_run = wandb.init(
            project=s.wandb_project,
            entity=s.wandb_entity or None,
            name=s.wandb_name or out_dir.name,
            config=cfg.to_dict(),
            dir=str(out_dir),
            mode=s.wandb_mode,
            tags=list(s.wandb_tags),
            id=state["wandb_id"] if state else None,
            resume="allow",
        )

    spec = get_env(cfg.env_name)
    primary = spec.primary_metric
    mechanism_cls = spec.mechanism.LEARNED_MECHANISM
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    dims = spec.mechanism.mechanism_dims_from_config(vcfg)
    layernorm = bool(cfg.env.get("learned_mechanism_layernorm", False))
    theta = fresh_theta(mechanism_cls, dims, seed=s.init_seed, layernorm=layernorm,
                        perturb=s.init_perturb)
    n = theta.numel()
    print(f"mechanism has {n} parameters; dims={dims}", flush=True)

    strategy = state["strategy"] if state else build_strategy(s, theta)
    gpus = [g.strip() for g in (s.gpus.split(",") if s.gpus else []) if g.strip()]
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"cfg.search.gpus contains duplicates: {gpus}")
    if int(s.threads_per_worker) < 1:
        raise ValueError("cfg.search.threads_per_worker must be positive")
    if len(gpus) <= 1 and int(s.get("exact_stream_workers", 0)):
        raise ValueError("exact_stream_workers currently requires sharded multi-GPU search")

    # Probes are named explicitly in config and excluded from the SHADE update. The
    # incumbent is the best member observed before this generation's seed. The
    # zero-reward mechanism is freshly initialized without perturbation: its final
    # layer is zero, so it produces a zero state/reward delta (and P=.5 in debate).
    probe_names = list(s.probes)
    supported_probes = {"incumbent", "zero_reward"}
    unknown_probes = set(probe_names) - supported_probes
    if unknown_probes or len(set(probe_names)) != len(probe_names):
        raise ValueError(
            f"cfg.search.probes must be unique names from {sorted(supported_probes)}; "
            f"got {probe_names}"
        )
    zero_reward_theta = fresh_theta(
        mechanism_cls, dims, seed=s.init_seed, layernorm=layernorm
    ).numpy().astype(np.float32)

    param_slices = _param_slices(mechanism_cls, dims, layernorm)
    if state:
        initial_incumbent = state["initial_incumbent"]
        prev_step = state["prev_step"]
        start_gen = state["gen"] + 1
    else:
        initial_incumbent = strategy.incumbent.copy()
        prev_step = None
        start_gen = 0

    for gen in range(start_gen, s.generations):
        seed = int(s.seed_base + gen)
        policy_seed_base = policy_seed_base_for_generation(gen)

        candidates = [row.astype(np.float32) for row in strategy.ask()]
        n_pop = len(candidates)
        eval_candidates = candidates
        if probe_names:
            probes = {
                "incumbent": strategy.incumbent.astype(np.float32),
                "zero_reward": zero_reward_theta,
            }
            eval_candidates = candidates + [probes[name] for name in probe_names]

        # Score each candidate on the environment's primary metric.
        if gpus and len(gpus) > 1:
            info = evaluate_population_sharded(
                eval_candidates, seed=seed, out_dir=out_dir, gen=gen,
                policy_seed_base=policy_seed_base,
                config_path=config_path,
                gpus=gpus, chunk_size=int(s.batched_chunk),
                threads_per_worker=int(s.threads_per_worker),
                crn=bool(s.crn),
            )
        else:
            # One in-process batched generation: the whole population trains folded
            # into a single env batch + K stacked policies (see evaluate_population).
            info = evaluate_population_generation(
                cfg, eval_candidates, seed=seed, policy_seed_base=policy_seed_base,
                inner_epochs=s.inner_epochs,
                tail_epochs=s.tail_epochs,
                chunk_size=int(s.batched_chunk),
                amp_dtype=str(s.amp_dtype),
                crn=bool(s.crn),
            )
        fit_all = np.asarray(info[primary], dtype=np.float64)
        fitness = fit_all[:n_pop]
        if not np.isfinite(fitness).all():
            raise ValueError(f"generation {gen} produced non-finite search metrics")
        # Anchors are debate eval metrics (prior-judge floor / full-graph ceiling);
        # envs without them skip the anchor telemetry.
        has_anchors = "judge_acc_prior" in info and "judge_acc_full" in info
        acc_prior = (float(np.asarray(info["judge_acc_prior"], dtype=np.float64)[:n_pop].mean())
                     if has_anchors else None)
        acc_full = (float(np.asarray(info["judge_acc_full"], dtype=np.float64)[:n_pop].mean())
                    if has_anchors else None)
        probe_values = dict(zip(probe_names, fit_all[n_pop:].tolist()))
        if probe_values and not np.isfinite(list(probe_values.values())).all():
            raise ValueError(f"generation {gen} produced non-finite probe metrics")

        previous_incumbent = strategy.incumbent.copy()
        strategy.tell(fitness)
        incumbent_theta = torch.from_numpy(strategy.incumbent.astype(np.float32))

        # Incumbent-trajectory diagnostics: step length, direction persistence,
        # drift from initialization, and whether the in-generation ordering relates
        # to distance from the previous incumbent.
        step = strategy.incumbent - previous_incumbent
        step_l2 = float(np.linalg.norm(step))
        step_cos = None
        if prev_step is not None:
            denom = float(np.linalg.norm(step) * np.linalg.norm(prev_step))
            if denom > 0.0:
                step_cos = float(step @ prev_step / denom)
        prev_step = step
        selection_gain = float(
            np.sort(fitness)[-strategy.pbest_count:].mean() - fitness.mean()
        )

        gen_best = float(fitness.max())
        best_idx = int(fitness.argmax())
        # Save each generation's elite so the search can be judged by re-evaluating
        # it on held-out seeds, not the noise-inflated in-run max(pop).
        gen_best_theta = torch.from_numpy(candidates[best_idx].copy())
        torch.save(gen_best_theta, out_dir / f"gen{gen:03d}_best.pt")
        torch.save(incumbent_theta.clone(), out_dir / "incumbent.pt")

        with (out_dir / "candidates.jsonl").open("a") as f:
            f.write(json.dumps({
                "gen": gen,
                primary: fitness.tolist(),
                "probes": probe_values,
                "acc_prior": acc_prior,
                "acc_full": acc_full,
            }) + "\n")
        anchor_span = (acc_full - acc_prior) if has_anchors else None
        progress = {
            "gen": gen,
            f"{primary}_mean": float(fitness.mean()),
            f"{primary}_max": float(fitness.max()),
            **_robust_stats(fitness, primary),
            "anchor/acc_prior": acc_prior,
            "anchor/acc_full": acc_full,
            "anchor/acc_frac_mean": (
                float((fitness.mean() - acc_prior) / anchor_span)
                if has_anchors and abs(anchor_span) > 1e-9 else None
            ),
            "theta_norm": float(incumbent_theta.norm()),
            **strategy.internals(),
            "shade/incumbent_step_l2": step_l2,
            "shade/incumbent_step_cos": step_cos,
            "shade/incumbent_dist_from_init": float(
                np.linalg.norm(strategy.incumbent - initial_incumbent)
            ),
            "shade/selection_gain": selection_gain,
        }
        coord_var = strategy.coordinate_variances()
        for name, sl in param_slices:
            progress[f"theta_block_norm/{name}"] = float(np.linalg.norm(strategy.incumbent[sl]))
            progress[f"shade/variance_share/{name}"] = float(
                coord_var[sl].sum() / coord_var.sum()
            )
        if "zero_reward" in probe_values:
            zero = probe_values["zero_reward"]
            progress["probe/zero_reward"] = zero
            progress["probe/median_minus_zero_reward"] = (
                progress[f"{primary}_median"] - zero
            )
        if "incumbent" in probe_values:
            incumbent = probe_values["incumbent"]
            progress["probe/incumbent"] = incumbent
            if "zero_reward" in probe_values:
                progress["probe/incumbent_minus_zero_reward"] = (
                    incumbent - probe_values["zero_reward"]
                )
            if has_anchors and abs(anchor_span) > 1e-9:
                progress["probe/incumbent_fraction_of_anchors"] = (
                    incumbent - acc_prior
                ) / anchor_span
        with (out_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(progress) + "\n")
        if wandb_run is not None:
            # commit=True: with an explicit step, W&B defaults to commit=False and
            # holds the row until the next generation's log call arrives.
            wandb_run.log({k: v for k, v in progress.items() if v is not None},
                          step=gen, commit=True)

        probe_str = ""
        if probe_names:
            probe_str = "| probes " + " ".join(
                f"{name} {probe_values[name]:.3f}" for name in probe_names
            ) + " "
        print(
            f"gen {gen:3d} | {primary} mean {fitness.mean():.3f} "
            f"max {gen_best:.3f} "
            f"| median {progress[f'{primary}_median']:.3f} "
            f"p75 {progress[f'{primary}_p75']:.3f} "
            f"top25 {progress[f'{primary}_top25_mean']:.3f} "
            + (f"(prior {acc_prior:.3f} full {acc_full:.3f}) " if has_anchors else "")
            + f"{probe_str}"
            f"| sigma {strategy.sigma:.3g} axis {strategy.axis_ratio:.1f}",
            flush=True,
        )

        # Write-then-rename so a mid-write kill (cluster time limit) can't corrupt
        # the resume point.
        torch.save({
            "strategy": strategy,
            "gen": gen,
            "prev_step": prev_step,
            "initial_incumbent": initial_incumbent,
            "wandb_id": wandb_run.id if wandb_run is not None else None,
        }, state_path.with_suffix(".tmp"))
        state_path.with_suffix(".tmp").replace(state_path)

        # No policy state carries across generations; bound temporary disk use.
        _cleanup_inner(out_dir / "inner")
    if wandb_run is not None:
        wandb_run.finish()

    if start_gen >= s.generations:
        print("done without evaluating a generation")
    else:
        print(f"done. final incumbent -> {out_dir/'incumbent.pt'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Black-box mechanism search (config-driven).")
    p.add_argument("--config", required=True,
                   help="run config: defines the outer search and inner learned world")
    p.add_argument("--out", default="results/learned_search", help="results directory")
    args = p.parse_args(argv)
    config_path = str(Path(args.config).resolve()) if args.config.endswith(".py") else args.config
    cfg = load_run_config(config_path)
    search(cfg, out_dir=Path(args.out).resolve(), config_path=config_path)


if __name__ == "__main__":
    main()
