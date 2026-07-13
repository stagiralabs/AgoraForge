"""Cold-policy population evaluation for outer mechanism search."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from agoraforge.training.metrics import local_mean_std_tensor, normalize_advantages_local
from agoraforge.training.population.models import build_population_models
from agoraforge.training.models import shared_optimizer
from agoraforge.training.ppo import PPOHparams, ppo_update_staged
from agoraforge.training.population.rollout import rollout_population
from agoraforge.envs.registry import get_env


def _prof_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def evaluate_population(
    thetas,
    *,
    cfg,
    vcfg,
    base_batch_size: int,
    inner_epochs: int,
    tail_epochs: int,
    seed: int,
    device,
    policy_seed_base: int = 777,
    chunk_size: int = 0,
    profiler=None,
    amp_dtype: str = "",
    crn: bool = False,
    log_every: int = 0,
    instances=None,
):
    K = len(thetas)
    spec = get_env(vcfg.env_name)
    if chunk_size and chunk_size < K:
        infos = {key: [] for key in spec.metric_keys}
        for start in range(0, K, chunk_size):
            sub = list(thetas[start:start + chunk_size])
            inf = evaluate_population(
                sub, cfg=cfg, vcfg=vcfg, base_batch_size=base_batch_size,
                inner_epochs=inner_epochs, tail_epochs=tail_epochs, seed=seed, device=device,
                policy_seed_base=policy_seed_base if crn else policy_seed_base + start,
                chunk_size=0,
                profiler=profiler,
                amp_dtype=amp_dtype,
                crn=crn,
                log_every=log_every,
                instances=instances,
            )
            for key in infos:
                infos[key].extend(inf[key])
        return infos

    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise NotImplementedError(
            f"population eval needs a cpu/cuda generator-backed RNG, got {device.type}")
    shared_models, _ = build_population_models(
        cfg, vcfg, device, K, init_seed_base=policy_seed_base, crn=crn,
    )
    shared_opts = [
        shared_optimizer(model, cfg.training) for model in shared_models
    ]
    generators = [
        torch.Generator(device=device).manual_seed(
            policy_seed_base if crn else policy_seed_base + k
        )
        for k in range(K)
    ]

    gamma = float(cfg.training.gamma)
    gae_lambda = float(cfg.training.gae_lambda)
    targs = PPOHparams.from_config(
        cfg,
        online_epochs=inner_epochs,
        ppo_diagnostics=False,
        amp_dtype=amp_dtype,
    )

    hist = {key: [[] for _ in range(K)] for key in spec.metric_keys}
    primary = spec.primary_metric

    for epoch in range(inner_epochs):
        for a in shared_models:
            a.train()
        step_seed = seed + 2000 + epoch
        env_seeds = [seed + 2000 + epoch * base_batch_size + i for i in range(base_batch_size)]
        base_instance = next(instances) if instances is not None else None
        with _prof_section(profiler, "pop_rollout"):
            staged_list, metrics = rollout_population(
                vcfg, spec, base_batch_size=base_batch_size, thetas=thetas,
                models=shared_models, action_generator=generators[0],
                action_generators=generators if crn else None, seeds=env_seeds,
                device=device, gamma=gamma, gae_lambda=gae_lambda, step_seed=step_seed,
                profiler=profiler, amp_dtype=amp_dtype, base_instance=base_instance,
            )
        for k in range(K):
            with _prof_section(profiler, "pop_ppo_prep"):
                staged = staged_list[k]
                staged["advantages"] = normalize_advantages_local(
                    staged["advantages"])
                rmean, rstd, rcount = local_mean_std_tensor(
                    staged["returns"])
                if rcount > 0:
                    shared_models[k].update_value_stats(rmean, rstd)

        for k in range(K):
            with _prof_section(profiler, "pop_ppo_shared_candidate"):
                ppo_update_staged(
                    shared_models[k], shared_opts[k], staged_list[k],
                    targs, device, epoch,
                    spec.policy.compute_log_probs_from_staged,
                    profiler=profiler,
                    generator=generators[k],
                    prebuild_graph=True,
                    return_stats=False,
                    profile_prefix="shared_",
                )
        for k in range(K):
            for key in hist:
                hist[key][k].append(float(metrics[key][k]))

        if log_every and (epoch % log_every == 0 or epoch == inner_epochs - 1):
            now = [hist[primary][k][-1] for k in range(K)]
            mean_now = sum(now) / K
            print(f"    epoch {epoch + 1:3d}/{inner_epochs} "
                  f"mean_{primary}={mean_now:.4f} min={min(now):.4f} max={max(now):.4f}",
                  flush=True)

    def tail(h):
        return float(sum(h[-tail_epochs:]) / max(1, len(h[-tail_epochs:])))

    info = {key: [tail(hist[key][k]) for k in range(K)] for key in spec.metric_keys}
    return info
