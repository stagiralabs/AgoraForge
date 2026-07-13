"""Mechanism-search run config -- the base recipe for a SHADE search.

Cold-start, batched scheme. Defines BOTH the inner learned world (``cfg.env``,
``cfg.training``) and the outer SHADE config (``cfg.search``). Each
generation's whole population is scored in one in-process batched run: K
mechanisms folded into a single env batch with K stacked shared actor/critic
policies trained cold. The N=64 deploy regime (n_agents=2) is folded into
``cfg.env`` at build time.

Recipe: cold candidates, buffer 256, 200 epochs, tail-50, base LR. Cheaper
recipes lose candidate-ranking fidelity; warm-start is a biased estimator (it
masks late collapse).

    python -m agoraforge.search --config=agoraforge/conf/runs/math/mechanism_search.py --out <dir>
"""
from agoraforge.envs.math import presets as math_presets
from agoraforge.envs.math.scaling import scale_env_overrides
from agoraforge.conf.schema import run_config
from agoraforge.conf.search import defaults as search_defaults

NUM_THEOREMS = 64


def base_config(*, device: str, online_batch_size: int, wandb_enabled: bool):
    """Inner-world + ES setup shared by the full run and the smoke."""
    cfg = run_config(env="math", overrides={
        "env": math_presets.small_learned(),
        "levels": [{"name": "small_learned"}],
        "schedule": [{"epoch": 0, "weights": {"small_learned": 1.0}}],
        "device": device,
        "training.online_batch_size": online_batch_size,
    })
    # Smallest PPO-epoch count that preserves mechanism ranking (rank-fidelity probe:
    # Spearman 0.972, top-16 14/16); buys inner-eval throughput for the search.
    cfg.training.online_ppo_epochs = 3
    # Fast decay to a low LR floor so a fitness drop is the mechanism's doing, not an
    # LR-floor PPO collapse late in training.
    cfg.training.lr_floor_frac = 0.10
    cfg.logging.wandb_enabled = wandb_enabled
    cfg.search = search_defaults.default()
    cfg.search.wandb_tags = ["mechanism-search", "shade", "cold-start"]
    return cfg


def get_config():
    cfg = base_config(device='cuda', online_batch_size=16384, wandb_enabled=False)
    # N=64 floor: n_agents=2, sparse targets, fixed obs cap, fitness horizon.
    for key, value in scale_env_overrides(NUM_THEOREMS).items():
        cfg.env[key] = value
    cfg.training.online_buffer_size = 256       # per-candidate env count (gold recipe)

    s = cfg.search
    s.generations = 1_000_000                   # explicit upper bound for long searches
    s.population = 32
    s.sigma = 0.02                              # sigma0
    s.init_perturb = 0.0                        # start at the fresh F=0 mechanism
    # Cold-start gold fitness: base LR and convergence-aware horizon.
    s.inner_epochs = 200
    s.tail_epochs = 50
    s.batched_chunk = 0                         # single-GPU default; 8GPU config shards this
    s.gpus = '0'                                # batched eval is single-GPU, in-process
    s.wandb_enabled = True                      # outer search progress is the run
    s.wandb_mode = 'offline'                    # sync later with `wandb sync`
    return cfg
