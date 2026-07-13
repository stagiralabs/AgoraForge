"""Eight-GPU SHADE debate search with asynchronous exact targets.

Inner recipe follows the gold recipe that produced the math env's SHADE
cold-start result: cold policies, buffer 256, 3 PPO epochs,
lr floor 0.10, 200 inner epochs with a 50-epoch fitness tail, and CRN on.
Objective = tail-mean accuracy against the full-graph target marginal.
"""

from agoraforge.conf.search import defaults as search_defaults
from agoraforge.envs.debate import presets as debate_presets
from agoraforge.conf.schema import run_config


def base_config():
    """Build the evaluator-neutral recipe shared with the CPU smoke config."""
    cfg = run_config()
    cfg.env = debate_presets.small_learned()
    cfg.levels = [{"name": "small_learned"}]
    cfg.schedule = [{"epoch": 0, "weights": {"small_learned": 1.0}}]

    cfg.training.online_buffer_size = 256
    cfg.training.online_ppo_epochs = 3
    cfg.training.lr_floor_frac = 0.10

    cfg.search = search_defaults.default()
    cfg.search.population = 18          # SHADE elites; ask() returns 2N trials+parents
    cfg.search.sigma = 0.02
    cfg.search.generations = 1_000_000  # explicit upper bound for long searches
    cfg.search.inner_epochs = 200
    cfg.search.tail_epochs = 50
    cfg.search.crn = True
    cfg.search.init_perturb = 0.0       # cold start exactly at the no-op mechanism
    # SHADE lessons from the math env's search: archive > N, run long past plateaus.
    cfg.search.archive_factor = 1.5
    cfg.search.wandb_tags = ["protocol-search", "shade", "debate"]
    return cfg


def get_config():
    cfg = base_config()
    cfg.device = "cuda"
    cfg.search.gpus = "0,1,2,3,4,5,6,7"
    cfg.search.batched_chunk = 5
    cfg.search.wandb_enabled = True
    cfg.search.wandb_mode = "online"
    cfg.search.wandb_tags.append("exact-target")
    # Injected exact instances bypass Gibbs entirely. Four single-threaded producers
    # per GPU worker stayed ahead of the rollout in the sharded pipeline smoke.
    cfg.search.exact_stream_workers = 4
    # 2N parents/trials + 2 probes = 40: fill all 8 GPUs x K=5 exactly.
    cfg.search.population = 19
    return cfg
