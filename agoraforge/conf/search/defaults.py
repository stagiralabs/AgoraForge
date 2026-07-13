"""Defaults for outer mechanism search with fixed-population SHADE."""
from ml_collections import ConfigDict


def default() -> ConfigDict:
    cfg = ConfigDict()
    cfg.generations = 30
    cfg.population = 16          # persistent SHADE members
    cfg.sigma = 0.05             # initial population spread
    cfg.pbest_frac = 0.25        # elite fraction targeted by current-to-pbest/1
    cfg.memory_size = 6          # success-history length for F and CR
    cfg.archive_factor = 1.0     # displaced-parent archive capacity relative to N
    cfg.inner_epochs = 50         # cold-start candidate budget
    cfg.seed_base = 42
    cfg.tail_epochs = 10          # fitness/penalty averaged over the last N epochs
    cfg.seed = 0                  # SHADE sampling RNG seed
    cfg.init_seed = 0             # seed for the initial mechanism
    cfg.init_perturb = 0.1        # Gaussian noise on the fresh mechanism (off the F=0 corner)
    # Named probe slots appended to the eval batch and excluded from the update.
    # The zero_reward probe uses an unperturbed fresh mechanism whose zero final
    # layer makes the mechanism output exactly zero.
    cfg.probes = ['incumbent', 'zero_reward']
    # Common random numbers on the policy side: every candidate gets the same
    # policy-init seed and identically-seeded action/PPO RNG streams (env draws
    # are already tiled at the base batch). Within-generation fitness differences
    # then reflect theta, not the policy-seed lottery; theta x seed interaction
    # noise remains.
    cfg.crn = False
    # Batched generation eval: score the whole population in one folded env batch
    # with K stacked shared actor/critic policies. batched_chunk > 0 evaluates the
    # population in K-chunks of that size (memory ceiling); 0 = whole population.
    cfg.batched_chunk = 0
    # Optional asynchronous exact debate-instance producer per GPU worker.
    cfg.exact_stream_workers = 0
    # Model-forward autocast for batched population eval. Empty/none = FP32.
    # BF16 is a throughput probe only until objective/rank stability is checked.
    cfg.amp_dtype = ''
    cfg.threads_per_worker = 1    # CPU threads per inner run (low avoids oversubscription)
    cfg.gpus = ''                 # comma list of device ids, round-robined over workers ('' = CPU)
    cfg.wandb_enabled = False     # outer search logging; inner train.py may keep W&B off
    cfg.wandb_project = 'agoraforge'
    cfg.wandb_entity = ''
    cfg.wandb_mode = 'online'     # 'online' | 'offline' | 'disabled'
    cfg.wandb_name = ''
    cfg.wandb_tags = []
    return cfg
