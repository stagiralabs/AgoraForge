"""Training presets."""
from ml_collections import ConfigDict


def default():
    # Buffer 128 with epoch-denominated schedules scaled by 54/300; equal
    # total episodes to the 300-epoch x buffer-23 runs these hyperparameters
    # were tuned under.
    return ConfigDict({
        'online_epochs': 54,
        'checkpoint_every_epochs': 9,
        # 4 (not 7): an N=64 sweep found resolved_frac flat across buffer
        # {128,256,512} x ppo {4,7}, so 4 buys a ~1.4x speedup at no quality cost.
        'online_ppo_epochs': 4,
        'online_batch_size': 1280,
        'actor_lr': 0.001079008783895993,
        'critic_lr': 0.0029059310826069448,
        'lr_decay_epochs': 2.74,
        'lr_decay_power': 1.0975821313067056,
        'lr_floor_frac': 0.7934465642876182,
        'online_buffer_size': 128,
        'gamma': 0.99,
        'clip_eps': 0.3304200806388339,
        'starting_kl_coeff': 0.002140379812432022,
        'ending_kl_coeff': 5.443900954499275e-05,
        'kl_decay_epochs': 8.56,
        'kl_decay_power': 1.5212335368370584,
        'gae_lambda': 0.993548931607006,
        'grad_norm_clip': 0.929289425653399,
        'critic_huber_beta': 0.39582919755504314,
        # Opt-in bit-reproducibility: torch.use_deterministic_algorithms +
        # CUBLAS_WORKSPACE_CONFIG. Default off because the deterministic CUDA
        # kernels (e.g. for the sparse-attention index_add) are slower; turn on
        # when you need same-seed runs to match exactly.
        'deterministic': False,
    })
