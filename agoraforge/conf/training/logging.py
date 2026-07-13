from ml_collections import ConfigDict


def default():
    return ConfigDict({
        'wandb_enabled': False,
        'wandb_mode': 'online',          # 'online' | 'offline' | 'disabled'
        'tee_tensorboard': True,         # also write local TensorBoard files
        'print_profile_breakdown': False,
        'log_profile_scalars': True,
        'ppo_diagnostics': False,
        'profile_sync': False,           # drain the device queue per section for truthful timing (adds overhead)
    })
