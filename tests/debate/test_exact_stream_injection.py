"""Precomputed exact instances enter population batches without candidate drift."""

import torch

from agoraforge.conf.schema import build_env_config, run_config
from agoraforge.envs.debate.presets import small_learned
from agoraforge.envs.debate.env import BatchedDebateEnv
from agoraforge.envs.debate.protocol import theta_from_config


def test_precomputed_instance_repeats_identically_over_candidates():
    cfg = run_config()
    cfg.env = small_learned()
    vcfg = build_env_config(cfg, level={"name": "small_learned"})
    batch, candidates, nodes = 3, 2, vcfg.num_claims
    setup = torch.Generator().manual_seed(5)
    raw = torch.randn(batch, nodes, nodes, generator=setup)
    couplings = torch.triu(raw, diagonal=1)
    couplings = couplings + couplings.transpose(1, 2)
    instance = {
        "couplings": couplings,
        "unary": torch.randn(batch, nodes, generator=setup),
        "truth": torch.randint(0, 2, (batch, nodes), generator=setup),
        "p_full": torch.rand(batch, nodes, generator=setup),
        "target_idx": torch.arange(batch) % nodes,
    }
    theta = theta_from_config(vcfg)
    env = BatchedDebateEnv.batched_from_config(
        vcfg, base_batch_size=batch, thetas=[theta] * candidates,
        device=torch.device("cpu"), instance=instance,
    )
    for name in instance:
        actual = getattr(env, name)
        actual = actual.reshape(candidates, batch, *actual.shape[1:])
        assert torch.equal(actual[0], actual[1])
