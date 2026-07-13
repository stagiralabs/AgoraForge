"""Batched K-candidate env: block parity with a solo run under identical actions."""

import torch

from agoraforge.envs.debate.env import BatchedDebateEnv, TensorActions
from agoraforge.envs.debate.protocol import (
    BatchedDebateMechanism,
    DebateMechanism,
    mechanism_dims_from_config,
)
from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX
from tests.conftest import small_config

REVEAL = REVEAL_TYPE_TO_IDX['reveal']
PASS = REVEAL_TYPE_TO_IDX['pass']


def _thetas(dims, K, scale=0.05):
    torch.manual_seed(3)
    out = []
    for _ in range(K):
        op = DebateMechanism(dims)
        theta = op.flat_params()
        out.append(theta + scale * torch.randn_like(theta))
    return out


def _actions(B, A, N, action_dim, step, gen):
    actions = TensorActions(
        reveal_type=torch.full((B, A), PASS, dtype=torch.long),
        reveal_claim=torch.full((B, A), -1, dtype=torch.long),
        signals=torch.randn((B, A, N, action_dim), generator=gen),
    )
    agent = step % A
    actions.reveal_type[:, agent] = REVEAL
    actions.reveal_claim[:, agent] = torch.randint(0, N, (B,), generator=gen)
    return actions


def test_batched_blocks_match_solo_runs():
    cfg = small_config(control_mode='learned', max_timestep=4, num_claims=10,
                       judge_claim_cap=3)
    dims = mechanism_dims_from_config(cfg)
    K, B_base = 3, 2
    thetas = _thetas(dims, K)
    seeds = (11, 12)

    batched = BatchedDebateEnv.batched_from_config(
        cfg, base_batch_size=B_base, thetas=thetas,
        device=torch.device('cpu'), seeds=seeds, step_seed=99,
    )
    # The same action tape must drive every run: draw once, replay per block.
    gen = torch.Generator().manual_seed(42)
    tapes = [_actions(B_base, cfg.n_agents, cfg.num_claims, cfg.action_dim, t, gen)
             for t in range(cfg.max_timestep)]
    batched_rewards = []
    for t in range(cfg.max_timestep):
        a = tapes[t]
        full = TensorActions(
            reveal_type=a.reveal_type.repeat(K, 1),
            reveal_claim=a.reveal_claim.repeat(K, 1),
            signals=a.signals.repeat(K, 1, 1, 1),
        )
        batched_rewards.append(batched.step(full).squeeze(-1))

    for k in range(K):
        cfg_k = small_config(control_mode='learned', max_timestep=4, num_claims=10,
                             judge_claim_cap=3)
        solo = BatchedDebateEnv.from_config(
            cfg_k, batch_size=B_base, device=torch.device('cpu'), seeds=seeds)
        solo.mechanism = BatchedDebateMechanism([thetas[k]], dims)
        solo.dims = solo.mechanism.dims
        gen_device = torch.device('cpu')
        solo._step_gen = torch.Generator(device=gen_device).manual_seed(99)
        sl = slice(k * B_base, (k + 1) * B_base)
        assert torch.equal(solo.couplings, batched.couplings[sl])
        assert torch.equal(solo.target_idx, batched.target_idx[sl])
        for t in range(cfg.max_timestep):
            r = solo.step(tapes[t]).squeeze(-1)
            assert torch.allclose(r, batched_rewards[t][sl], atol=1e-5), (k, t)
        assert torch.allclose(solo.final_p, batched.final_p[sl], atol=1e-6)
        assert torch.allclose(solo.s, batched.s[sl], atol=1e-5)
