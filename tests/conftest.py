from __future__ import annotations

import pytest

from agoraforge.envs.debate.config import DebateConfig


def small_config(**overrides) -> DebateConfig:
    base = dict(
        num_claims=12,
        n_agents=2,
        max_timestep=6,
        judge_claim_cap=5,
        claim_graph_m=2.0,
        coupling_min=0.2,
        coupling_max=1.5,
        p_attack=0.3,
        unary_cutoff=1.0,
        truth_gibbs_sweeps=5,
        marginal_gibbs_sweeps=5,
        control_mode='vanilla',
    )
    base.update(overrides)
    return DebateConfig(**base)


@pytest.fixture
def debate_cfg():
    return small_config()
