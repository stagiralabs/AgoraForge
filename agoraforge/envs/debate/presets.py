"""Debate presets and the run-config -> DebateConfig bridge."""

from __future__ import annotations

from ml_collections import ConfigDict

from agoraforge.conf.schema import resolve_level_group, validate_level_keys
from agoraforge.envs.debate.config import DebateConfig

GROUPS = ("env", "graph")
DEFAULT_LEVEL = "small_vanilla"


# ── Environment presets ─────────────────────────────────────────────────────

def small_vanilla():
    """N=32 claims, 2 debaters, alternating reveals, zero-sum judge-verdict reward.

    Eight turns plus the target keep the full-transcript exact judge at 2^9
    assignments. The transcript itself is bounded by the horizon, not a judge cap.
    """
    return ConfigDict({
        'num_claims': 32,
        'n_agents': 2,
        'max_timestep': 8,
        'judge_claim_cap': 9,
        'control_mode': 'vanilla',
        'action_dim': 2,
    })


def small_learned():
    """Protocol-bound F,w,P game on the small vanilla factor-graph world.

    `learned_mechanism_params` (a path) is set per inner run by the outer search,
    '' meaning a fresh no-op mechanism. The transcript may contain nine claims;
    the learned direction w chooses at most five for the terminal judge call.
    """
    cfg = small_vanilla()
    cfg.control_mode = 'learned'
    cfg.judge_claim_cap = 5
    cfg.learned_state_dim = 3
    cfg.learned_mechanism_hidden = 16
    cfg.learned_predictor_hidden = 4
    cfg.learned_mechanism_layernorm = False
    cfg.learned_mechanism_params = ''
    return cfg


# ── Instance-distribution presets ───────────────────────────────────────────
# The distribution over judge factor graphs is the load-bearing modelling choice.
# Sampled-truth probes established the qualitative requirement that unaries
# dominate couplings; otherwise induced-subgraph truncation overwhelms revealed
# evidence. Their quantitative anchors do not transfer to the corrected
# full-marginal objective, so re-establish them before trusting protocol
# comparisons in this preset.

def scale_free():
    return ConfigDict({
        'claim_graph_m': 2.0,        # Barabási–Albert attachment count
        'coupling_min': 0.2,         # |J| ~ U[min, max]
        'coupling_max': 1.0,
        'p_attack': 0.3,             # P(edge is an attack: J < 0)
        'unary_cutoff': 3.0,         # b_v ~ U[-c, c]: prior evidence per claim
        'truth_gibbs_sweeps': 20,    # burn-in before the truth sample
        'marginal_gibbs_sweeps': 20, # sweeps averaged into the p_full estimate
    })


# ── Action-decoding presets ─────────────────────────────────────────────────
# The per-claim signal head is a diagonal Gaussian; ``signal_init_std`` sets the
# initial std of that Gaussian (the head's log-std base), which strongly shapes
# early exploration of the protocol-facing signal channels.

def decoding_standard():
    return ConfigDict({
        'signal_init_std': 0.5,
        # Clamp bounds for the Gaussian head's log-std: std stays within
        # [exp(min), exp(max)] ~ [0.0067, 7.39].
        'log_std_min': -5.0,
        'log_std_max': 2.0,
    })


def base_groups() -> dict:
    return {
        "env": small_vanilla(),
        "graph": scale_free(),
        "decoding": decoding_standard(),
    }


def build_env_config(cfg, *, level) -> DebateConfig:
    """Build a DebateConfig for one curriculum level."""
    validate_level_keys(level, GROUPS)
    e = resolve_level_group(cfg, level, "env")
    g = resolve_level_group(cfg, level, "graph")
    return DebateConfig(
        num_claims=int(e.num_claims),
        n_agents=int(e.n_agents),
        max_timestep=int(e.max_timestep),
        judge_claim_cap=int(e.judge_claim_cap),
        claim_graph_m=float(g.claim_graph_m),
        coupling_min=float(g.coupling_min),
        coupling_max=float(g.coupling_max),
        p_attack=float(g.p_attack),
        unary_cutoff=float(g.unary_cutoff),
        truth_gibbs_sweeps=int(g.truth_gibbs_sweeps),
        marginal_gibbs_sweeps=int(g.marginal_gibbs_sweeps),
        control_mode=str(e.control_mode),
        action_dim=int(e.get("action_dim", 2)),
        learned_state_dim=int(e.get("learned_state_dim", 3)),
        learned_mechanism_hidden=int(e.get("learned_mechanism_hidden", 16)),
        learned_predictor_hidden=int(e.get("learned_predictor_hidden", 4)),
        learned_mechanism_layernorm=bool(e.get("learned_mechanism_layernorm", False)),
        learned_mechanism_params=(
            None if e.get("learned_mechanism_params", "") in ("", None)
            else str(e.get("learned_mechanism_params"))
        ),
    )
