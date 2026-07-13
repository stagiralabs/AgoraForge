"""Math presets and the run-config -> MathConfig bridge."""

from __future__ import annotations

from ml_collections import ConfigDict

from agoraforge.conf.schema import resolve_level_group, validate_level_keys
from agoraforge.envs.math.config import MathConfig

GROUPS = ("env", "graph", "kernels", "query")
DEFAULT_LEVEL = "small_baseline"
# Math runs pin the minibatch size their reported numbers were produced under
# (the shared default is tuned for debate).
TRAINING_OVERRIDES = {"online_batch_size": 3993}


# ── Environment presets ─────────────────────────────────────────────────────

def small_baseline():
    return ConfigDict({
        'num_theorems': 12,
        'n_agents': 2,
        'max_timestep': 50,
        'initial_cash': 1.0,
        'negative_return_penalty': 1.0,
        'prob_initially_resolved': 0.0834,
        'prob_initially_target': 0.0834,
        'bounty_demand': 10.0,
        'first_prover_bonus': 0.0,
        'prover_bonus_targets_only': False,
        'control_mode': 'market',
        # Reporting-only resolution-time discount scale.
        'fitness_tau': 10.0,
    })


def small_centralized():
    """small_baseline world, planner control (no markets, team reward)."""
    cfg = small_baseline()
    cfg.control_mode = 'centralized'
    return cfg


def small_learned():
    """Tiny world (4 theorems) under the learned market mechanism.

    Small on purpose: the outer loop searches over the mechanism parameters and
    needs many fast inner evaluations. Agents submit continuous per-formula
    signals and publish proofs; state updates and rewards come from the learned
    mechanism (`negative_return_penalty` below is unused in this mode).
    `learned_mechanism_params` (a path) is set per inner run by the outer search,
    '' meaning a freshly initialized zero-output mechanism.
    """
    return ConfigDict({
        'num_theorems': 4,
        'n_agents': 2,
        'max_timestep': 30,
        'initial_cash': 1.0,
        'negative_return_penalty': 0.0,
        'prob_initially_resolved': 0.25,
        'prob_initially_target': 0.25,
        'bounty_demand': 0.0,
        'first_prover_bonus': 0.0,
        'prover_bonus_targets_only': False,
        'control_mode': 'learned',
        'action_dim': 2,
        'obs_formula_cap': 0,
        'learned_state_dim': 3,
        'learned_mechanism_hidden': 16,
        'learned_mechanism_layernorm': False,
        'learned_mechanism_params': '',
        # Reporting-only resolution-time discount scale for this short horizon.
        'fitness_tau': 6.0,
    })


# ── Formula-graph (instance distribution) presets ───────────────────────────
# The formula graph is generated in two stages: a
# Barabási–Albert theorem graph defines a factor-graph distribution, from which
# a concrete formula graph (truths + discretized edge weights) is sampled.

def scale_free():
    return ConfigDict({
        'theorem_graph_m': 1.5,  # Barabási–Albert attachment count (expected edges each new theorem brings; may be fractional)
        'n_weight_levels': 9,    # number of discrete weight levels in [weight_min, weight_max]
        'weight_min': 0.0,       # lowest weight level (0 ⇒ absent edge)
        'weight_max': 1.0,       # highest weight level
        'weight_penalty': 4.0,   # strength λ of the "high weight ∧ src true ∧ dst false" penalty
        'truth_gibbs_sweeps': 5,  # Gibbs sweeps for sampling theorem truths
        # Random unary factors: per-truth bias_t * T_t and per-weight bias_W * w, with the
        # biases drawn Uniform[-cutoff, +cutoff]. 0 ⇒ that family is disabled.
        'truth_unary_cutoff': 2.0,   # log-odds spread of the per-theorem truth bias
        'weight_unary_cutoff': 2.0,  # tilt spread (at w=1) of the per-weight bias
        # Deterministic age-direction tilt on weight unaries. Positive values encourage
        # edges from earlier theorem ids to later theorem ids and discourage reverse edges.
        # 0 ⇒ disabled, preserving the symmetric random-unary model.
        'age_directional_weight_bias': 5.0,
    })


# ── Proof/conjecture kernel presets ─────────────────────────────────────────

def kernels_default():
    return ConfigDict({
        'rho': 0.4,  # Proof kernel
        'eta': 2.0,  # Conjecture kernel
    })


# ── Query presets ───────────────────────────────────────────────────────────

def noisy_graph():
    return ConfigDict({
        'horizon_H': 1000,
        'truth_prior_correct': 0.8,
        'weight_noise': 0.1,
        'num_related': 5,
    })


# ── Action-decoding presets ─────────────────────────────────────────────────
# The budget head samples a latent z ~ N(mu, sigma) and the env maps it to an
# integer budget via max(1, round(exp(z))), so the budget is log-normal on a
# continuous scale. The centering and spread are init offsets in the policy head --
# the mean starts at log(``budget_init``) and the log-std at log(``budget_init_std``)
# -- and the KL base policy is that same init distribution, so the prior pulls the
# budget back toward ``budget_init``.
#
# The demand-delta head is likewise a Gaussian latent z mapped linearly to the
# demand change; ``demand_init_std`` sets the initial std of that Gaussian, which
# strongly shapes early exploration.

def decoding_standard():
    return ConfigDict({
        'budget_init': 9.36964023613551,
        'budget_init_std': 0.2823763678533231,
        'demand_init_std': 0.4216072441824628,
        # Overflow guard on the budget latent exponent, far above useful budgets.
        'budget_latent_cap': 20.0,
        # Clamp bounds for the Gaussian heads' log-std: std stays within
        # [exp(min), exp(max)] ~ [0.0067, 7.39].
        'log_std_min': -5.0,
        'log_std_max': 2.0,
    })


def base_groups() -> dict:
    return {
        "env": small_baseline(),
        "graph": scale_free(),
        "kernels": kernels_default(),
        "query": noisy_graph(),
        "decoding": decoding_standard(),
    }


def build_env_config(cfg, *, level) -> MathConfig:
    """Build a MathConfig for one curriculum level."""
    validate_level_keys(level, GROUPS)
    e = resolve_level_group(cfg, level, "env")
    g = resolve_level_group(cfg, level, "graph")
    k = resolve_level_group(cfg, level, "kernels")
    q = resolve_level_group(cfg, level, "query")
    return MathConfig(
        num_theorems=int(e.num_theorems),
        n_agents=int(e.n_agents),
        theorem_graph_m=float(g.theorem_graph_m),
        n_weight_levels=int(g.n_weight_levels),
        weight_min=float(g.weight_min),
        weight_max=float(g.weight_max),
        weight_penalty=float(g.weight_penalty),
        truth_gibbs_sweeps=int(g.truth_gibbs_sweeps),
        truth_unary_cutoff=float(g.truth_unary_cutoff),
        weight_unary_cutoff=float(g.weight_unary_cutoff),
        age_directional_weight_bias=float(g.get("age_directional_weight_bias", 0.0)),
        max_timestep=e.max_timestep,
        initial_cash=float(e.initial_cash),
        negative_return_penalty=float(e.negative_return_penalty),
        prob_initially_resolved=float(e.prob_initially_resolved),
        prob_initially_target=float(e.prob_initially_target),
        rho=float(k.rho),
        eta=float(k.eta),
        horizon_H=q.horizon_H,
        query_truth_prior_correct=float(q.truth_prior_correct),
        query_weight_noise=float(q.weight_noise),
        query_num_related=int(q.num_related),
        bounty_demand=float(e.bounty_demand),
        first_prover_bonus=float(e.first_prover_bonus),
        prover_bonus_targets_only=bool(e.prover_bonus_targets_only),
        control_mode=str(e.control_mode),
        action_dim=int(e.get("action_dim", 2)),
        obs_formula_cap=int(e.get("obs_formula_cap", 0)),
        budget_latent_cap=float(cfg.decoding.budget_latent_cap),
        learned_state_dim=int(e.get("learned_state_dim", 3)),
        learned_mechanism_hidden=int(e.get("learned_mechanism_hidden", 16)),
        learned_mechanism_layernorm=bool(e.get("learned_mechanism_layernorm", False)),
        learned_mechanism_params=(
            None if e.get("learned_mechanism_params", "") in ("", None)
            else str(e.get("learned_mechanism_params"))
        ),
        fitness_tau=float(e.get("fitness_tau", 10.0)),
    )
