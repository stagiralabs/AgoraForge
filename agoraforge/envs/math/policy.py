"""Math action sampling and log-prob scoring (the EnvSpec policy contract)."""

from __future__ import annotations

import math

import torch

from agoraforge.envs.distributions import (
    as_1d,
    as_2d_logits,
    as_3d_logits,
    cat_entropy,
    cat_log_prob,
    cat_log_softmax,
    categorical_uniform_kl,
    crn_normal,
    gaussian_kl_to_base,
    gumbel_sample,
    normal_log_prob,
)
from agoraforge.envs.math.env import TensorActions
from agoraforge.envs.math.actions import MATH_ACTION_TYPES

def sample_actions(
    logits: dict,
    masks: dict,
    batch,
    generator: torch.Generator | None = None,
    crn_blocks: int = 1,
) -> tuple[TensorActions, dict]:
    """Sample actions as tensors and return PPO-staged action fields.

    Per-formula fields (``math_formula``, ``publish_*``, ``market_action``) index
    the OBSERVED formula axis, whose width ``F`` equals the obs cap ``K`` when the
    cap is on and ``batch.F`` otherwise. The caller scatters them back to absolute
    formula ids before stepping the env.

    ``generator`` threads a per-candidate policy RNG stream (gumbel uniforms +
    Gaussian noise) for the population path so candidate ``k``'s sampling is
    independent of the others' and reproduces its single-candidate run; ``None``
    keeps the global-RNG draws (production single-policy rollout, unchanged).
    """
    B, A = batch.B, batch.A
    F = masks["prove"].shape[-1]
    BA = B * A
    device = batch.device
    log_probs = torch.zeros(BA, dtype=torch.float32, device=device)

    type_logits = as_2d_logits(logits["math_type_logits"])
    type_masks = masks["math_type_mask"].reshape(BA, len(MATH_ACTION_TYPES))
    active_math = type_masks.any(dim=-1)
    masked_type = type_logits.masked_fill(type_masks == 0, -1e8)
    type_idx_raw = gumbel_sample(masked_type, generator, crn_blocks)
    type_lp = cat_log_prob(cat_log_softmax(masked_type), type_idx_raw)
    type_idx = torch.where(
        active_math,
        type_idx_raw,
        torch.full((BA,), -1, dtype=torch.long, device=device),
    )
    log_probs = log_probs + type_lp * active_math.to(log_probs.dtype)

    formula_choice = torch.full((BA,), -1, dtype=torch.long, device=device)
    formula_masks = {}
    per_type = logits.get("formula_logits_per_type", {})
    fallback = logits.get("formula_logits")
    if fallback is not None:
        fallback = as_2d_logits(fallback)
    for type_i, name in enumerate(MATH_ACTION_TYPES):
        if name not in ("prove", "conj", "query_prob_time", "query_related"):
            continue
        fl = as_2d_logits(per_type[name]) if name in per_type else fallback
        fm = masks[name].reshape(BA, F)
        formula_masks[name] = fm
        sampled = gumbel_sample(
            fl.masked_fill(fm == 0, -1e8), generator, crn_blocks,
        )
        active = type_idx == type_i
        lp = cat_log_prob(cat_log_softmax(fl.masked_fill(fm == 0, -1e8)), sampled)
        formula_choice = torch.where(active, sampled, formula_choice)
        log_probs = log_probs + lp * active.to(log_probs.dtype)

    prove_idx = MATH_ACTION_TYPES.index("prove")
    conj_idx = MATH_ACTION_TYPES.index("conj")
    qrel_idx = MATH_ACTION_TYPES.index("query_related")
    active_budget = (type_idx == prove_idx) | (type_idx == conj_idx)
    mu = logits["budget_mu"].reshape(BA)
    std = logits["budget_log_std"].reshape(BA).exp()
    budget_raw = crn_normal(mu, std, generator=generator, crn_blocks=crn_blocks)
    budget = torch.where(active_budget, budget_raw, torch.full_like(budget_raw, -1.0))
    log_probs = log_probs + normal_log_prob(budget_raw, mu, std) * active_budget.to(log_probs.dtype)

    mode_logits = as_2d_logits(logits["mode_logits"])
    active_mode = (type_idx == conj_idx) | (type_idx == qrel_idx)
    mode_raw = gumbel_sample(mode_logits, generator, crn_blocks)
    mode_lp = cat_log_prob(cat_log_softmax(mode_logits), mode_raw)
    mode = torch.where(active_mode, mode_raw, torch.full_like(mode_raw, -1))
    log_probs = log_probs + mode_lp * active_mode.to(log_probs.dtype)

    def binary(logits_name: str, mask_name: str):
        nonlocal log_probs
        raw = as_3d_logits(logits[logits_name])
        mask = masks[mask_name].reshape(BA, F)
        masked = raw.masked_fill(mask.unsqueeze(-1) == 0, -1e8).clone()
        masked[..., 0] = 0.0
        sampled = torch.where(
            mask > 0,
            gumbel_sample(masked, generator, crn_blocks),
            torch.zeros((BA, F), dtype=torch.long, device=device),
        )
        log_probs = log_probs + cat_log_prob(cat_log_softmax(masked), sampled).sum(dim=1)
        return sampled, mask

    publish_statement, publish_statement_mask = binary(
        "publish_statement_logits", "publish_statement_mask")
    publish_proof, publish_proof_mask = binary(
        "publish_proof_logits", "publish_proof_mask")

    demand_mu = logits["market_action_mu"]
    demand_std = logits["market_action_log_std"].exp()
    market_action_mask = masks["market_action_mask"].reshape(BA, F)
    market_action = crn_normal(
        demand_mu, demand_std, generator=generator, crn_blocks=crn_blocks,
    ) * market_action_mask.unsqueeze(-1)
    log_probs = log_probs + (
        normal_log_prob(market_action, demand_mu, demand_std).sum(dim=-1) * market_action_mask
    ).sum(dim=1)

    actions = TensorActions(
        math_type=type_idx.reshape(B, A),
        math_formula=formula_choice.reshape(B, A),
        budget=budget.reshape(B, A),
        math_mode=mode.clamp(min=0).reshape(B, A),
        publish_statement=publish_statement.reshape(B, A, F).bool(),
        publish_proof=publish_proof.reshape(B, A, F).bool(),
        market_action=market_action.reshape(B, A, F, batch.cfg.action_dim),
    )  # Per-formula fields are at obs width F (= K when capped).
    staged = {
        "old_log_probs": log_probs.detach(),
        "type_indices": type_idx.detach(),
        "type_masks": type_masks.detach(),
        "math_formula": formula_choice.detach(),
        "formula_masks": {k: v.detach() for k, v in formula_masks.items()},
        "math_budget": budget.detach(),
        "math_mode": mode.detach(),
        "publish_statement": publish_statement.detach(),
        "publish_statement_mask": publish_statement_mask.detach(),
        "publish_proof": publish_proof.detach(),
        "publish_proof_mask": publish_proof_mask.detach(),
        "market_action_mask": market_action_mask.detach(),
        "market_action": market_action.detach(),
    }
    return actions, staged


def to_env_actions(actions: TensorActions, ctx, batch) -> TensorActions:
    """Map K-wide (obs-axis) per-formula actions back onto absolute formula ids.

    ``sample_actions`` produces per-formula fields at the observed cap width K;
    the env steps over the full F universe, so the selected slots are scattered
    back to their absolute ids (unselected ids stay no-op). ``ctx`` is the
    top-k selection from ``policy_inputs()``; None (uncapped or centralized) means the
    sampled actions are already env-shaped.
    """
    if ctx is None:
        return actions
    sel_idx = ctx
    B, A, F = batch.B, batch.A, batch.F
    # math_formula indexes the K obs axis; map the chosen slot to its absolute id.
    chosen = actions.math_formula.clamp(min=0).unsqueeze(-1)
    abs_formula = torch.gather(sel_idx, 2, chosen).squeeze(-1)
    math_formula = torch.where(actions.math_formula >= 0, abs_formula, actions.math_formula)

    def scatter(src: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((B, A, F, *src.shape[3:]), dtype=src.dtype, device=src.device)
        idx = sel_idx.reshape(B, A, sel_idx.shape[-1], *([1] * (src.dim() - 3)))
        idx = idx.expand(B, A, sel_idx.shape[-1], *src.shape[3:])
        return out.scatter(2, idx, src)

    return TensorActions(
        math_type=actions.math_type,
        math_formula=math_formula,
        budget=actions.budget,
        math_mode=actions.math_mode,
        publish_statement=scatter(actions.publish_statement),
        publish_proof=scatter(actions.publish_proof),
        market_action=scatter(actions.market_action),
    )


def compute_log_probs_from_staged(logits_dict, s):
    """Log-probs and base-policy KL from pre-staged minibatch tensors.

    ``kls`` is the per-head-normalized KL of the policy from a base policy --
    uniform for the discrete heads, N(0,1) for the Gaussian demand heads -- with
    the formula head divided by log F and the per-formula heads averaged over
    their active formulas, so neither the action cardinality nor the formula
    count leaks into its scale.
    """
    type_indices = s["type_indices"]
    B = type_indices.shape[0]
    device = type_indices.device
    log_probs = torch.zeros(B, device=device)
    kls = torch.zeros(B, device=device)

    # Full-batch (no .any() guards or boolean-index subselects, which sync the host):
    # every head scores all B rows and zeroes out the inactive ones with a multiply.
    # Inactive rows carry a sentinel index, so gathers clamp to 0 and the zeroed
    # contribution makes this identical to the per-active-row path, gradients and all.
    type_logits = as_2d_logits(logits_dict["math_type_logits"])
    type_masks = s["type_masks"]
    active_math = (type_indices >= 0).to(log_probs.dtype)
    type_lp = cat_log_softmax(type_logits.masked_fill(type_masks == 0, -1e8))
    log_probs = log_probs + cat_log_prob(type_lp, type_indices.clamp(min=0)) * active_math
    kls = kls + categorical_uniform_kl(cat_entropy(type_lp), type_masks.sum(dim=1)) * active_math

    formula_logits = logits_dict.get("formula_logits")
    per_type_logits = logits_dict.get("formula_logits_per_type", {})
    if formula_logits is not None:
        formula_logits = as_2d_logits(formula_logits)
    math_formula = s["math_formula"]
    formula_idx = math_formula.clamp(min=0)
    for type_idx, type_name in enumerate(MATH_ACTION_TYPES):
        active = ((type_indices == type_idx) & (math_formula >= 0)).to(log_probs.dtype)
        fl = as_2d_logits(per_type_logits[type_name]) if type_name in per_type_logits else formula_logits
        fm_t = s["formula_masks"][type_name]
        f_lp = cat_log_softmax(fl.masked_fill(fm_t == 0, -1e8))
        log_probs = log_probs + cat_log_prob(f_lp, formula_idx) * active
        n_valid_f = fm_t.sum(dim=1)
        log_f = torch.log(n_valid_f.clamp(min=2.0))
        kls = kls + categorical_uniform_kl(cat_entropy(f_lp), n_valid_f) / log_f * active

    prove_idx = MATH_ACTION_TYPES.index("prove")
    conj_idx = MATH_ACTION_TYPES.index("conj")
    active_budget = ((type_indices == prove_idx) | (type_indices == conj_idx)).to(log_probs.dtype)
    mu = as_1d(logits_dict["budget_mu"])
    budget_log_std = as_1d(logits_dict["budget_log_std"])
    log_probs = log_probs + normal_log_prob(
        s["math_budget"], mu, budget_log_std.exp()) * active_budget
    kls = kls + gaussian_kl_to_base(
        mu, budget_log_std,
        float(logits_dict["budget_mu_base"]), float(logits_dict["budget_log_std_base"])) * active_budget

    mode_logits = as_2d_logits(logits_dict["mode_logits"])
    mode_indices = s["math_mode"]
    query_related_idx = MATH_ACTION_TYPES.index("query_related")
    active_mode = (((type_indices == conj_idx) | (type_indices == query_related_idx))
                  & (mode_indices >= 0)).to(log_probs.dtype)
    mode_lp = cat_log_softmax(mode_logits)
    log_probs = log_probs + cat_log_prob(mode_lp, mode_indices.clamp(min=0)) * active_mode
    mode_ent = cat_entropy(mode_lp)
    n_modes = torch.full_like(mode_ent, float(mode_logits.shape[1]))
    kls = kls + categorical_uniform_kl(mode_ent, n_modes) * active_mode

    def _binary(action_t, logits_name, mask_t):
        nonlocal log_probs, kls
        logits = as_3d_logits(logits_dict[logits_name])
        logits = logits.masked_fill(mask_t.unsqueeze(-1) == 0, -1e8).clone()
        logits[:, :, 0] = 0.0
        lp_table = cat_log_softmax(logits)
        log_probs = log_probs + cat_log_prob(lp_table, action_t).sum(dim=1)
        per_formula_kl = (math.log(2.0) - cat_entropy(lp_table)).clamp(min=0.0)
        active_formulas = mask_t.sum(dim=1).clamp(min=1.0)
        kls = kls + (per_formula_kl * mask_t).sum(dim=1) / active_formulas

    _binary(s["publish_statement"], "publish_statement_logits", s["publish_statement_mask"])
    _binary(s["publish_proof"], "publish_proof_logits", s["publish_proof_mask"])

    def _demand(z_t, valid_t):
        nonlocal log_probs, kls
        mu = logits_dict["market_action_mu"]            # (B, F, A)
        log_std = logits_dict["market_action_log_std"]  # (B, F, A)
        std = log_std.exp()
        log_probs = log_probs + (
            normal_log_prob(z_t, mu, std).sum(dim=-1) * valid_t
        ).sum(dim=1)
        # KL(N(mu,std) || N(0,1)) summed over the action_dim, per formula.
        per_formula_kl = (
            -log_std + 0.5 * (std * std + mu * mu - 1.0)
        ).sum(dim=-1).clamp(min=0.0)
        active_formulas = valid_t.sum(dim=1).clamp(min=1.0)
        kls = kls + (per_formula_kl * valid_t).sum(dim=1) / active_formulas

    _demand(s["market_action"], s["market_action_mask"])

    return log_probs, kls
