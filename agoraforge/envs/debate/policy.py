"""Debate action sampling and log-prob scoring (the EnvSpec policy contract)."""

from __future__ import annotations

import torch

from agoraforge.envs.distributions import (
    as_2d_logits,
    cat_entropy,
    cat_log_prob,
    cat_log_softmax,
    categorical_uniform_kl,
    crn_normal,
    gumbel_sample,
    normal_log_prob,
)
from agoraforge.envs.debate.env import TensorActions
from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX

def sample_actions(
    logits: dict,
    masks: dict,
    batch,
    generator: torch.Generator | None = None,
    crn_blocks: int = 1,
) -> tuple[TensorActions, dict]:
    """Sample debate actions as tensors and return PPO-staged action fields.

    ``generator`` threads a per-candidate policy RNG stream (gumbel uniforms +
    Gaussian noise) for the population path so candidate ``k``'s sampling is
    independent of the others' and reproduces its single-candidate run; ``None``
    keeps the global-RNG draws (production single-policy rollout).
    """
    B, A = batch.B, batch.A
    N = masks["reveal_claim_mask"].shape[-1]
    BA = B * A
    device = masks["reveal_claim_mask"].device
    log_probs = torch.zeros(BA, dtype=torch.float32, device=device)

    type_logits = as_2d_logits(logits["reveal_type_logits"])
    type_masks = masks["reveal_type_mask"].reshape(BA, type_logits.shape[-1])
    masked_type = type_logits.masked_fill(type_masks == 0, -1e8)
    reveal_type = gumbel_sample(masked_type, generator, crn_blocks)
    log_probs = log_probs + cat_log_prob(cat_log_softmax(masked_type), reveal_type)

    reveal_idx = REVEAL_TYPE_TO_IDX["reveal"]
    claim_logits = as_2d_logits(logits["claim_logits"])
    claim_mask = masks["reveal_claim_mask"].reshape(BA, N)
    # Rows without any revealable claim never sample reveal (the type mask gates
    # them), but the pointer is still scored full-batch; give those rows a dummy
    # all-valid mask so the softmax is finite, then zero their contribution.
    has_claim = claim_mask.any(dim=-1)
    safe_mask = torch.where(has_claim.unsqueeze(-1), claim_mask, torch.ones_like(claim_mask))
    masked_claim = claim_logits.masked_fill(safe_mask == 0, -1e8)
    claim_raw = gumbel_sample(masked_claim, generator, crn_blocks)
    active_claim = (reveal_type == reveal_idx) & has_claim
    claim_lp = cat_log_prob(cat_log_softmax(masked_claim), claim_raw)
    reveal_claim = torch.where(
        active_claim, claim_raw, torch.full((BA,), -1, dtype=torch.long, device=device)
    )
    log_probs = log_probs + claim_lp * active_claim.to(log_probs.dtype)

    signal_mu = logits["signal_mu"]                 # (BA, N, d_action)
    signal_std = logits["signal_log_std"].exp()
    signal_mask = masks["signal_mask"].reshape(BA, N)
    # A debater may signal on the claim it adds this turn. The base mask covers the
    # pre-turn transcript; extend it after sampling the reveal pointer so PPO scores
    # the newly public claim's continuous action as well.
    if batch.cfg.control_mode == "learned":
        revealed_onehot = torch.nn.functional.one_hot(
            reveal_claim.clamp(min=0), N
        ).to(signal_mask.dtype)
        signal_mask = torch.maximum(
            signal_mask, revealed_onehot * active_claim.to(signal_mask.dtype).unsqueeze(1)
        )
    signals = crn_normal(
        signal_mu, signal_std, generator=generator, crn_blocks=crn_blocks,
    ) * signal_mask.unsqueeze(-1)
    log_probs = log_probs + (
        normal_log_prob(signals, signal_mu, signal_std).sum(dim=-1) * signal_mask
    ).sum(dim=1)

    actions = TensorActions(
        reveal_type=reveal_type.reshape(B, A),
        reveal_claim=reveal_claim.reshape(B, A),
        signals=signals.reshape(B, A, N, batch.cfg.action_dim),
    )
    staged = {
        "old_log_probs": log_probs.detach(),
        "reveal_type": reveal_type.detach(),
        "reveal_type_mask": type_masks.detach(),
        "reveal_claim": reveal_claim.detach(),
        "reveal_claim_mask": claim_mask.detach(),
        "signals": signals.detach(),
        "signal_mask": signal_mask.detach(),
    }
    return actions, staged


def to_env_actions(actions: TensorActions, ctx, batch) -> TensorActions:
    """Debate decodes over all claims: sampled actions are already env-shaped."""
    return actions


def compute_log_probs_from_staged(logits_dict, s):
    """Log-probs and base-policy KL from pre-staged minibatch tensors.

    ``kls`` is the per-head-normalized KL of the policy from a base policy --
    uniform for the discrete heads, N(0,1) for the Gaussian signal head -- with
    the claim-pointer head divided by log N and the per-claim signal KL averaged
    over active claims, so neither the action cardinality nor the claim count
    leaks into its scale.

    Full-batch (no .any() guards or boolean-index subselects, which sync the host):
    every head scores all B rows and zeroes out the inactive ones with a multiply.
    Inactive rows carry a sentinel index, so gathers clamp to 0 and the zeroed
    contribution makes this identical to the per-active-row path, gradients and all.
    """
    reveal_type = s["reveal_type"]
    B = reveal_type.shape[0]
    device = reveal_type.device
    log_probs = torch.zeros(B, device=device)
    kls = torch.zeros(B, device=device)

    type_logits = as_2d_logits(logits_dict["reveal_type_logits"])
    type_masks = s["reveal_type_mask"]
    type_lp = cat_log_softmax(type_logits.masked_fill(type_masks == 0, -1e8))
    log_probs = log_probs + cat_log_prob(type_lp, reveal_type)
    kls = kls + categorical_uniform_kl(cat_entropy(type_lp), type_masks.sum(dim=1))

    reveal_claim = s["reveal_claim"]
    claim_mask = s["reveal_claim_mask"]
    active_claim = (reveal_claim >= 0).to(log_probs.dtype)
    # Rows with no revealable claim keep a dummy all-valid mask so the softmax is
    # finite; their contribution is zeroed by active_claim (mirrors the sampler).
    has_claim = claim_mask.any(dim=-1)
    safe_mask = torch.where(has_claim.unsqueeze(-1), claim_mask, torch.ones_like(claim_mask))
    claim_logits = as_2d_logits(logits_dict["claim_logits"])
    claim_lp = cat_log_softmax(claim_logits.masked_fill(safe_mask == 0, -1e8))
    log_probs = log_probs + cat_log_prob(claim_lp, reveal_claim.clamp(min=0)) * active_claim
    n_valid = claim_mask.sum(dim=1)
    log_n = torch.log(n_valid.clamp(min=2.0))
    kls = kls + categorical_uniform_kl(cat_entropy(claim_lp), n_valid) / log_n * active_claim

    mu = logits_dict["signal_mu"]                # (B, N, action_dim)
    log_std = logits_dict["signal_log_std"]
    std = log_std.exp()
    valid = s["signal_mask"]
    log_probs = log_probs + (
        normal_log_prob(s["signals"], mu, std).sum(dim=-1) * valid
    ).sum(dim=1)
    # KL(N(mu,std) || N(0,1)) summed over the action_dim, averaged per active claim.
    per_claim_kl = (
        -log_std + 0.5 * (std * std + mu * mu - 1.0)
    ).sum(dim=-1).clamp(min=0.0)
    kls = kls + (per_claim_kl * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    return log_probs, kls
