"""Learned market mechanism: a pooled MLP over market states.

The math instance of the shared mechanism family (``agoraforge.envs.mechanism``).
The market is a set of per-``(formula, agent)`` learned states evolved by a
single small mechanism; the agent's action is just a vector into that mechanism.

State:

    s[phi, j]   in R^{d_state}   agent j's state in formula phi's market

There is no separate public market state: a public quantity (e.g. a clearing
price) is carried redundantly in every agent's state and stays synchronized
because the mechanism updates it from the shared aggregates only.

Each step the mechanism consumes the agents' action vectors ``a[phi, j]`` and the
per-agent environment update ``e[phi, j]`` (``[theorem_resolved_by_this_agent,
proven_true, theorem_resolved, is_last_step, is_target,
originally_made_public_by_this_agent]``; the private bits identify the current
agent's resolution event and persistent public-origin ownership, while the other
four are identical across a formula's agents) and advances the state residually
with sum-pooling messages:

    delta_s[phi, j], r[phi, j] = F( s[phi,j], a[phi,j], e[phi,j],
                                    sum_k s[phi,k], sum_k a[phi,k] )
    s'[phi, j] = (1-decay) * s[phi, j]
                 + bound * tanh(delta_s[phi, j] / bound)

``F`` sees the same aggregates a dedicated market update would, so it recomputes
any market quantity (the clearing price ``f(sum_k a)``) locally and settles in one
shot; there is no separate market-update pass.

Agent j's reward on a step is the sum of the rewards returned by ``F`` over its
formulas. The agent OBSERVES its own state ``s[phi, j]`` -- the observation codec
appends it in 'learned' mode. The
mechanism parameters ``theta`` are frozen during inner RL training and searched
over by an outer mechanism search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from agoraforge.envs.mechanism import (
    EPS,
    FlatParamsMixin,
    MechanismDims as CoreMechanismDims,
    bounded_residual,
    candidate_layernorm,
    candidate_linear,
    finite as _finite,
    first_layernorm,
    linears,
    scaled_input,
    stack_params,
    zero_final_mlp,
)


@dataclass(frozen=True)
class MarketMechanismDims(CoreMechanismDims):
    # Per-agent env update e = [theorem_resolved_by_this_agent, proven_true,
    # theorem_resolved, is_last_step, is_target,
    # originally_made_public_by_this_agent]. The first and last bits are private:
    # the first is a one-step resolution event, and the last stays set for the
    # agent that first introduced the theorem pair to the public library. The
    # other four are per-formula and broadcast identically across the formula's
    # agents.
    # `proven_true` separates a TRUE resolution from a FALSE one (a theorem
    # resolves in pairs; the negation's market gets `theorem_resolved` with
    # `proven_true=0`). `is_target` is a static bit (the principal's "I want this
    # proven" signal) set every step for markets whose theorem is an external
    # target -- the mechanism decides whether/how to turn it into agent incentives.
    d_e: int = 6


class MarketMechanism(FlatParamsMixin, nn.Module):
    """Layout and initialization spec of the learned market mechanism.

    Holds the real parameters of ``F``, whose joint output is the state delta
    followed by reward, so ``theta`` vectors are created from and decoded through
    it. Execution --
    including ordinary K=1 runs -- lives in :class:`BatchedMarketMechanism`."""

    def __init__(self, dims: MarketMechanismDims = MarketMechanismDims(), *, layernorm: bool = False):
        super().__init__()
        self.dims = dims
        # F inputs: own state, own action, own env, and the sum-pooled state and
        # action aggregates over the formula's agents.
        f_in = 2 * dims.d_state + 2 * dims.d_action + dims.d_e
        self.f = zero_final_mlp(
            f_in, dims.hidden, dims.d_state + 1, layernorm=layernorm
        )


class BatchedMarketMechanism(nn.Module):
    """K stacked :class:`MarketMechanism` weight sets applied per candidate.

    ``thetas`` is the K flat parameter vectors (a [K, n_params] tensor or a list);
    every candidate shares ``dims``/``layernorm``. The stacked weights are kept as
    buffers so ``.to(device)`` moves them with the module.
    """

    def __init__(self, thetas, dims: MarketMechanismDims, *, layernorm: bool = False):
        super().__init__()
        thetas = [thetas[k] for k in range(thetas.shape[0])] if torch.is_tensor(thetas) else list(thetas)
        self.K = len(thetas)
        self.dims = dims
        self.layernorm = bool(layernorm)
        if self.K < 1:
            raise ValueError("need at least one candidate theta")

        models = []
        for theta in thetas:
            m = MarketMechanism(dims, layernorm=layernorm).eval()
            m.load_flat_params(theta)
            models.append(m)

        self.register_buffer(
            "f_hidden_w", stack_params(models, lambda m: linears(m.f)[0].weight)
        )
        self.register_buffer(
            "f_hidden_b", stack_params(models, lambda m: linears(m.f)[0].bias)
        )
        self.register_buffer(
            "f_output_w", stack_params(models, lambda m: linears(m.f)[1].weight)
        )
        self.register_buffer(
            "f_output_b", stack_params(models, lambda m: linears(m.f)[1].bias)
        )
        if self.layernorm:
            self.ln_eps = float(first_layernorm(models[0].f).eps)
            self.register_buffer("ln_w", stack_params(models, lambda m: first_layernorm(m.f).weight))
            self.register_buffer("ln_b", stack_params(models, lambda m: first_layernorm(m.f).bias))

    def _input(self, s: torch.Tensor, a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Typed mechanism input; binary env bits are intentionally left raw."""
        A = s.shape[2]
        agg_s = s.sum(dim=2, keepdim=True).expand(-1, -1, A, -1)
        agg_a = a.sum(dim=2, keepdim=True).expand(-1, -1, A, -1)
        root_a = math.sqrt(float(A))
        d = self.dims
        return scaled_input(d, [
            (s, d.state_scale),
            (a, d.action_scale),
            (e, 1.0),
            (agg_s, root_a * d.state_scale),
            (agg_a, root_a * d.action_scale),
        ])

    def step(
        self,
        s: torch.Tensor,   # [K, M, A, d_state]
        a: torch.Tensor,   # [K, M, A, d_action]
        e: torch.Tensor,   # [K, M, A, d_e]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step for all K candidates.

        Returns ``(s', reward)``: the next states [K, M, A, d_state] and the
        per-(formula, agent) reward contributions [K, M, A], both returned by
        ``F``."""
        x = self._input(s, a, e)
        if self.layernorm:
            x = candidate_layernorm(x, self.ln_w, self.ln_b, self.ln_eps)
        hidden = F.gelu(candidate_linear(x, self.f_hidden_w, self.f_hidden_b))
        output = candidate_linear(hidden, self.f_output_w, self.f_output_b)
        delta, reward = output[..., :-1], output[..., -1]
        s_next = bounded_residual(
            s, delta, float(self.dims.state_bound), float(self.dims.state_decay)
        )
        return s_next, _finite(reward)


# Per-agent state width of the closed-form baselines (s = [p_old, cash_flow,
# position, marked]). Fixed, unlike the learned mechanism's searchable dim -- the
# observation codec and model builder read it through `mechanism_state_dims` so obs
# and model agree on the per-formula feature width.
BASELINE_STATE_DIM = 4


# The searched-mechanism class the outer loop instantiates for this env.
LEARNED_MECHANISM = MarketMechanism


def mechanism_state_dims(cfg) -> int:
    """The per-agent state width d_state of the mechanism backing this control mode."""
    if cfg.control_mode == 'learned':
        return int(cfg.learned_state_dim)
    return BASELINE_STATE_DIM


def mechanism_dims_from_config(cfg) -> MarketMechanismDims:
    return MarketMechanismDims(
        d_state=cfg.learned_state_dim,
        d_action=cfg.action_dim,
        hidden=cfg.learned_mechanism_hidden,
    )


class BaselineMarketMechanism(nn.Module):
    """Closed-form mechanism reproducing the hand-coded decentralized baselines.

    A drop-in learned-mechanism replacement whose ``F`` is an exact closed form,
    not a learned MLP -- so flat / fp3 / tgt / bounty are exact points of the
    mechanism family.
    Shape-generic over leading dims (the env applies it on the same
    ``[K, M, A, *]`` fold as the learned executor). Each step ``F``:

      * clears the market on the aggregated demand curve ``sum_k a`` plus an
        external bounty curve gated on ``is_target`` (in market mode bountied
        markets are exactly the target markets);
      * runs the per-agent ledger: midpoint-executed P&L, the resolution payout
        (YES shares pay 1), and a minted first-prover bonus.

    State ``s = [p_old, cash_flow, position, marked]`` (d_state=4); the carried
    ``p_old`` is last step's clearing price, identical across agents. Net wealth reads the marked coordinate,
    ``V = c0 + sum_phi marked``. ``trading=False`` drops the market dynamics (the
    bounty preset): only the bonus accrues. The action ``a = (q0, q1)`` is the
    agent's demand curve, sign clamped (``q0 >= 0``, ``q1 <= 0``) inside ``F`` --
    last-price marking only.
    """

    def __init__(self, *, bonus: float = 0.0, targets_only: bool = False,
                 team_reward: bool = False, trading: bool = True, bounty_demand: float = 0.0,
                 initial_cash: float = 0.0, negative_return_penalty: float = 0.0):
        super().__init__()
        self.dims = MarketMechanismDims(d_state=BASELINE_STATE_DIM, d_action=2)
        self.bonus = float(bonus)
        self.targets_only = bool(targets_only)
        self.team_reward = bool(team_reward)
        self.trading = bool(trading)
        self.bounty_demand = float(bounty_demand)
        self._c0 = float(initial_cash)
        self._lambda = float(negative_return_penalty)
        # Value reads the marked coordinate (index 3); already unit norm.
        self.register_buffer("_w", torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def value(self, agg_state: torch.Tensor) -> torch.Tensor:
        """Closed-form value g(V), V = c0 + sum_phi w . s, with the downside-risk
        penalty -- so flat/fp3/tgt/bounty stay exact points of the family."""
        v = self._c0 + agg_state @ self._w
        return v - self._lambda * torch.clamp(-v, min=0.0).pow(2)

    def forward(self, s, a, e):
        resolved_by_i = e[..., 0]                        # [..., A] this agent closed it
        # The four global bits are broadcast across agents; collapse to per-market.
        proven_true = e[..., 1].amax(dim=-1)             # [...] resolved TRUE
        theorem_resolved = e[..., 2].amax(dim=-1)        # [...] theorem closed
        is_target = e[..., 4].amax(dim=-1)               # [...] static target bit
        gate = is_target if self.targets_only else torch.ones_like(is_target)
        if self.team_reward:
            any_prover = (resolved_by_i > 0).to(s.dtype).amax(dim=-1)      # [...]
            bonus_term = (self.bonus * any_prover * gate).unsqueeze(-1)    # -> all agents
        else:
            bonus_term = self.bonus * resolved_by_i * gate.unsqueeze(-1)   # [..., A]

        p_old = s[..., 0]                                # [..., A] carried clearing price
        if not self.trading:
            cash = s[..., 1] + bonus_term
            return torch.stack([p_old, cash, s[..., 2], cash], dim=-1)

        q0 = torch.relu(a[..., 0])                       # [..., A]  q0 >= 0
        q1 = -torch.relu(-a[..., 1])                     # [..., A]  q1 <= 0
        bounty_q1 = -self.bounty_demand * is_target      # [...] external bounty curve
        Sq0 = q0.sum(dim=-1)                             # [...] aggregate net demand
        Sq1 = q1.sum(dim=-1) + bounty_q1                 # [...]
        denom = Sq0 - Sq1
        allzero = (Sq0.abs() <= EPS) & (Sq1.abs() <= EPS)
        safe_denom = torch.where(allzero, torch.ones_like(denom), denom)
        p_new = torch.clamp(Sq0 / safe_denom, 0.0, 1.0)
        p_new = torch.where(allzero, torch.zeros_like(p_new), p_new)
        p_new_b = p_new.unsqueeze(-1)                    # [..., 1]

        pos_old = s[..., 2]                              # [..., A]
        pos_new = (1.0 - p_new_b) * q0 + p_new_b * q1    # quantity at the new price
        avg = 0.5 * (p_old + p_new_b)                    # midpoint execution price
        cash = s[..., 1] - (pos_new - pos_old) * avg
        cash = cash + (theorem_resolved * proven_true).unsqueeze(-1) * pos_new + bonus_term
        resolved_b = theorem_resolved.bool().unsqueeze(-1).expand_as(cash)
        marked = torch.where(resolved_b, cash, cash + p_new_b * pos_new)
        p_old_next = p_new_b.expand_as(cash)             # next step's p_old <- p_new
        return torch.stack([p_old_next, cash, pos_new, marked], dim=-1)


def theta_from_config(cfg) -> torch.Tensor:
    """The configured flat theta: loaded from disk, or a fresh spec's init."""
    if cfg.learned_mechanism_params is not None:
        return torch.load(cfg.learned_mechanism_params, weights_only=True)
    return MarketMechanism(
        mechanism_dims_from_config(cfg), layernorm=cfg.learned_mechanism_layernorm
    ).flat_params()


def mechanism_from_config(cfg) -> nn.Module:
    if cfg.control_mode == "learned":
        return BatchedMarketMechanism(
            [theta_from_config(cfg)],
            mechanism_dims_from_config(cfg),
            layernorm=cfg.learned_mechanism_layernorm,
        )
    return BaselineMarketMechanism(
        bonus=cfg.first_prover_bonus,
        targets_only=cfg.prover_bonus_targets_only,
        team_reward=(cfg.control_mode == "collaborative"),
        trading=(cfg.control_mode == "market"),
        bounty_demand=cfg.bounty_demand,
        initial_cash=cfg.initial_cash,
        negative_return_penalty=cfg.negative_return_penalty,
    )
