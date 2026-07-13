"""Protocol-bound debate mechanism.

For each public claim ``c`` and debater ``i`` the protocol carries a state
``s[i,c]``.  Turns are sequential.  On debater ``i``'s turn, ``F`` reads that
debater's state and signal, the other debater's state, and the sum of ``i``'s
states across claims. ``F`` returns both the state update and reward.

After the debate, ``w`` selects the bounded set of claims sent to the judge.  The
judge's marginals are fed through one final zero-action update, then ``P`` maps the
mean public state to the protocol's target probability.  ``F``, ``w``, and ``P``
are all part of the flat parameter vector searched by the outer loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from agoraforge.envs.mechanism import (
    FlatParamsMixin,
    MechanismDims,
    bounded_residual,
    candidate_layernorm,
    candidate_linear,
    finite,
    first_layernorm,
    linears,
    scaled_input,
    stack_params,
    zero_final_mlp,
)


# Environment input e[i,c,t]. Judge columns are zero during debate turns and are
# populated only for the single terminal judge-feedback update.
E_SUBMITTED_BY_ME = 0
E_IN_TRANSCRIPT = 1
E_REVEALED_NOW = 2
E_IS_TARGET = 3
E_ROLE = 4
E_IS_MY_TURN = 5
E_JUDGE_SELECTED = 6
E_JUDGE_P = 7
E_IS_JUDGE_STEP = 8
E_TIMESTEP_RATIO = 9
D_E = 10


@dataclass(frozen=True)
class DebateMechanismDims(MechanismDims):
    d_e: int = D_E
    predictor_hidden: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.predictor_hidden < 1:
            raise ValueError("predictor_hidden must be >= 1")


class DebateMechanism(FlatParamsMixin, nn.Module):
    """Layout and initialization spec of the protocol mechanism.

    Holds the real parameters (the judge selector ``w``, the sequential map
    ``F``, and the output predictor ``P``) so
    ``theta`` vectors are created from and decoded through it. Execution --
    including ordinary K=1 runs -- lives in :class:`BatchedDebateMechanism`."""

    def __init__(self, dims: DebateMechanismDims = DebateMechanismDims(), *, layernorm: bool = False):
        super().__init__()
        self.dims = dims
        self.layernorm = bool(layernorm)
        # own state, action, environment input, opponent state, own global state.
        f_in = 3 * dims.d_state + dims.d_action + dims.d_e
        self.judge_w = nn.Parameter(torch.zeros(dims.d_state))
        self.f = zero_final_mlp(
            f_in, dims.hidden, dims.d_state + 1, layernorm=layernorm
        )
        self.p = zero_final_mlp(
            dims.d_state, dims.predictor_hidden, 1, layernorm=False
        )


class BatchedDebateMechanism(nn.Module):
    """K stacked :class:`DebateMechanism` weight sets applied per candidate."""

    def __init__(self, thetas, dims: DebateMechanismDims, *, layernorm: bool = False):
        super().__init__()
        theta_list = [thetas[k] for k in range(thetas.shape[0])] if torch.is_tensor(thetas) else list(thetas)
        if not theta_list:
            raise ValueError("need at least one candidate theta")
        self.K = len(theta_list)
        self.dims = dims
        self.layernorm = bool(layernorm)
        models = []
        for theta in theta_list:
            model = DebateMechanism(dims, layernorm=layernorm).eval()
            model.load_flat_params(theta)
            models.append(model)

        self.register_buffer("w_q", stack_params(models, lambda m: m.judge_w))
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
        self.register_buffer("p1_w", stack_params(models, lambda m: linears(m.p)[0].weight))
        self.register_buffer("p1_b", stack_params(models, lambda m: linears(m.p)[0].bias))
        self.register_buffer("p2_w", stack_params(models, lambda m: linears(m.p)[1].weight))
        self.register_buffer("p2_b", stack_params(models, lambda m: linears(m.p)[1].bias))
        if self.layernorm:
            self.ln_eps = float(first_layernorm(models[0].f).eps)
            self.register_buffer("ln_w", stack_params(models, lambda m: first_layernorm(m.f).weight))
            self.register_buffer("ln_b", stack_params(models, lambda m: first_layernorm(m.f).bias))

    def _input(self, s, a, e, agent: int):
        own = s[..., agent, :]
        other = s[..., 1 - agent, :]
        own_sum = own.sum(dim=-2, keepdim=True).expand_as(own)
        root_n = math.sqrt(float(s.shape[-3]))
        d = self.dims
        return scaled_input(d, [
            (own, d.state_scale),
            (a, d.action_scale),
            (e, 1.0),
            (other, d.state_scale),
            (own_sum, root_n * d.state_scale),
        ])

    def turn(self, s, a, e, agent: int):
        """Return the active debater's next per-claim state and reward contributions."""
        x = self._input(s, a, e, agent)
        if self.layernorm:
            x = candidate_layernorm(x, self.ln_w, self.ln_b, self.ln_eps)
        hidden = F.gelu(candidate_linear(x, self.f_hidden_w, self.f_hidden_b))
        output = candidate_linear(hidden, self.f_output_w, self.f_output_b)
        delta, reward = output[..., :-1], output[..., -1]
        own = s[..., agent, :]
        own_next = bounded_residual(
            own, delta, float(self.dims.state_bound), float(self.dims.state_decay)
        )
        return own_next, finite(reward)

    def judge_query_scores(self, s):
        """Score every claim by ``<sum_i s[i,c], w>``."""
        return finite(torch.einsum("kbnd,kd->kbn", s.sum(dim=-2), self.w_q))

    def predict(self, s, transcript):
        """Protocol output ``P(mean_{public c,i} s[i,c])`` in ``[0, 1]``."""
        mask = transcript.to(s.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = transcript.sum(dim=-1).clamp_min(1).to(s.dtype) * s.shape[-2]
        mean_state = (s * mask).sum(dim=(-3, -2)) / denom.unsqueeze(-1)
        hidden = F.gelu(candidate_linear(mean_state, self.p1_w, self.p1_b))
        logits = candidate_linear(hidden, self.p2_w, self.p2_b)
        return torch.sigmoid(logits.squeeze(-1))


LEARNED_MECHANISM = DebateMechanism
BASELINE_STATE_DIM = 1


def mechanism_state_dims(cfg) -> int:
    return int(cfg.learned_state_dim) if cfg.control_mode == "learned" else BASELINE_STATE_DIM


def mechanism_dims_from_config(cfg) -> DebateMechanismDims:
    return DebateMechanismDims(
        d_state=cfg.learned_state_dim,
        d_action=cfg.action_dim,
        hidden=cfg.learned_mechanism_hidden,
        predictor_hidden=cfg.learned_predictor_hidden,
    )


class VanillaDebateMechanism(nn.Module):
    """Marker object for the closed-form vanilla terminal reward."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.dims = DebateMechanismDims(d_state=BASELINE_STATE_DIM, d_action=action_dim)


def theta_from_config(cfg) -> torch.Tensor:
    """The configured flat theta: loaded from disk, or a fresh spec's init."""
    # ConfigDict represents the optional path as ``''`` in run configs.
    if cfg.learned_mechanism_params:
        return torch.load(cfg.learned_mechanism_params, weights_only=True)
    return DebateMechanism(
        mechanism_dims_from_config(cfg), layernorm=cfg.learned_mechanism_layernorm
    ).flat_params()


def mechanism_from_config(cfg) -> nn.Module:
    if cfg.control_mode == "vanilla":
        return VanillaDebateMechanism(cfg.action_dim)
    return BatchedDebateMechanism(
        [theta_from_config(cfg)],
        mechanism_dims_from_config(cfg),
        layernorm=cfg.learned_mechanism_layernorm,
    )
