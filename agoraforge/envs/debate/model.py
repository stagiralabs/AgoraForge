"""Debate model pieces: node encoder, spatial-graph builder, and action head.

The graph-transformer trunk is shared (``agoraforge.models.graph_transformer``);
this module owns everything debate-specific about the models -- how claim
observations become nodes and edges, and how the trunk's outputs decode into
debate actions -- plus the builders the generic training code constructs models
through.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from agoraforge.models.graph_transformer import (
    GraphActor,
    GraphTransformerConfig,
    SharedGraphActorCritic,
    pointer_logits,
    profile_section,
)
from agoraforge.envs.debate.actions import NUM_REVEAL_TYPES
from agoraforge.envs.debate.obs import (
    AGENT_SCALAR_DIM,
    claim_feat_dim,
)
from agoraforge.envs.debate.protocol import mechanism_state_dims

EDGE_SELF = 0
EDGE_GLOBAL = 1
EDGE_COUPLING = 2        # signed Ising coupling J_uv
EDGE_COUPLING_ABS = 3
EDGE_FEAT_DIM = 4


class DebateNodeEncoder(nn.Module):
    """Project claim observations into [global] + claim nodes and build the graph."""

    edge_feat_dim = EDGE_FEAT_DIM

    def __init__(self, config: GraphTransformerConfig, feat_width: int):
        super().__init__()
        self.claim_proj = nn.Sequential(nn.Linear(int(feat_width), config.n_embd), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(config.agent_scalar_dim, config.n_embd), nn.GELU())
        self.type_emb = nn.Embedding(2, config.n_embd)  # global, claim

    def _build_spatial_graph(
        self,
        claim_mask: torch.Tensor,
        claim_couplings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Graph over [global] + claims: self-loops, global hub, and coupling edges.

        The attention bias carries the coupling magnitude prior ``log1p(|J|)``; the
        signed coupling itself rides the edge features. Symmetric (the factor graph
        is undirected). Param-free, so the population path can prebuild it once.
        """
        bsz, n_claims = claim_mask.shape
        n_nodes = 1 + n_claims
        device = claim_mask.device

        valid = torch.cat(
            [torch.ones(bsz, 1, dtype=torch.bool, device=device), claim_mask.bool()], dim=1
        )
        graph = torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0).expand(bsz, -1, -1).clone()
        bias = torch.zeros(bsz, n_nodes, n_nodes, dtype=claim_mask.dtype, device=device)
        edge_features = torch.zeros(
            bsz, n_nodes, n_nodes, self.edge_feat_dim, dtype=claim_mask.dtype, device=device
        )
        diag = torch.arange(n_nodes, device=device)
        edge_features[:, diag, diag, EDGE_SELF] = valid.to(edge_features.dtype)

        claim_valid = claim_mask.bool()
        graph[:, 0, 1:] = claim_valid
        graph[:, 1:, 0] = claim_valid
        edge_features[:, 0, 1:, EDGE_GLOBAL] = claim_valid.to(edge_features.dtype)
        edge_features[:, 1:, 0, EDGE_GLOBAL] = claim_valid.to(edge_features.dtype)

        coupling_active = (claim_couplings != 0) & claim_valid.unsqueeze(1) & claim_valid.unsqueeze(2)
        claims = slice(1, n_nodes)
        graph[:, claims, claims] |= coupling_active
        bias[:, claims, claims] = torch.log1p(claim_couplings.abs()) * coupling_active.to(bias.dtype)
        edge_features[:, claims, claims, EDGE_COUPLING] = claim_couplings * coupling_active.to(bias.dtype)
        edge_features[:, claims, claims, EDGE_COUPLING_ABS] = (
            claim_couplings.abs() * coupling_active.to(bias.dtype)
        )

        edge_features = edge_features * graph.unsqueeze(-1).to(edge_features.dtype)
        return graph, bias, edge_features

    def prebuild_graph(self, obs: dict):
        """Param-free graph build for the population path's vmap prestaging."""
        return self._build_spatial_graph(obs["claim_mask"], obs["claim_couplings"])

    def forward(self, obs: dict, profiler=None, profile_prefix: str = ""):
        claim_mask = obs["claim_mask"]
        global_nodes = self.global_proj(obs["agent_scalars"]) + self.type_emb.weight[0]
        claim_nodes = self.claim_proj(obs["claim_features"]) + self.type_emb.weight[1]
        nodes = torch.cat([global_nodes.unsqueeze(1), claim_nodes], dim=1)

        # A pre-built (graph, bias, edge_features) skips the spatial-graph build. It is
        # param-free, so the population path builds it once over the K*B_base batch and
        # vmaps only the param-dependent encode/core over the K stacked policies.
        prebuilt = obs.get("_prebuilt_graph")
        if prebuilt is not None:
            graph, bias, edge_features = prebuilt
        else:
            with profile_section(profiler, f"{profile_prefix}graph_build"):
                graph, bias, edge_features = self._build_spatial_graph(
                    claim_mask, obs["claim_couplings"]
                )
        return nodes, graph, bias, edge_features, claim_mask


class DebateActionHead(nn.Module):
    """Decode the current graph state into factored debate-action logits."""

    def __init__(
        self,
        n_embd: int,
        signal_log_std_base: float,
        log_std_min: float,
        log_std_max: float,
        action_dim: int = 2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.signal_log_std_base = signal_log_std_base
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.reveal_type_head = nn.Linear(n_embd, NUM_REVEAL_TYPES)
        self.claim_query = nn.Linear(n_embd, n_embd)
        # Continuous signal head: a diagonal Gaussian over an `action_dim`-vector
        # latent per claim; the env applies the sample directly as the protocol
        # mechanism input a[v, j].
        self.signal_mu = nn.Linear(n_embd, self.action_dim)
        self.signal_log_std = nn.Linear(n_embd, self.action_dim)

    def forward(
        self,
        global_state: torch.Tensor,
        claim_nodes: torch.Tensor,
        claim_mask: torch.Tensor,
    ) -> dict:
        result = {"reveal_type_logits": self.reveal_type_head(global_state)}
        cq = self.claim_query(global_state)
        result["claim_logits"] = pointer_logits(claim_nodes, cq, claim_mask)
        result["signal_mu"] = self.signal_mu(claim_nodes)
        # nan_to_num BEFORE clamp: clamp propagates NaN, so a NaN log_std (from a
        # divergent mechanism poisoning the obs) would survive to exp() -> a NaN std
        # -> torch.normal crash. Mapping NaN to log_std_min makes the clamp a guard.
        result["signal_log_std"] = torch.nan_to_num(
            self.signal_log_std(claim_nodes) + self.signal_log_std_base,
            nan=self.log_std_min,
        ).clamp(self.log_std_min, self.log_std_max)
        return result


def build_model_config(model_cfg_input, env_cfg, decoding_cfg=None) -> GraphTransformerConfig:
    # The claim-feature input width includes the protocol mechanism's per-agent
    # state vector, so size the model's claim projection to match its d_state.
    d_state = mechanism_state_dims(env_cfg)
    return GraphTransformerConfig(
        env_name="debate",
        n_embd=model_cfg_input.n_embd,
        n_head=model_cfg_input.n_head,
        n_layer=model_cfg_input.n_layer,
        init_std=model_cfg_input.init_std,
        attention_mode=model_cfg_input.attention_mode or "dense",
        sparse_attention_min_nodes=model_cfg_input.sparse_attention_min_nodes,
        action_dim=int(env_cfg.action_dim),
        node_feat_dim=claim_feat_dim(d_state),
        agent_scalar_dim=AGENT_SCALAR_DIM,
        head={
            "signal_log_std_base": math.log(float(decoding_cfg.signal_init_std)),
            "log_std_min": float(decoding_cfg.log_std_min),
            "log_std_max": float(decoding_cfg.log_std_max),
        },
    )


def _head_factory(config: GraphTransformerConfig):
    return lambda: DebateActionHead(
        config.n_embd,
        signal_log_std_base=config.head["signal_log_std_base"],
        log_std_min=config.head["log_std_min"],
        log_std_max=config.head["log_std_max"],
        action_dim=config.action_dim,
    )


def build_actor(config: GraphTransformerConfig) -> GraphActor:
    encoder = DebateNodeEncoder(config, feat_width=config.node_feat_dim)
    return GraphActor(config, encoder, _head_factory(config))


def build_shared(config: GraphTransformerConfig) -> SharedGraphActorCritic:
    encoder = DebateNodeEncoder(config, feat_width=config.node_feat_dim)
    return SharedGraphActorCritic(config, encoder, _head_factory(config))
