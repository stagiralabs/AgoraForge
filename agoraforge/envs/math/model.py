"""Math model pieces: node encoders, spatial-graph builders, and the action head.

The graph-transformer trunk is shared (``agoraforge.models.graph_transformer``);
this module owns everything math-specific about the models -- formula/agent
observation encoding, the edge structure, the
centralized planner's model-slot machinery, and the factored math action head --
plus the builders the generic training code constructs models through.
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
from agoraforge.envs.math.actions import MATH_ACTION_TYPES, NUM_MATH_ACTION_TYPES
from agoraforge.envs.math.obs import (
    AGENT_SCALAR_DIM,
    FORMULA_FEATURE_DIM,
    SLOT_FEATURE_DIM,
    learned_formula_feat_dim,
)
from agoraforge.envs.math.mechanism import mechanism_state_dims

EDGE_SELF = 0
EDGE_GLOBAL = 1
EDGE_NEGATION = 2
EDGE_QUERY_FORWARD = 3
EDGE_QUERY_REVERSE = 4
EDGE_QUERY_FORWARD_TIME = 5
EDGE_QUERY_REVERSE_TIME = 6
EDGE_FEAT_DIM = 7

# Centralized mode extends edges with slot->formula job channels (active-job
# indicator plus cumulative proof/conjecture counts).
EDGE_SLOT_JOB = 7
EDGE_SLOT_CUM_PROOF = 8
EDGE_SLOT_CUM_CONJ = 9
CENTRALIZED_EDGE_FEAT_DIM = 10


def _wire_slot_nodes(graph, edge_features, slot_formula_edges, f_count, formula_valid):
    """Add model-slot node edges to a built graph.

    Each slot node connects only to the formula its model is working on -- no
    slot<->slot edges, which would be O(A^2) as the model count grows: the
    slot->formula job edge onto the formula node (occupancy every model's dense
    pointer reads) and the global hub; the global edge also keeps an idle slot's
    node reachable.
    """
    n_nodes = graph.shape[1]
    slot_nodes = slice(1 + f_count, n_nodes)

    graph[:, 0, slot_nodes] = True
    graph[:, slot_nodes, 0] = True
    edge_features[:, 0, slot_nodes, EDGE_GLOBAL] = 1.0
    edge_features[:, slot_nodes, 0, EDGE_GLOBAL] = 1.0

    sf = slot_formula_edges  # (B, S, F, SLOT_EDGE_DIM)
    sf_active = sf[..., 0] > 0
    formula_nodes = slice(1, 1 + f_count)
    graph[:, slot_nodes, formula_nodes] |= sf_active
    graph[:, formula_nodes, slot_nodes] |= sf_active.transpose(1, 2)
    for channel, edge_idx in enumerate((EDGE_SLOT_JOB, EDGE_SLOT_CUM_PROOF, EDGE_SLOT_CUM_CONJ)):
        edge_features[:, slot_nodes, formula_nodes, edge_idx] = sf[..., channel]
        edge_features[:, formula_nodes, slot_nodes, edge_idx] = sf[..., channel].transpose(1, 2)


class _MathEncoderBase(nn.Module):
    """Shared node-embedding machinery for the math node encoder."""

    def __init__(self, config: GraphTransformerConfig, feat_width: int):
        super().__init__()
        self.centralized = bool(config.head.get("centralized", False))
        self.edge_feat_dim = CENTRALIZED_EDGE_FEAT_DIM if self.centralized else EDGE_FEAT_DIM
        self.formula_proj = nn.Sequential(nn.Linear(int(feat_width), config.n_embd), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(config.agent_scalar_dim, config.n_embd), nn.GELU())
        # global, formula (+ model slot when centralized)
        self.type_emb = nn.Embedding(3 if self.centralized else 2, config.n_embd)
        if self.centralized:
            self.slot_proj = nn.Sequential(nn.Linear(SLOT_FEATURE_DIM, config.n_embd), nn.GELU())

    def _encode_node_embeddings(self, obs: dict):
        formula = obs["formula_features"]
        agent = obs["agent_scalars"]
        formula_mask = obs["formula_mask"]

        global_nodes = self.global_proj(agent) + self.type_emb.weight[0]
        formula_nodes = self.formula_proj(formula) + self.type_emb.weight[1]

        node_groups = [global_nodes.unsqueeze(1), formula_nodes]
        slot_formula_edges = None
        if self.centralized:
            node_groups.append(self.slot_proj(obs["slot_features"]) + self.type_emb.weight[2])
            slot_formula_edges = obs["slot_formula_edges"]
        return torch.cat(node_groups, dim=1), formula_mask, slot_formula_edges

    def _graph_scaffold(self, formula_mask, slot_formula_edges):
        bsz, f_count = formula_mask.shape
        s_count = slot_formula_edges.shape[1] if slot_formula_edges is not None else 0
        n_nodes = 1 + f_count + s_count
        device = formula_mask.device

        valid = torch.cat(
            [
                torch.ones(bsz, 1, dtype=torch.bool, device=device),
                formula_mask.bool(),
                torch.ones(bsz, s_count, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        graph = torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0).expand(bsz, -1, -1).clone()
        graph &= valid.unsqueeze(1) & valid.unsqueeze(2)
        graph |= torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        bias = torch.zeros(bsz, n_nodes, n_nodes, dtype=formula_mask.dtype, device=device)
        edge_features = torch.zeros(
            bsz, n_nodes, n_nodes, self.edge_feat_dim,
            dtype=formula_mask.dtype, device=device,
        )
        diag = torch.arange(n_nodes, device=device)
        edge_features[:, diag, diag, EDGE_SELF] = valid.to(edge_features.dtype)

        formula_valid = formula_mask.bool()
        graph[:, 0, 1:1 + f_count] = formula_valid
        graph[:, 1:1 + f_count, 0] = formula_valid
        edge_features[:, 0, 1:1 + f_count, EDGE_GLOBAL] = formula_valid
        edge_features[:, 1:1 + f_count, 0, EDGE_GLOBAL] = formula_valid
        return graph, bias, edge_features, formula_valid, f_count, s_count

    @staticmethod
    def _negation_active(formula_ids, neg_formula_ids, formula_valid):
        return (
            (formula_ids[:, :, None] == neg_formula_ids[:, None, :])
            & formula_valid[:, :, None]
            & formula_valid[:, None, :]
            & (formula_ids[:, :, None] >= 0)
            & (neg_formula_ids[:, None, :] >= 0)
        )


class MathActorNodeEncoder(_MathEncoderBase):
    """Actor graph: negation pairs and (public) query-time edges."""

    def _build_spatial_graph(
        self,
        formula_mask: torch.Tensor,
        formula_ids: torch.Tensor,
        neg_formula_ids: torch.Tensor,
        query_related_edges: torch.Tensor,
        slot_formula_edges: torch.Tensor | None = None,
    ):
        graph, bias, edge_features, formula_valid, f_count, s_count = self._graph_scaffold(
            formula_mask, slot_formula_edges)

        neg_active = self._negation_active(formula_ids, neg_formula_ids, formula_valid)
        query_edges = query_related_edges[:, :f_count, :f_count]
        query_active = (query_edges > 0) & formula_valid.unsqueeze(1) & formula_valid.unsqueeze(2)
        formula_nodes = slice(1, 1 + f_count)
        graph[:, formula_nodes, formula_nodes] |= neg_active
        graph[:, formula_nodes, formula_nodes] |= query_active
        graph[:, formula_nodes, formula_nodes] |= query_active.transpose(1, 2)
        query_improvement = (1.0 - query_edges).clamp(min=0.0, max=1.0)
        query_bias = torch.log1p(query_improvement) * query_active.to(bias.dtype)
        bias[:, formula_nodes, formula_nodes] = query_bias + query_bias.transpose(1, 2)
        edge_features[:, formula_nodes, formula_nodes, EDGE_NEGATION] = neg_active
        edge_features[:, formula_nodes, formula_nodes, EDGE_QUERY_FORWARD] = query_active.transpose(1, 2)
        edge_features[:, formula_nodes, formula_nodes, EDGE_QUERY_REVERSE] = query_active
        edge_features[:, formula_nodes, formula_nodes, EDGE_QUERY_FORWARD_TIME] = query_edges.transpose(1, 2)
        edge_features[:, formula_nodes, formula_nodes, EDGE_QUERY_REVERSE_TIME] = query_edges

        if s_count:
            _wire_slot_nodes(graph, edge_features, slot_formula_edges, f_count, formula_valid)

        edge_features = edge_features * graph.unsqueeze(-1).to(edge_features.dtype)
        return graph, bias, edge_features

    def prebuild_graph(self, obs: dict):
        """Param-free graph build for the population path's vmap prestaging."""
        return self._build_spatial_graph(
            obs["formula_mask"],
            obs["formula_ids"],
            obs["neg_formula_ids"],
            obs["query_related_edges"],
        )

    def forward(self, obs: dict, profiler=None, profile_prefix: str = ""):
        formula_ids = obs["formula_ids"]
        neg_formula_ids = obs["neg_formula_ids"]
        query_related_edges = obs["query_related_edges"]
        nodes, formula_mask, slot_formula_edges = self._encode_node_embeddings(obs)

        prebuilt = obs.get("_prebuilt_graph")
        if prebuilt is not None:
            graph, bias, edge_features = prebuilt
        else:
            with profile_section(profiler, f"{profile_prefix}graph_build"):
                graph, bias, edge_features = self._build_spatial_graph(
                    formula_mask,
                    formula_ids,
                    neg_formula_ids,
                    query_related_edges,
                    slot_formula_edges,
                )
        return nodes, graph, bias, edge_features, formula_mask


class MathActionHead(nn.Module):
    """Decode the current graph state into factored math-action logits."""

    def __init__(
        self,
        n_embd: int,
        demand_log_std_base: float,
        budget_log_std_base: float,
        budget_mu_base: float,
        log_std_min: float,
        log_std_max: float,
        action_dim: int = 2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.demand_log_std_base = demand_log_std_base
        self.budget_log_std_base = budget_log_std_base
        self.budget_mu_base = budget_mu_base
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.math_type_head = nn.Linear(n_embd, NUM_MATH_ACTION_TYPES)
        self.formula_query = nn.Linear(n_embd, n_embd)
        self.type_formula_bias = nn.Embedding(NUM_MATH_ACTION_TYPES, n_embd)
        # Budget head: a Gaussian over a latent z from the global state, mapped to an
        # integer budget by the env (mirrors the demand-delta heads below).
        self.budget_mu = nn.Linear(n_embd, 1)
        self.budget_log_std = nn.Linear(n_embd, 1)
        self.publish_statement_head = nn.Linear(n_embd, 2)
        self.publish_proof_head = nn.Linear(n_embd, 2)
        # Continuous market-action head: a diagonal Gaussian over an `action_dim`-
        # vector latent per formula. The env applies it directly as the action, so
        # the mean is the action and the std is additive exploration noise; both are
        # per-formula heads, so the exploration scale can depend on the
        # formula/state. (action_dim=2 is the demand-delta (q0, q1).)
        self.market_action_mu = nn.Linear(n_embd, self.action_dim)
        self.market_action_log_std = nn.Linear(n_embd, self.action_dim)
        self.mode_head = nn.Linear(n_embd, 2)

    def _gaussian_log_std(self, head: nn.Linear, x: torch.Tensor, base: float) -> torch.Tensor:
        # nan_to_num BEFORE clamp: clamp propagates NaN, so a NaN log_std (from a
        # divergent mechanism poisoning the obs) would survive to exp() -> a NaN std
        # -> torch.normal crash. Mapping NaN to log_std_min first makes the clamp
        # a hard guard; behavior-preserving for finite log_std.
        raw = torch.nan_to_num(head(x).squeeze(-1) + base, nan=self.log_std_min)
        return raw.clamp(self.log_std_min, self.log_std_max)

    def forward(
        self,
        global_state: torch.Tensor,
        formula_nodes: torch.Tensor,
        formula_mask: torch.Tensor,
    ) -> dict:
        result = {
            "math_type_logits": self.math_type_head(global_state),
        }

        fq = self.formula_query(global_state)
        result["formula_logits"] = pointer_logits(formula_nodes, fq, formula_mask)
        result["formula_logits_per_type"] = {}
        for type_idx, type_name in enumerate(MATH_ACTION_TYPES):
            # Every batch element looks up the same row, so this is a broadcast of
            # one weight row -- written as direct indexing rather than an Embedding
            # lookup on purpose: nn.Embedding's backward scatter-adds all the
            # repeated indices into one row via atomics, which is nondeterministic;
            # weight[i] broadcast sums deterministically. Mathematically identical.
            typed_query = fq + self.type_formula_bias.weight[type_idx]
            result["formula_logits_per_type"][type_name] = pointer_logits(
                formula_nodes, typed_query, formula_mask
            )

        result["budget_mu"] = self.budget_mu(global_state).squeeze(-1) + self.budget_mu_base
        result["budget_log_std"] = self._gaussian_log_std(
            self.budget_log_std, global_state, self.budget_log_std_base)
        # Surface the init offsets so the loss can use N(mu_base, std_base^2) -- the
        # init distribution -- as the budget KL base policy.
        result["budget_mu_base"] = self.budget_mu_base
        result["budget_log_std_base"] = self.budget_log_std_base
        result["publish_statement_logits"] = self.publish_statement_head(formula_nodes)
        result["publish_proof_logits"] = self.publish_proof_head(formula_nodes)
        # (B, F, action_dim) mean and log-std for the per-formula market action.
        result["market_action_mu"] = self.market_action_mu(formula_nodes)
        result["market_action_log_std"] = torch.nan_to_num(
            self.market_action_log_std(formula_nodes) + self.demand_log_std_base,
            nan=self.log_std_min,
        ).clamp(self.log_std_min, self.log_std_max)
        result["mode_logits"] = self.mode_head(global_state)
        return result


def _actor_from_slots(x: torch.Tensor, formula_mask: torch.Tensor, action_head) -> dict:
    """Centralized decode: apply the shared head to every model-slot embedding.

    One forward per env decodes every model slot, with the (shared) formula nodes
    broadcast across slots. Rows come out env-major (env, slot) -> env*A + slot,
    matching the (B, A) action/advantage layout the sampler and PPO use.
    """
    B, n_nodes, n_embd = x.shape
    f_count = formula_mask.shape[-1]
    A = n_nodes - 1 - f_count
    head_state = x[:, 1 + f_count:].reshape(B * A, n_embd)
    formula_nodes = x[:, 1:1 + f_count].unsqueeze(1).expand(B, A, f_count, n_embd).reshape(B * A, f_count, n_embd)
    formula_mask = formula_mask.unsqueeze(1).expand(B, A, f_count).reshape(B * A, f_count)
    return action_head(head_state, formula_nodes, formula_mask)


class CentralizedGraphActor(GraphActor):
    """Centralized planner actor: one joint graph per env (global + formula nodes
    + one model-slot node per agent); the factored head decodes every slot."""

    def forward(self, obs: dict, profiler=None, profile_prefix: str = "") -> dict:
        x, formula_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        return _actor_from_slots(x, formula_mask, self.action_head)


def build_model_config(model_cfg_input, env_cfg, decoding_cfg=None) -> GraphTransformerConfig:
    # Mechanism modes replace the demand-curve columns with the mechanism state
    # vector, so the formula-feature input width changes; size the model's formula
    # projection to match the mechanism's d_state.
    centralized = env_cfg.control_mode == 'centralized'
    if centralized:
        node_feat_dim = FORMULA_FEATURE_DIM
    else:
        node_feat_dim = learned_formula_feat_dim(mechanism_state_dims(env_cfg))
    # The centralized joint graph is large, so its unpinned default switches to
    # sparse attention at the configured threshold. Other modes default dense.
    attention_mode = model_cfg_input.attention_mode or ("auto" if centralized else "dense")
    return GraphTransformerConfig(
        env_name="math",
        n_embd=model_cfg_input.n_embd,
        n_head=model_cfg_input.n_head,
        n_layer=model_cfg_input.n_layer,
        init_std=model_cfg_input.init_std,
        attention_mode=attention_mode,
        sparse_attention_min_nodes=model_cfg_input.sparse_attention_min_nodes,
        action_dim=int(env_cfg.action_dim),
        node_feat_dim=node_feat_dim,
        agent_scalar_dim=AGENT_SCALAR_DIM,
        head={
            "centralized": centralized,
            "demand_log_std_base": math.log(float(decoding_cfg.demand_init_std)),
            "budget_log_std_base": math.log(float(decoding_cfg.budget_init_std)),
            "budget_mu_base": math.log(float(decoding_cfg.budget_init)),
            "log_std_min": float(decoding_cfg.log_std_min),
            "log_std_max": float(decoding_cfg.log_std_max),
        },
    )


def _head_factory(config: GraphTransformerConfig):
    return lambda: MathActionHead(
        config.n_embd,
        demand_log_std_base=config.head["demand_log_std_base"],
        budget_log_std_base=config.head["budget_log_std_base"],
        budget_mu_base=config.head["budget_mu_base"],
        log_std_min=config.head["log_std_min"],
        log_std_max=config.head["log_std_max"],
        action_dim=config.action_dim,
    )


def build_actor(config: GraphTransformerConfig):
    encoder = MathActorNodeEncoder(config, feat_width=config.node_feat_dim)
    cls = CentralizedGraphActor if config.head.get("centralized") else GraphActor
    return cls(config, encoder, _head_factory(config))


class CentralizedSharedActorCritic(SharedGraphActorCritic):
    """Centralized planner on the shared trunk: the factored head decodes every
    model slot of the joint graph; the value head reads the global token (one
    team value per env)."""

    def forward_actor(self, obs: dict, profiler=None, profile_prefix: str = "") -> dict:
        x, formula_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        return _actor_from_slots(x, formula_mask, self.action_head)

    def forward_actor_critic(self, obs: dict, profiler=None, profile_prefix: str = ""):
        x, formula_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        logits = _actor_from_slots(x, formula_mask, self.action_head)
        return logits, self.value_head(x[:, 0, :])


def build_shared(config: GraphTransformerConfig) -> SharedGraphActorCritic:
    encoder = MathActorNodeEncoder(config, feat_width=config.node_feat_dim)
    cls = CentralizedSharedActorCritic if config.head.get("centralized") else SharedGraphActorCritic
    return cls(config, encoder, _head_factory(config))
