"""Graph-transformer trunk shared by every environment's actor/critic.

The trunk is env-agnostic: transformer layers with learned directed edge
messages, dense or sparse attention, and a PopArt value head. Everything an
environment owns -- node/edge feature encoding, the spatial-graph builder, and
the action-decoding head -- is injected by the env's model module (see
``agoraforge/envs/<name>/model.py``), which also chooses the concrete actor and
critic classes to construct.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math

import torch
import torch.nn as nn

# Attention implementation. "dense" forms the fixed-shape (B,H,N,N) score matrix:
# one kernel, flat memory, the standard masked softmax -- correct while the node
# count is small. "sparse" runs only over active edges (O(edges), for the large-N
# regime), but its index_add aggregation is nondeterministic on CUDA backward.
# "auto" picks dense below the node threshold.
_VALID_ATTN_MODES = {"dense", "sparse", "auto"}
_DEFAULT_SPARSE_ATTENTION_MIN_NODES = 64


def profile_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def build_edge_bundle(graph: torch.Tensor, edge_features: torch.Tensor, bias: torch.Tensor) -> dict:
    """Flatten the active edges of a batched graph for sparse attention.

    Per active edge ``i <- j`` this returns the edge feature, the scalar attention
    ``bias`` term, and the flattened source/destination node ids used to gather q/k/v
    and to segment-softmax over each destination's incoming edges. The graph, edge
    features and bias are identical across layers, so this runs once per forward.
    """
    _, n_nodes, _ = graph.shape
    b_idx, i_idx, j_idx = graph.nonzero(as_tuple=True)
    return {
        "feat": edge_features[b_idx, i_idx, j_idx],
        "bias": bias[b_idx, i_idx, j_idx],
        "src": b_idx * n_nodes + j_idx,
        "dst": b_idx * n_nodes + i_idx,
    }


def build_population_edge_bundle(
    graph: torch.Tensor, edge_features: torch.Tensor, bias: torch.Tensor,
) -> dict:
    """Build fixed-width sparse bundles for a ``[K, B, N, N]`` population.

    ``nonzero`` is not vmap-compatible. Running it once outside vmap and retaining
    the candidate axis lets the param-dependent sparse attention core stay vmapped.
    Population-folded environments have fixed graph geometry per candidate; fail
    rather than silently mix rows if an environment violates that contract.
    """
    K, B, N, _ = graph.shape
    k_idx, b_idx, i_idx, j_idx = graph.nonzero(as_tuple=True)
    counts = torch.bincount(k_idx, minlength=K)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("sparse population attention requires equal edge counts")
    edges = int(counts[0])
    return {
        "feat": edge_features[k_idx, b_idx, i_idx, j_idx].reshape(
            K, edges, edge_features.shape[-1]
        ),
        "bias": bias[k_idx, b_idx, i_idx, j_idx].reshape(K, edges),
        "src": (b_idx * N + j_idx).reshape(K, edges),
        "dst": (b_idx * N + i_idx).reshape(K, edges),
    }


def pointer_logits(
    nodes: torch.Tensor,
    query: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Per-node pointer logits ``<node, query>`` with invalid nodes masked out."""
    logits = torch.bmm(nodes, query.unsqueeze(-1)).squeeze(-1)
    if node_mask is not None:
        # -1e8 is outside IEEE fp16's finite range.  -1e4 still underflows
        # to exactly zero probability, while FP32/BF16 keep their old value.
        mask_value = -1e4 if logits.dtype == torch.float16 else -1e8
        logits = logits.masked_fill(node_mask == 0, mask_value)
    return logits


class GraphTransformerLayer(nn.Module):
    """Transformer block with learned directed edge messages."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        edge_feat_dim: int = 4,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )
        self.edge_score = nn.Linear(n_embd, n_head, bias=False)
        self.edge_value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.GELU(),
            nn.Linear(2 * n_embd, n_embd),
        )

    def forward(self, x, graph=None, bias=None, edge_features=None, *, edge_bundle=None):
        # An edge_bundle means the caller wants the sparse (per-edge) path; otherwise
        # run dense attention over the fixed (B,H,N,N) graph. Both compute the same
        # masked attention (edge bias + learned edge score, weighted v + edge value).
        if edge_bundle is None:
            return self._forward_dense(x, graph, bias, edge_features)
        return self._forward_sparse(x, edge_bundle)

    def _forward_dense(self, x, graph, bias, edge_features):
        bsz, n_nodes, n_embd = x.shape
        qkv = self.qkv(self.ln1(x)).view(bsz, n_nodes, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # each (B, H, N, head_dim)
        edge_h = self.edge_encoder(edge_features)  # (B, N, N, n_embd)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, N, N)
        scores = scores + bias.unsqueeze(1) + self.edge_score(edge_h).permute(0, 3, 1, 2)
        # Every node has a self-loop, so no row is fully masked (no all--inf row -> nan).
        scores = scores.masked_fill(~graph.bool().unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, H, N, N)
        edge_val = self.edge_value(edge_h).view(bsz, n_nodes, n_nodes, self.n_head, self.head_dim)
        y = weights @ v + torch.einsum("bhij,bijhd->bhid", weights, edge_val)
        y = y.permute(0, 2, 1, 3).reshape(bsz, n_nodes, n_embd)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))

    def _forward_sparse(self, x: torch.Tensor, edge_bundle: dict) -> torch.Tensor:
        # Attention runs only over active edges: q/k/v gathered per edge, scored,
        # softmaxed per destination node, weighted messages scattered back. Cost is
        # O(edges) in compute and memory rather than O(N^2).
        bsz, n_nodes, n_embd = x.shape
        qkv = self.qkv(self.ln1(x)).view(bsz * n_nodes, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each (B*N, H, head_dim)

        src, dst = edge_bundle["src"], edge_bundle["dst"]
        edge_h = self.edge_encoder(edge_bundle["feat"])
        scores = (q[dst] * k[src]).sum(-1) / math.sqrt(self.head_dim)  # (E, H)
        scores = scores + edge_bundle["bias"].unsqueeze(-1) + self.edge_score(edge_h)

        # Segment softmax over each destination's incoming edges. The per-segment max
        # is detached so the gradient matches a plain softmax; self-loops guarantee no
        # empty segment.
        n_seg = bsz * n_nodes
        seg_max = scores.new_full((n_seg, self.n_head), float("-inf"))
        seg_max = seg_max.scatter_reduce(
            0, dst.unsqueeze(-1).expand(-1, self.n_head), scores,
            reduce="amax", include_self=False,
        )
        exp_scores = torch.exp(scores - seg_max[dst].detach())  # (E, H)
        denom = torch.zeros(n_seg, self.n_head, dtype=x.dtype, device=x.device)
        denom = denom.index_add(0, dst, exp_scores)
        weights = exp_scores / denom[dst]  # (E, H)

        edge_val = self.edge_value(edge_h).view(-1, self.n_head, self.head_dim)
        msg = weights.unsqueeze(-1) * (v[src] + edge_val)  # (E, H, head_dim)
        y = torch.zeros(n_seg, self.n_head, self.head_dim, dtype=x.dtype, device=x.device)
        y = y.index_add(0, dst, msg)
        y = y.view(bsz, n_nodes, n_embd)

        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


@dataclass
class GraphTransformerConfig:
    """Trunk configuration plus the env name and env-owned head kwargs.

    ``env_name`` selects the environment's model module (encoder, graph builder,
    action head); ``head`` carries that module's head kwargs verbatim. The config
    is checkpoint-self-describing: ``from_dict`` restores everything needed to
    rebuild the model offline via the env registry.
    """

    env_name: str
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 3
    init_std: float = 0.02
    action_dim: int = 2
    node_feat_dim: int = 1
    agent_scalar_dim: int = 1
    attention_mode: str = "dense"
    sparse_attention_min_nodes: int = _DEFAULT_SPARSE_ATTENTION_MIN_NODES
    head: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.env_name = str(self.env_name)
        self.n_embd = int(self.n_embd)
        self.n_head = int(self.n_head)
        self.n_layer = int(self.n_layer)
        self.init_std = float(self.init_std)
        self.action_dim = int(self.action_dim)
        self.node_feat_dim = int(self.node_feat_dim)
        self.agent_scalar_dim = int(self.agent_scalar_dim)
        self.attention_mode = str(self.attention_mode)
        self.sparse_attention_min_nodes = int(self.sparse_attention_min_nodes)
        self.head = dict(self.head)
        if self.n_embd < 1 or self.n_head < 1 or self.n_layer < 1:
            raise ValueError("n_embd, n_head, and n_layer must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be positive")
        if min(self.action_dim, self.node_feat_dim, self.agent_scalar_dim) < 1:
            raise ValueError("action and input dimensions must be positive")
        if self.attention_mode not in _VALID_ATTN_MODES:
            raise ValueError(
                f"attention_mode must be one of {sorted(_VALID_ATTN_MODES)}, "
                f"got {self.attention_mode!r}"
            )
        if self.sparse_attention_min_nodes < 1:
            raise ValueError("sparse_attention_min_nodes must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphTransformerConfig":
        return cls(**data)


class GraphBackbone(nn.Module):
    """Env-provided node encoder followed by the shared transformer trunk.

    ``encoder`` owns feature projections and the spatial-graph builder; its
    ``forward(obs, profiler, profile_prefix)`` returns
    ``(nodes, graph, bias, edge_features, node_mask)`` and its ``edge_feat_dim``
    sizes the layers' edge encoders.
    """

    def __init__(self, config: GraphTransformerConfig, encoder: nn.Module):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.encoder = encoder
        self.attention_mode = config.attention_mode
        self.sparse_attention_min_nodes = int(config.sparse_attention_min_nodes)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    config.n_embd,
                    config.n_head,
                    edge_feat_dim=int(encoder.edge_feat_dim),
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, obs: dict, profiler=None, profile_prefix: str = ""):
        """Encode nodes and run the trunk; returns ``(x, node_mask)``."""
        nodes, graph, bias, edge_features, node_mask = self.encoder(
            obs, profiler=profiler, profile_prefix=profile_prefix)
        x = self._run_backbone(
            nodes, graph, bias, edge_features,
            edge_bundle=obs.get("_prebuilt_edge_bundle"),
            profiler=profiler, profile_prefix=profile_prefix,
        )
        return x, node_mask

    def _run_backbone(
        self,
        nodes: torch.Tensor,
        graph: torch.Tensor,
        bias: torch.Tensor,
        edge_features: torch.Tensor,
        edge_bundle=None,
        profiler=None,
        profile_prefix: str = "",
    ) -> torch.Tensor:
        with profile_section(profiler, f"{profile_prefix}attn_compute"):
            attention_mode = self._resolve_attention_mode(nodes.shape[1])
            if attention_mode == "sparse" and edge_bundle is None:
                edge_bundle = build_edge_bundle(graph, edge_features, bias)
            if attention_mode != "sparse":
                edge_bundle = None
            x = nodes
            for layer in self.layers:
                x = layer(x, graph, bias, edge_features, edge_bundle=edge_bundle)
            return self.ln_f(x)

    def _resolve_attention_mode(self, n_nodes: int) -> str:
        if self.attention_mode == "auto":
            return "sparse" if int(n_nodes) >= self.sparse_attention_min_nodes else "dense"
        return self.attention_mode


class GraphActor(GraphBackbone):
    """Actor decoding one action per (env, agent) row: global node queries the
    env's action head over the per-node states."""

    def __init__(self, config: GraphTransformerConfig, encoder: nn.Module, head_factory):
        super().__init__(config, encoder)
        self.action_head = head_factory()
        self.apply(self._init_weights)

    def forward(self, obs: dict, profiler=None, profile_prefix: str = "") -> dict:
        x, node_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        return self.action_head(x[:, 0], x[:, 1:], node_mask)


class PopArtValueHead(nn.Module):
    """Linear value head with PopArt normalization."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.linear = nn.Linear(n_embd, 1, bias=True)
        self.register_buffer("value_mean", torch.zeros(()))
        self.register_buffer("value_std", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)[:, 0]

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.value_mean) / self.value_std

    def denormalize_value(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.value_std + self.value_mean

    @torch.no_grad()
    def update_value_stats(self, new_mean: float, new_std: float) -> None:
        new_std = max(float(new_std), 1e-6)
        new_mean = float(new_mean)
        old_mean = float(self.value_mean)
        old_std = float(self.value_std)
        self.linear.weight.mul_(old_std / new_std)
        self.linear.bias.copy_((old_std * self.linear.bias + old_mean - new_mean) / new_std)
        self.value_mean.fill_(new_mean)
        self.value_std.fill_(new_std)


class SharedGraphActorCritic(GraphActor):
    """Unprivileged actor/critic with one public graph-transformer trunk.

    The critic deliberately reads the actor observation, not critic-only
    privileged features. Used by the population (mechanism-search) path, where
    the duplicated actor/critic attention work is collapsed without crossing the
    privileged-information boundary.
    """

    def __init__(self, config: GraphTransformerConfig, encoder: nn.Module, head_factory):
        super().__init__(config, encoder, head_factory)
        self.value_head = PopArtValueHead(config.n_embd)
        self.value_head.apply(self._init_weights)

    def forward_actor(self, obs: dict, profiler=None, profile_prefix: str = "") -> dict:
        x, node_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        return self.action_head(x[:, 0], x[:, 1:], node_mask)

    def forward_actor_critic(self, obs: dict, profiler=None, profile_prefix: str = "") -> tuple[dict, torch.Tensor]:
        x, node_mask = self.encode(obs, profiler=profiler, profile_prefix=profile_prefix)
        logits = self.action_head(x[:, 0], x[:, 1:], node_mask)
        values = self.value_head(x[:, 0, :])
        return logits, values

    def forward(self, obs: dict, profiler=None, profile_prefix: str = "", mode: str = "actor"):
        if mode == "actor_critic":
            return self.forward_actor_critic(obs, profiler=profiler, profile_prefix=profile_prefix)
        if mode != "actor":
            raise ValueError(f"unknown SharedGraphActorCritic mode {mode!r}")
        return self.forward_actor(obs, profiler=profiler, profile_prefix=profile_prefix)

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        return self.value_head.normalize_value(value)

    def denormalize_value(self, value: torch.Tensor) -> torch.Tensor:
        return self.value_head.denormalize_value(value)

    @torch.no_grad()
    def update_value_stats(self, new_mean: float, new_std: float) -> None:
        self.value_head.update_value_stats(new_mean, new_std)
