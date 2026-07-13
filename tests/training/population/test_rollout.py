"""Population sparse-attention bundle tests."""

import torch

from agoraforge.models.graph_transformer import (
    build_edge_bundle,
    build_population_edge_bundle,
)


def test_population_sparse_bundle_matches_per_candidate_build():
    K, B, N, D = 3, 2, 4, 2
    graph = torch.eye(N, dtype=torch.bool).expand(K, B, N, N).clone()
    graph[:, :, 0, 1] = True
    graph[:, :, 1, 0] = True
    features = torch.randn(K, B, N, N, D)
    bias = torch.randn(K, B, N, N)

    population = build_population_edge_bundle(graph, features, bias)
    for k in range(K):
        solo = build_edge_bundle(graph[k], features[k], bias[k])
        for key in solo:
            assert torch.equal(population[key][k], solo[key])
