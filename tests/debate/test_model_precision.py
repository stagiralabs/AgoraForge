"""Precision-specific model guards."""

import torch

from agoraforge.models.graph_transformer import pointer_logits


def test_pointer_mask_is_representable_in_fp16():
    query = torch.randn(2, 4).half()
    nodes = torch.randn(2, 3, 4).half()
    mask = torch.tensor([[1, 0, 1], [0, 0, 1]])
    logits = pointer_logits(nodes, query, mask)
    assert torch.isfinite(logits).all()
    assert logits[mask == 0].eq(-1e4).all()
