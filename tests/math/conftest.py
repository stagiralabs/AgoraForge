from __future__ import annotations

import torch
import pytest


class FakeMechanism:
    def __init__(self, update=None):
        self.update = update

    def step(self, s, a, e):
        s_next = s if self.update is None else self.update(s, a, e)
        return s_next, torch.zeros(s.shape[:-1], dtype=s.dtype, device=s.device)


@pytest.fixture
def fake_mechanism_cls():
    return FakeMechanism
