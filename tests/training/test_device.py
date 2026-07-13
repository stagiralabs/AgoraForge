import pytest
import torch

from agoraforge.training.device import select_device


def test_cpu_device_is_explicit():
    assert select_device("cpu") == torch.device("cpu")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="cpu.*cuda"):
        select_device("gloo")


def test_unavailable_cuda_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        select_device("cuda")
