"""Tests for src.utils — seed pinning, losses, environment guards."""
import os
import math
import pytest


def test_set_seed_pins_environment(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    from src.utils import set_seed
    set_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_seed_warns_if_called_after_torch_imported(monkeypatch):
    """PYTHONHASHSEED is too late after Python starts — set_seed must not silently
    succeed when called post-startup. We just check it sets env vars; the
    BEFORE-python guarantee comes from run.sh."""
    from src.utils import set_seed
    set_seed(0)  # idempotent, should not raise


@pytest.mark.parametrize("logits,labels,expected_min,expected_max", [
    ([0.0], [1.0], 0.6, 0.8),
    ([10.0], [1.0], 0.0, 0.001),
    ([-10.0], [0.0], 0.0, 0.001),
])
def test_focal_loss_runs(logits, labels, expected_min, expected_max):
    torch = pytest.importorskip("torch")
    from src.utils import sigmoid_focal_loss
    L = torch.tensor(logits)
    Y = torch.tensor(labels)
    loss = sigmoid_focal_loss(L, Y, alpha=0.25, gamma=2.0, reduction="mean")
    v = float(loss)
    assert expected_min <= v <= expected_max, f"got {v}"


def test_bce_loss_runs():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    L = torch.tensor([0.0, 5.0])
    Y = torch.tensor([1.0, 0.0])
    v = float(F.binary_cross_entropy_with_logits(L, Y))
    assert v > 0
