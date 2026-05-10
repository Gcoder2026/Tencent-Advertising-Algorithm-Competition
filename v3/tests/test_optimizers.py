"""Tests for src.optimizers."""
import math
import pytest


def test_cosine_warmup_lambda_warmup_phase():
    from src.optimizers import cosine_warmup_lambda
    f = cosine_warmup_lambda(warmup_steps=100, total_steps=1000, min_lr_factor=0.1)
    assert f(0) == 0.0
    assert abs(f(50) - 0.5) < 1e-6
    assert abs(f(100) - 1.0) < 1e-6


def test_cosine_warmup_lambda_decay_phase():
    from src.optimizers import cosine_warmup_lambda
    f = cosine_warmup_lambda(warmup_steps=100, total_steps=1000, min_lr_factor=0.1)
    assert abs(f(550) - (0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 0.5)))) < 1e-6
    assert abs(f(1000) - 0.1) < 1e-6


def test_build_optimizers_rejects_unknown_dense():
    pytest.importorskip("torch")
    from src.optimizers import build_optimizers
    with pytest.raises(ValueError, match="dense_optimizer"):
        build_optimizers([], [], dense_optimizer="lion", sparse_optimizer="adagrad",
                         dense_lr=1e-4, sparse_lr=0.05)


def test_build_optimizers_rejects_unknown_sparse():
    pytest.importorskip("torch")
    from src.optimizers import build_optimizers
    with pytest.raises(ValueError, match="sparse_optimizer"):
        build_optimizers([], [], dense_optimizer="adamw", sparse_optimizer="adam",
                         dense_lr=1e-4, sparse_lr=0.05)


def test_build_optimizers_returns_pair_for_valid_combo():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.optimizers import build_optimizers
    dense = [nn.Linear(4, 4).weight]
    sparse = [nn.Embedding(10, 4).weight]
    dense_opt, sparse_opt = build_optimizers(
        dense_params=dense, sparse_params=sparse,
        dense_optimizer="adamw", sparse_optimizer="adagrad",
        dense_lr=1e-4, sparse_lr=0.05,
    )
    assert dense_opt is not None
    assert sparse_opt is not None
