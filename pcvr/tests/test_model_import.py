"""Smoke test for src.model — verify imports work."""
import pytest


def test_model_imports():
    pytest.importorskip("torch")
    from src.model import PCVRHyFormer, ModelInput
    assert PCVRHyFormer is not None
    assert ModelInput is not None
