"""Tests for src.audit."""
import json
import os
import pytest


def test_audit_accepts_clean_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.audit import audit_single_model
    from src.checkpoint import save_checkpoint
    from configs.baseline import Config

    model = nn.Linear(4, 4)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"user_int": [], "item_int": [], "user_dense": [], "seq": {}}))
    ckpt = save_checkpoint(
        out_dir=str(tmp_path / "ck"), global_step=1, model=model, cfg=Config(),
        schema_path=str(schema_path), ns_groups_path=None,
        user_id_sample=["u0"], is_best=True,
    )
    rep = audit_single_model(ckpt)
    assert rep["n_tensors"] > 0
    assert "sha256" in rep


def test_audit_rejects_ema_in_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    from src.audit import audit_single_model
    os.makedirs(tmp_path / "ck", exist_ok=True)
    bad = {"w": torch.zeros(3), "ema_w": torch.zeros(3)}
    torch.save(bad, str(tmp_path / "ck" / "model.pt"))
    with pytest.raises(ValueError, match="forbidden"):
        audit_single_model(str(tmp_path / "ck"))
