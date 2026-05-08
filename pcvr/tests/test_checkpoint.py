"""Tests for src.checkpoint."""
import json
import os
import shutil
from dataclasses import asdict
import pytest


def test_build_step_dir_name_uses_global_step_prefix():
    from src.checkpoint import build_step_dir_name
    name = build_step_dir_name(2500, layer=2, head=4, hidden=64, is_best=False)
    assert name.startswith("global_step")
    assert "global_step2500" in name


def test_save_load_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.checkpoint import save_checkpoint, load_state_dict
    from configs.baseline import Config

    model = nn.Sequential(nn.Linear(4, 4))
    cfg = Config(d_model=8)
    schema = {"user_int": [], "item_int": [], "user_dense": [], "seq": {}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))

    ckpt_dir = save_checkpoint(
        out_dir=str(tmp_path / "ckpts"), global_step=10, model=model, cfg=cfg,
        schema_path=str(schema_path), ns_groups_path=None,
        user_id_sample=["u0", "u1", "u2"], is_best=True,
    )
    assert os.path.isfile(os.path.join(ckpt_dir, "model.pt"))
    assert os.path.isfile(os.path.join(ckpt_dir, "schema.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "train_config.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "user_id_sample.json"))

    sd = load_state_dict(ckpt_dir)
    assert all(k in dict(sd).keys() for k in dict(model.state_dict()).keys())


def test_save_refuses_ema_keys(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.checkpoint import _assert_single_state_dict
    bad = {"weight": torch.zeros(3), "ema_weight": torch.zeros(3)}
    with pytest.raises(ValueError, match="forbidden"):
        _assert_single_state_dict(bad)


def test_load_refuses_globs(tmp_path):
    pytest.importorskip("torch")
    from src.checkpoint import load_state_dict
    with pytest.raises(ValueError, match="single"):
        load_state_dict(str(tmp_path / "global_step*"))
