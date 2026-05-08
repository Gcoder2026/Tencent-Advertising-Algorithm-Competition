"""Tests for src.data — port smoke + validators."""
import json
import pytest


def test_dataset_iterates_synthetic_data(synth_data_root):
    pytest.importorskip("torch")
    from src.data import PCVRParquetDataset
    ds = PCVRParquetDataset(
        parquet_path=synth_data_root["data_dir"],
        schema_path=synth_data_root["schema_path"],
        batch_size=8,
        shuffle=False,
        buffer_batches=0,
        is_training=True,
    )
    batches = list(ds)
    assert len(batches) > 0
    b = batches[0]
    assert "user_int_feats" in b
    assert "item_int_feats" in b
    assert "user_dense_feats" in b
    assert "label" in b
    assert "user_id" in b
    assert "_seq_domains" in b
    for d in b["_seq_domains"]:
        assert d in b
        assert f"{d}_len" in b
        assert f"{d}_time_bucket" in b


def test_dataset_inference_zeros_labels(synth_data_root):
    pytest.importorskip("torch")
    from src.data import PCVRParquetDataset
    ds = PCVRParquetDataset(
        parquet_path=synth_data_root["data_dir"],
        schema_path=synth_data_root["schema_path"],
        batch_size=8,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    b = next(iter(ds))
    assert int(b["label"].sum()) == 0
