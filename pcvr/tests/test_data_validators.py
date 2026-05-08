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


def test_time_split_monotonic_passes_on_sorted_data(synth_data_root):
    from src.data import assert_time_split_monotonic
    res = assert_time_split_monotonic(
        synth_data_root["data_dir"], valid_ratio=0.2,
    )
    assert res["passed"] is True
    assert res["train_max_ts"] <= res["valid_min_ts"]


def test_time_split_monotonic_fails_on_unsorted(tmp_path):
    """Build an unsorted parquet and assert the validator raises."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 30
    rng = np.random.default_rng(0)
    ts = rng.permutation(np.arange(1700000000, 1700000000 + n).astype(np.int64))
    user_id = [f"u{i}" for i in range(n)]
    label_type = [2] * n
    table = pa.table({"user_id": user_id, "timestamp": ts.tolist(), "label_type": label_type})
    p = tmp_path / "part-00000.parquet"
    pq.write_table(table, p, row_group_size=10)

    from src.data import assert_time_split_monotonic
    with pytest.raises(ValueError, match="not monotonic"):
        assert_time_split_monotonic(str(tmp_path), valid_ratio=0.3)
