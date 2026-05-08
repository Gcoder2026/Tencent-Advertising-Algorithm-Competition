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


def test_label_rate_sane_passes_on_balanced(synth_data_root):
    from src.data import assert_label_rate_sane
    res = assert_label_rate_sane(synth_data_root["data_dir"], min_rate=0.001, max_rate=0.99)
    assert res["passed"] is True
    assert 0.001 < res["pos_rate"] < 0.99


def test_label_rate_sane_fails_when_all_negative(tmp_path):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    n = 100
    table = pa.table({
        "user_id": [f"u{i}" for i in range(n)],
        "timestamp": list(range(n)),
        "label_type": [1] * n,
    })
    p = tmp_path / "part-00000.parquet"
    pq.write_table(table, p, row_group_size=20)
    from src.data import assert_label_rate_sane
    with pytest.raises(ValueError, match="positive rate"):
        assert_label_rate_sane(str(tmp_path), min_rate=0.001, max_rate=0.99)


def test_oob_rate_check_passes_on_clean_stats():
    from src.data import oob_rate_check
    train_stats = {("user_int", 0): {"count": 5, "vocab": 100}}
    eval_stats = {("user_int", 0): {"count": 8, "vocab": 100}}
    n_rows = 1000
    oob_rate_check(eval_stats, train_stats, n_rows, threshold_abs=0.01, threshold_ratio=2.0)


def test_oob_rate_check_fails_on_spike():
    from src.data import oob_rate_check
    train_stats = {("user_int", 0): {"count": 5, "vocab": 100}}
    eval_stats = {("user_int", 0): {"count": 200, "vocab": 100}}
    n_rows = 1000
    with pytest.raises(ValueError, match="OOB"):
        oob_rate_check(eval_stats, train_stats, n_rows, threshold_abs=0.01, threshold_ratio=2.0)
