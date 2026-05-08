"""Pytest fixtures — synthetic flattened-parquet data + companion schema.json.

These fixtures let validators and the smoke test run without real data.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _build_schema(
    user_int_fids: List[tuple],
    item_int_fids: List[tuple],
    user_dense_fids: List[tuple],
    seq_cfg: Dict[str, dict],
) -> dict:
    return {
        "user_int": [list(t) for t in user_int_fids],
        "item_int": [list(t) for t in item_int_fids],
        "user_dense": [list(t) for t in user_dense_fids],
        "seq": seq_cfg,
    }


@pytest.fixture
def synth_data_root(tmp_path):
    """Builds a tiny synthetic flattened-parquet dataset.

    Returns a dict with keys:
      - data_dir: str (directory holding the parquet)
      - schema_path: str
      - n_rows: int
    """
    rng = np.random.default_rng(0)
    n = 50
    user_int_fids = [(1, 100, 1), (2, 50, 1)]              # fid, vocab, dim
    item_int_fids = [(11, 200, 1), (12, 300, 1)]
    user_dense_fids = [(61, 4)]                             # fid, dim
    seq_cfg = {
        "seq_a": {
            "prefix": "domain_a_seq",
            "ts_fid": 999,
            "features": [[201, 1000], [999, 0]],            # sideinfo + ts
        },
        "seq_b": {
            "prefix": "domain_b_seq",
            "ts_fid": 998,
            "features": [[202, 500], [998, 0]],
        },
    }
    schema = _build_schema(user_int_fids, item_int_fids, user_dense_fids, seq_cfg)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))

    cols: Dict[str, list] = {}
    cols["user_id"] = [f"u{i}" for i in range(n)]
    base_ts = 1_700_000_000
    cols["timestamp"] = [base_ts + i * 10 for i in range(n)]
    # Roughly 30% positive rate for sane label sanity check.
    cols["label_type"] = rng.choice([2, 1], size=n, p=[0.3, 0.7]).tolist()
    for fid, vocab, _ in user_int_fids:
        cols[f"user_int_feats_{fid}"] = rng.integers(1, vocab, size=n).tolist()
    for fid, vocab, _ in item_int_fids:
        cols[f"item_int_feats_{fid}"] = rng.integers(1, vocab, size=n).tolist()
    for fid, dim in user_dense_fids:
        cols[f"user_dense_feats_{fid}"] = [
            rng.standard_normal(dim).astype(np.float32).tolist() for _ in range(n)
        ]
    for domain, cfg in seq_cfg.items():
        prefix = cfg["prefix"]
        for fid, vocab in cfg["features"]:
            seq_len_per_row = rng.integers(5, 20, size=n)
            if vocab == 0:
                # timestamp column: monotonic offsets back from row ts
                cols[f"{prefix}_{fid}"] = [
                    [int(cols["timestamp"][i] - 60 * j) for j in range(seq_len_per_row[i])]
                    for i in range(n)
                ]
            else:
                cols[f"{prefix}_{fid}"] = [
                    rng.integers(1, vocab, size=int(seq_len_per_row[i])).tolist()
                    for i in range(n)
                ]

    table = pa.table(cols)
    parquet_path = tmp_path / "part-00000.parquet"
    pq.write_table(table, parquet_path, row_group_size=10)

    return {
        "data_dir": str(tmp_path),
        "schema_path": str(schema_path),
        "n_rows": n,
    }
