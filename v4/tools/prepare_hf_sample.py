#!/usr/bin/env python3
"""Prepare the official Hugging Face 1k demo sample for local smoke tests.

The competition platform provides ``schema.json`` next to the official Parquet
data. The Hugging Face demo only ships ``demo_1000.parquet``, so this script
infers a sample-only schema and rewrites the Parquet file with multiple Row
Groups. That keeps the local data path compatible with ``train.py`` without
changing the official training contract.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


HF_SAMPLE_URL = (
    "https://huggingface.co/datasets/TAAC2026/data_sample_1000/"
    "resolve/main/demo_1000.parquet"
)

USER_INT_RE = re.compile(r"^user_int_feats_(\d+)$")
ITEM_INT_RE = re.compile(r"^item_int_feats_(\d+)$")
USER_DENSE_RE = re.compile(r"^user_dense_feats_(\d+)$")
SEQ_RE = re.compile(r"^domain_([a-z])_seq_(\d+)$")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open("wb") as out_f:
        shutil.copyfileobj(response, out_f)


def _is_list_type(dtype: pa.DataType) -> bool:
    return pa.types.is_list(dtype) or pa.types.is_large_list(dtype)


def _iter_non_null_values(column: pa.ChunkedArray) -> Iterable[Any]:
    for value in column.to_pylist():
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    yield item
        else:
            yield value


def _max_positive_int(column: pa.ChunkedArray) -> int:
    max_value = 0
    for value in _iter_non_null_values(column):
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value > max_value:
            max_value = int_value
    return max_value


def _max_list_len(column: pa.ChunkedArray) -> int:
    max_len = 0
    for value in column.to_pylist():
        if value is not None:
            max_len = max(max_len, len(value))
    return max_len


def _positive_sample(column: pa.ChunkedArray, limit: int = 512) -> list[int]:
    values: list[int] = []
    for value in _iter_non_null_values(column):
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value > 0:
            values.append(int_value)
            if len(values) >= limit:
                break
    return values


def _infer_time_fid(table: pa.Table, seq_cols: list[tuple[int, str]]) -> int | None:
    """Pick the timestamp-like sequence feature for one domain.

    In the released sample these columns contain Unix-second values around
    1.7e9, while ordinary categorical ids are much smaller or not consistently
    timestamp-shaped.
    """
    best_fid: int | None = None
    best_score = -1
    for fid, name in seq_cols:
        values = _positive_sample(table[name])
        if not values:
            continue
        large_count = sum(1 for value in values if value >= 1_000_000_000)
        if large_count > best_score:
            best_score = large_count
            best_fid = fid
    if best_score <= 0:
        return None
    return best_fid


def infer_schema(table: pa.Table) -> dict[str, Any]:
    user_int: list[list[int]] = []
    item_int: list[list[int]] = []
    user_dense: list[list[int]] = []
    seq_by_domain: dict[str, list[tuple[int, str]]] = {}

    for field in table.schema:
        name = field.name
        dtype = field.type

        match = USER_INT_RE.match(name)
        if match:
            fid = int(match.group(1))
            dim = _max_list_len(table[name]) if _is_list_type(dtype) else 1
            dim = max(dim, 1)
            vocab_size = _max_positive_int(table[name]) + 1
            user_int.append([fid, max(vocab_size, 1), dim])
            continue

        match = ITEM_INT_RE.match(name)
        if match:
            fid = int(match.group(1))
            dim = _max_list_len(table[name]) if _is_list_type(dtype) else 1
            dim = max(dim, 1)
            vocab_size = _max_positive_int(table[name]) + 1
            item_int.append([fid, max(vocab_size, 1), dim])
            continue

        match = USER_DENSE_RE.match(name)
        if match:
            fid = int(match.group(1))
            dim = _max_list_len(table[name]) if _is_list_type(dtype) else 1
            user_dense.append([fid, max(dim, 1)])
            continue

        match = SEQ_RE.match(name)
        if match:
            domain_letter = match.group(1)
            fid = int(match.group(2))
            seq_by_domain.setdefault(f"seq_{domain_letter}", []).append((fid, name))

    seq: dict[str, Any] = {}
    for domain in sorted(seq_by_domain):
        seq_cols = sorted(seq_by_domain[domain])
        letter = domain.split("_", 1)[1]
        ts_fid = _infer_time_fid(table, seq_cols)
        features = []
        for fid, name in seq_cols:
            vocab_size = _max_positive_int(table[name]) + 1
            features.append([fid, max(vocab_size, 1)])
        seq[domain] = {
            "prefix": f"domain_{letter}_seq",
            "ts_fid": ts_fid,
            "features": features,
        }

    return {
        "user_int": sorted(user_int),
        "item_int": sorted(item_int),
        "user_dense": sorted(user_dense),
        "seq": seq,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TAAC2026 Hugging Face demo data for local runs."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/private/tmp/taac_demo_1000.parquet"),
        help="Path to an existing demo_1000 parquet file.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data_sample_1000"),
        help="Output directory containing demo_1000.parquet and schema.json.",
    )
    parser.add_argument(
        "--row_group_size",
        type=int,
        default=200,
        help="Rows per output Row Group; use >0 so train/valid splitting works.",
    )
    parser.add_argument(
        "--download_if_missing",
        action="store_true",
        help="Download the official HF demo parquet if --source is missing.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"{args.source} not found. Pass --download_if_missing or "
                "download demo_1000.parquet from Hugging Face first."
            )
        _download(HF_SAMPLE_URL, args.source)

    table = pq.read_table(args.source)
    schema = infer_schema(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = args.out_dir / "demo_1000.parquet"
    out_schema = args.out_dir / "schema.json"

    pq.write_table(table, out_parquet, row_group_size=args.row_group_size)
    with out_schema.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    pf = pq.ParquetFile(out_parquet)
    label_counts: dict[int, int] = {}
    if "label_type" in table.column_names:
        for value in table["label_type"].to_pylist():
            label_counts[int(value)] = label_counts.get(int(value), 0) + 1

    print(f"Wrote {out_parquet}")
    print(f"Wrote {out_schema}")
    print(f"Rows: {table.num_rows}, columns: {table.num_columns}")
    print(f"Row groups: {pf.metadata.num_row_groups}")
    print(f"label_type counts: {label_counts}")
    print(
        "Schema groups: "
        f"user_int={len(schema['user_int'])}, "
        f"item_int={len(schema['item_int'])}, "
        f"user_dense={len(schema['user_dense'])}, "
        f"seq_domains={list(schema['seq'].keys())}"
    )
    for domain, cfg in schema["seq"].items():
        print(
            f"{domain}: {len(cfg['features'])} features, "
            f"ts_fid={cfg['ts_fid']}"
        )


if __name__ == "__main__":
    main()
