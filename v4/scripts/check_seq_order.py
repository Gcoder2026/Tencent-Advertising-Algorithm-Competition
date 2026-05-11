"""Read one parquet RG from data_sample_1000 and per-domain check if seq
event timestamps are decreasing (newest-first) or increasing (oldest-first).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_sample_1000"
SCHEMA = json.loads((DATA / "schema.json").read_text())
parquet_files = sorted(DATA.glob("*.parquet"))
if not parquet_files:
    print("NO_PARQUET — run tools/prepare_hf_sample.py first")
    sys.exit(0)

pf = pq.ParquetFile(parquet_files[0])
for domain, cfg in SCHEMA["seq"].items():
    ts_fid = cfg.get("ts_fid")
    if ts_fid is None:
        print(f"{domain}: NO_TS_COLUMN")
        continue
    col = f"{cfg['prefix']}_{ts_fid}"
    tbl = pf.read_row_group(0, columns=[col])
    arrs = tbl.column(col).to_pylist()[:20]  # 20 rows
    decreasing, increasing, flat = 0, 0, 0
    for events in arrs:
        if not events or len(events) < 2:
            continue
        if events[0] > events[-1]:
            decreasing += 1
        elif events[0] < events[-1]:
            increasing += 1
        else:
            flat += 1
    direction = ("DECREASING (newest-first; head truncation correct)"
                 if decreasing > increasing else
                 "INCREASING (oldest-first; head truncation drops most-recent — FIX!)"
                 if increasing > decreasing else "MIXED/UNCLEAR")
    print(f"{domain}: dec={decreasing} inc={increasing} flat={flat} → {direction}")
