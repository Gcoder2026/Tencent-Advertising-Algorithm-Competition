# Tencent Advertising Algorithm Competition 2026 — pCVR Track

Team workspace for the [Tencent Advertising Algorithm Competition 2026 (TAAC2026)](https://algo.qq.com/), the pCVR (post-click conversion-rate) prediction track co-hosted with KDD 2026.

The model is `PCVRHyFormer` — a hybrid backbone that jointly tokenises non-sequential user/item features and four cross-domain user behaviour sequences, then trains a single sigmoid head with binary cross-entropy. The platform ranks submissions on a single ROC-AUC metric, with a per-submission inference latency budget.

## Quick start

```bash
# 1. Clone
git clone https://github.com/Gcoder2026/Tencent-Advertising-Algorithm-Competition.git
cd Tencent-Advertising-Algorithm-Competition

# 2. Install dependencies (Python 3.13 recommended, matching the .pyc cache)
pip install torch pyarrow pandas scikit-learn numpy

# 3. Regenerate the 1k-row demo parquet (44 MB, not committed)
python tools/prepare_hf_sample.py \
    --out_dir data_sample_1000 \
    --row_group_size 200 \
    --download_if_missing

# 4. Smoke test the latest version on the demo (1 epoch, tiny model)
cd v2
bash ../v1/run_sample.sh   # uses run.sh from the version you cd'd into
```

## Folder layout

```
v1/                      # baseline as submitted on 2026-05-06
  dataset.py model.py train.py trainer.py utils.py infer.py
  run.sh run_sample.sh ns_groups.json
  submission_v1.zip      # platform upload (training side)
  evaluation_v1.zip      # platform upload (inference side)

v2/                      # next submission staging (transformer + temporal split)
  dataset.py model.py train.py trainer.py utils.py infer.py
  run.sh ns_groups.json
  submission_v2.zip      # ready to upload
  evaluation_v2.zip      # ready to upload

tools/
  prepare_hf_sample.py   # downloads + rewrites the HF demo parquet

data_sample_1000/
  schema.json            # feature layout reference (committed)
  demo_1000.parquet      # 44 MB, .gitignored — regenerate via tools/

.gitignore
README.md
```

Each `vN/` folder is a self-contained snapshot of the source for one platform submission. **Do not edit code in `v1/`** — it's the reference baseline. New work goes into a fresh `vN+1/` so we can diff and roll back per submission.

## The task in one paragraph

For every (user, item) row, predict whether the user will convert (`label_type == 2`). The training data has three label types — exposure (0), click (1), conversion (2) — and the positive class is severely imbalanced (~12% in the demo, expected lower at full scale). Each row carries:

- 5 ID/label columns (`user_id`, `item_id`, `label_type`, `label_time`, `timestamp`).
- 46 user int features (some scalar, some list-typed) + 10 user dense (float-array) embeddings.
- 14 item int features.
- 45 cross-domain behaviour sequences spread across 4 domains (`seq_a`, `seq_b`, `seq_c`, `seq_d`), each with its own timestamp column and side-info.

See [data_sample_1000/schema.json](data_sample_1000/schema.json) for the exact layout.

## Submission flow

Each submission is two zip files uploaded to the Tencent Angel platform:

| Zip | Contents | Used for |
|---|---|---|
| `submission_vN.zip` | `train.py`, `trainer.py`, `model.py`, `dataset.py`, `utils.py`, `run.sh`, `ns_groups.json` | Training container |
| `evaluation_vN.zip` | `infer.py`, `dataset.py`, `model.py`, `ns_groups.json` | Inference container (sandboxed, no network) |

The platform calls `infer.py:main()` with `MODEL_OUTPUT_PATH` / `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` env vars. Output is a `result.json` with one sigmoid score per (user_id, item_id) row.

**Limits**: 3 submissions per 24 h. Inference must finish in 30 min (1800 s). Submissions with ROC-AUC ≤ 0.50 don't show on the leaderboard. **Ensembling is forbidden** and is checked at Round-1→Round-2 promotion and at award verification.

## Round 1 timeline

- **Apr 24, 2026** — Round 1 opens
- **May 23, 2026 23:59:59 AOE** — Round 1 closes; top 50 academic / top 20 industry advance
- **May 25, 2026** — Round 2 starts on a ~10× larger dataset

## Improvement roadmap

The current plan lives in `~/.claude/plans/this-is-the-official-memoized-zebra.md` (local — not committed). High-level:

| Version | Status | Change | Goal |
|---|---|---|---|
| **v1** | Submitted | Baseline (`swiglu` encoder, row-group val split) | Establish floor |
| **v2** | Ready | `swiglu` → `transformer` sequence encoder; timestamp-sorted train/val split | First real attention; trustworthy local val |
| v3 | Pending | Longer sequences (`seq_*:256/512`) + RoPE | Capture longer-range dependencies |
| v4 | Pending | Focal loss + multi-task click head | Address class imbalance + free auxiliary signal |
| v5 | Pending | LR warmup/cosine + epoch-2 sparse re-init + mid-epoch eval | Training-loop hygiene |
| v6 | Pending | DIN-style target-attention query generator | Replace mean-pool query seeding |

Each version is one CLI/code change isolated for clean attribution; we **never** bundle multiple unrelated levers into one submission.

## Adding a new version

```bash
# 1. Stage from the previous version
cp -r v2 v3
cd v3

# 2. Make your one targeted change (e.g. edit run.sh or dataset.py)

# 3. Bundle the zips with the v1/v2 file layout
zip -j submission_v3.zip dataset.py model.py train.py trainer.py utils.py ns_groups.json run.sh
zip -j evaluation_v3.zip infer.py dataset.py model.py ns_groups.json

# 4. Smoke test, then commit
cd ..
git add v3
git commit -m "v3: <one-line description of the change>"
git push
```

## Key files to read first

- [v2/run.sh](v2/run.sh) — current training config and CLI flags
- [v2/dataset.py](v2/dataset.py) — parquet → tensor pipeline; sequence truncation; train/val split
- [v2/model.py](v2/model.py) — `PCVRHyFormer` architecture (NS tokenizer, sequence encoders, HyFormer blocks, RankMixer)
- [v2/trainer.py](v2/trainer.py) — training loop, BCE/Focal loss, AUC monitor, sparse-embedding re-init
- [v2/infer.py](v2/infer.py) — what the Angel platform actually executes during scoring

## Status

| | Score | Inference time |
|---|---|---|
| v1 (submitted) | ROC-AUC 0.806713 | 381.66 s / 1800 s |
| v2 (ready) | TBD | TBD |
