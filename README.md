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
  training_v1.zip        # platform upload (training side)
  evaluation_v1.zip      # platform upload (inference side)

v2/                      # next submission staging (transformer + temporal split)
  dataset.py model.py train.py trainer.py utils.py infer.py
  run.sh ns_groups.json
  training_v2.zip        # ready to upload
  evaluation_v2.zip      # ready to upload

pcvr/                    # alternative LAYERED architecture (now on main as of 2026-05-10)
  configs/{baseline,first_submission}.py
  src/{data,model,trainer,optimizers,checkpoint,audit,utils}.py
  tests/                 # 37 pytest tests, all passing on conda env
  tools/prepare_hf_sample.py + data_sample_1000/schema.json (sub-copy)
  scripts/check_seq_order.py
  train.py infer.py run.sh prepare.sh
  build_step{1,3}_submission.py
  README.md ARCHITECTURE.md requirements.txt pyproject.toml Makefile

docs/superpowers/plans/   # decision docs (on main as of 2026-05-10)
  2026-05-08-pcvr-architecture-v0.md   # build plan, executed
  2026-05-08-pcvr-roadmap-v1.md        # 14-day Round-1 sprint roadmap, critic-revised
  2026-05-08-pcvr-merge-plan-v1.md     # cross-team merge plan v1.1
  2026-05-09-pcvr-roadmap-v2.md        # post-merge roadmap (current; 13 days remaining)

tools/
  prepare_hf_sample.py   # downloads + rewrites the HF demo parquet

data_sample_1000/
  schema.json            # feature layout reference (committed)
  demo_1000.parquet      # 44 MB, .gitignored — regenerate via tools/

.gitignore
README.md
```

Each `vN/` folder is a self-contained snapshot of the source for one platform submission. **Do not edit code in `v1/`** — it's the reference baseline. New work goes into a fresh `vN+1/` so we can diff and roll back per submission.

The `pcvr/` folder is an alternative layered architecture with the same starter-kit ancestor, but split into `configs/`, `src/`, `tests/` for a more modular development style. Coexists with `vN/`; doesn't replace them. Now on `main` as of 2026-05-10 (was previously on a separate `pcvr-jw2333` branch; merged via fast-forward push).

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
| `training_vN.zip` | `train.py`, `trainer.py`, `model.py`, `dataset.py`, `utils.py`, `run.sh`, `ns_groups.json` | Training container |
| `evaluation_vN.zip` | `infer.py`, `dataset.py`, `model.py`, `ns_groups.json` | Inference container (sandboxed, no network) |

The platform calls `infer.py:main()` with `MODEL_OUTPUT_PATH` / `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` env vars. Output is a `predictions.json` with one sigmoid score per user_id (note: `pcvr/infer.py` writes this format; ensure all `vN/infer.py` use the platform-mandated `predictions.json` filename, not `result.json`).

**Limits**: 3 submissions per 24 h. Inference must finish in 30 min (1800 s). Submissions with ROC-AUC ≤ 0.50 don't show on the leaderboard. **Ensembling is forbidden** and is checked at Round-1→Round-2 promotion and at award verification.

## Round 1 timeline

- **Apr 24, 2026** — Round 1 opens
- **May 23, 2026 23:59:59 AOE** — Round 1 closes; top 50 academic / top 20 industry advance
- **May 25, 2026** — Round 2 starts on a ~10× larger dataset

## Improvement roadmap

The current plan lives in `~/.claude/plans/this-is-the-official-memoized-zebra.md` (local — not committed) and at `docs/superpowers/plans/` on `main`. The most recent committed plan is `2026-05-09-pcvr-roadmap-v2.md` (refreshed 2026-05-10). High-level:

| Version | Status | Change | Goal |
|---|---|---|---|
| **v1** | Submitted | Baseline (`swiglu` encoder, row-group val split) | Establish floor |
| **v2** | Ready | `swiglu` → `transformer` sequence encoder; timestamp-sorted train/val split | First real attention; trustworthy local val |
| **pcvr v0** | Submitted (on `main`) | Layered architecture port + bf16 + cosine+warmup + 4 wired validators + single-model audit | AUC **0.81144** (anchor) |
| pcvr v0.5 | Built; not yet submitted | Phase-1 merges: validators wired into `train.py`, `valid_ratio 0.05`, patience bumps, DDP-prefix strip | AUC TBD |
| v3 | Pending | Longer sequences (`seq_*:256/512`) + RoPE | Capture longer-range dependencies |
| v4 | Pending | Focal loss + multi-task click head | Address class imbalance + free auxiliary signal |
| v5 | Pending | LR warmup/cosine + epoch-2 sparse re-init + mid-epoch eval | Training-loop hygiene |
| v6 | Pending | DIN-style target-attention query generator | Replace mean-pool query seeding |

Each version is one CLI/code change isolated for clean attribution; we **never** bundle multiple unrelated levers into one submission.

## Single-model rule enforcement (`pcvr/`)

The competition forbids ensembling, weight averaging (SWA/EMA), multi-checkpoint averaging, multi-seed averaging, and stacking. The `pcvr/` codebase enforces this in code:

- `pcvr/src/checkpoint.py:_assert_single_state_dict` rejects state_dicts with keys starting with `ema_`, `swa_`, `shadow_`, `averaged_`, `polyak_`.
- `pcvr/src/checkpoint.py:load_state_dict` refuses paths containing glob characters (no accidental "average all matching checkpoints" patterns).
- `pcvr/src/optimizers.py:build_optimizers` whitelists `{adamw, sgd}` (dense) + `{adagrad}` (sparse).
- `pcvr/src/audit.py:audit_single_model` returns SHA256 fingerprint of the state_dict for code-review reproducibility.

Run `python -m pytest pcvr/tests/` to verify (37/37 pass on a torch-enabled env).

## Adding a new version

```bash
# 1. Stage from the previous version
cp -r v2 v3
cd v3

# 2. Make your one targeted change (e.g. edit run.sh or dataset.py)

# 3. Bundle the zips with the v1/v2 file layout
zip -j training_v3.zip dataset.py model.py train.py trainer.py utils.py ns_groups.json run.sh
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
- [pcvr/README.md](pcvr/README.md) — layered architecture quickstart
- [pcvr/ARCHITECTURE.md](pcvr/ARCHITECTURE.md) — design decisions doc
- [docs/superpowers/plans/2026-05-09-pcvr-roadmap-v2.md](docs/superpowers/plans/2026-05-09-pcvr-roadmap-v2.md) — current Round-1 plan (13 days remaining)

## Status

| | Score | Inference time |
|---|---|---|
| v1 (submitted 2026-05-06) | ROC-AUC 0.806713 | 381.66 s / 1800 s |
| v2 (ready) | TBD | TBD |
| pcvr v0 (submitted 2026-05-08, now on `main`) | **ROC-AUC 0.81144** (current anchor) | TBD |
| pcvr v0.5 (built 2026-05-09, not yet submitted) | TBD | TBD |
