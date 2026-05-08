# PCVR Architecture (v0)

Last updated: 2026-05-08.

## Goals
- Top leaderboard finish on Tencent KDD Cup 2026 PCVR (Academic Track).
- Reproducible end-to-end (code review on best submission must reproduce).
- Single-model rule auditable: no EMA/SWA/multi-checkpoint averaging.
- Iteration cadence: >= 5 distinct experiments per day, ablations cheap.

## Layered separation
- `configs/` — typed Python `@dataclass` configs. One experiment = one file.
- `src/data.py` — PCVRParquetDataset + 4 validators (time-split monotonicity,
  label-rate sanity, OOB rate guard, sequence-history leak probe).
- `src/model.py` — PCVRHyFormer (port of starter kit, monolithic for v0).
- `src/optimizers.py` — whitelisted optimizer factory + cosine+warmup scheduler.
- `src/checkpoint.py` — save/load + sidecars; refuses globs + ensemble-key prefixes.
- `src/audit.py` — single-model rule audit (CI + pre-zip).
- `src/trainer.py` — training loop, bf16 default, resume, registry append.
- `src/utils.py` — set_seed (full RNG + env pin), losses, EarlyStopping.
- `train.py`, `infer.py` — thin platform entries.
- `run.sh` — exports `PYTHONHASHSEED` + `CUBLAS_WORKSPACE_CONFIG` BEFORE python.
- `prepare.sh` — no-op (platform image complete).
- `build_submission.sh` — deterministic zip; excludes pycache/docs/tests.

## Reproducibility floor
- `set_seed` pins random/numpy/torch (CPU + CUDA), `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`,
  env `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `run.sh` exports the env vars BEFORE python starts (Python-side fallback only
  helps subprocesses).
- Inference (`infer.py`) uses `num_workers=0`, no AMP, deterministic ordering.

## Single-model auditability
- `checkpoint._assert_single_state_dict` rejects keys starting with
  `ema_`, `swa_`, `shadow_`, `averaged_`, `polyak_`.
- `checkpoint.load_state_dict` refuses paths with glob characters.
- `optimizers.build_optimizers` whitelists `{adamw, sgd}` (dense) +
  `{adagrad}` (sparse) — no SWA/SAM/Lookahead wrappers.
- `audit_single_model(ckpt_dir)` returns SHA256 of the param fingerprint and
  is run from `tests/test_smoke.py` and `build_submission.sh`.

## predictions.json hygiene
- Schema for inference comes ONLY from the checkpoint's bundled `schema.json`
  (never falls back to eval-data dir).
- `user_id_sample.json` sidecar is checked at infer time; format mismatch
  (digit vs non-digit) raises before submission.
- Per-user MEAN aggregation; logs warning on >1 rows per user.
- Scores clipped to `[1e-7, 1-1e-7]`; `json.dump(..., allow_nan=False)`.
- Empty eval set → hard error.

## Validation discipline (in v0)
- `assert_time_split_monotonic` runs at the start of `train.py`. Fails if
  the row-group time order doesn't support the tail-split.
- `assert_label_rate_sane` checks the head 100k rows; positive rate must
  be in `[0.001, 0.10]`.
- `oob_rate_check` and `sequence_history_leak_probe` are exposed for
  ad-hoc use; not yet wired into the training entry.

## Iteration infrastructure
- bf16 mixed precision in trainer (default ON; toggle via `Config.use_bf16`).
- Cosine + linear warmup LR schedule, cfg-driven warmup_steps + min_lr_factor.
- Resume support: trainer reads `train_state.pt` if `cfg.resume_from` is set;
  bundles dense_opt + sparse_opt + RNG state.
- Experiment registry: `experiments.csv` appended on training end with
  `(run_id, config_hash, best_val_auc, best_val_logloss, global_step,
  seed, ckpt_path, wall_clock_min, timestamp, notes)`.

## Open items / v0.5 candidates
- Continuous log-delta time encoding (replace 64-bucket Embeddings).
- Longer per-domain `seq_max_lens` with gradient checkpointing.
- Wire `oob_rate_check` and `sequence_history_leak_probe` into `train.py`'s
  pre-training validation gate.
- Per-position vocab handling (currently `max(...)` collapses heterogeneous
  positions).
- Round-2 latency posture: profile, consider `torch.inference_mode`.
- Notebook directory for ad-hoc EDA.

## Deliberately deferred
- Hydra / config composition (overkill for 14 days).
- Plugin/registry layer for model variants (add when 2nd variant lands).
- Splitting `model.py` (port monolithic; split when an arch swap arrives).
- W&B / MLflow integration (CSV registry sufficient for v0).
