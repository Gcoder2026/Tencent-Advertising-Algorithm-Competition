# Tencent TAAC 2026 — PCVR Submission

Solution scaffold for the **Tencent KDD Cup 2026 PCVR (post-click conversion-rate)** challenge — Academic Track.

**Current leaderboard anchor:** AUC = **0.81144** (Step-1 baseline + Step-3 evaluation, single model, no ensembles).

## Repository layout

| Path | Purpose |
|---|---|
| [`pcvr/`](pcvr/) | **The project**. Layered Python code: configs/, src/, tests/, train.py, infer.py, build scripts. See [`pcvr/README.md`](pcvr/README.md) for the quickstart. |
| [`pcvr/ARCHITECTURE.md`](pcvr/ARCHITECTURE.md) | Decisions doc — reproducibility floor, single-model auditability, validation discipline, iteration infra. |
| [`docs/superpowers/plans/`](docs/superpowers/plans/) | Three written plans, in order: |
| ↳ `2026-05-08-pcvr-architecture-v0.md` | Initial scaffold build plan (executed). |
| ↳ `2026-05-08-pcvr-roadmap-v1.md` | 14-day Round-1 sprint roadmap (5-agent synthesized + critic-revised). |
| ↳ `2026-05-08-pcvr-merge-plan-v1.md` | Cross-team merge plan v1.1 (8 items applied locally). |

## Quickstart (local dev)

```bash
# Anaconda Python required (or any Python 3.10+ with the pinned deps)
conda create -n taac python=3.10 -y
conda activate taac
pip install -r pcvr/requirements.txt pytest

cd pcvr/
python -m pytest tests/      # 37/37 should pass

# Build submission zips
python build_step1_submission.py    # -> step1_train.zip
python build_step3_submission.py    # -> step3_infer.zip
```

## Submission flow (AngelML platform)

1. **Step 1 (Training Job):** upload `pcvr/step1_train.zip`. Platform runs `run.sh`, produces a `global_step*` checkpoint under `$TRAIN_CKPT_PATH`.
2. **Step 2 (Export Model):** in the AngelML UI, publish a `*.best_model` checkpoint.
3. **Step 3 (Model Evaluation):** upload `pcvr/step3_infer.zip`. Platform runs `infer.py`, writes `predictions.json` to `$EVAL_RESULT_PATH`. Leaderboard score returns ~3 hours after the 16:00 AOE cutoff.

## Single-model rule

The competition forbids any ensembling, weight averaging (SWA/EMA), multi-checkpoint averaging, multi-seed averaging, or stacking. This codebase enforces that explicitly:

- [`pcvr/src/checkpoint.py:_assert_single_state_dict`](pcvr/src/checkpoint.py) rejects state_dicts with keys starting with `ema_`, `swa_`, `shadow_`, `averaged_`, `polyak_`.
- [`pcvr/src/checkpoint.py:load_state_dict`](pcvr/src/checkpoint.py) refuses paths containing glob characters (no accidental "average all matching checkpoints" patterns).
- [`pcvr/src/optimizers.py:build_optimizers`](pcvr/src/optimizers.py) whitelists `{adamw, sgd}` (dense) + `{adagrad}` (sparse).
- [`pcvr/src/audit.py:audit_single_model`](pcvr/src/audit.py) returns SHA256 fingerprint of the state_dict for code-review reproducibility.
