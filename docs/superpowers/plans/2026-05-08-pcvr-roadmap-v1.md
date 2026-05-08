# PCVR Roadmap (v1) — TAAC 2026 / KDD Cup Round 1

**Date drafted:** 2026-05-08 (T-15 days to Round 1 deadline 2026-05-23, AOE).
**Authors:** Synthesis of 5 specialist agents (AUC-leverage, iteration-speed, repro/risk, Round-2-readiness, strategic sequencer).
**Companion docs:** `docs/superpowers/plans/2026-05-08-pcvr-architecture-v0.md` (architecture plan, executed). `pcvr/ARCHITECTURE.md` (current decisions doc).

> Status legend: P0 (BLOCKER — before first submission), P1 (anchor — days 1-2), P2 (parallel iteration — days 3-10), P2.5 (Round 2 prep, additive), P3 (lock-in — days 13-14), Deferred (post-Round-1).

---

## Why this roadmap exists

v0 scaffold is done (~20 commits at `pcvr/`). It is **submitable in principle** but has 5 known gaps that should land **before** the first leaderboard submission to avoid wasting a 1/3-day slot on a recoverable bug. After P0, the team enters a 14-day sprint with 3 daily submissions to climb the leaderboard via parallel AUC-iteration streams. Round 2 prep is interleaved as additive low-risk work to avoid scrambling once the 10×-data window opens.

This roadmap is a **plan-of-record**, not a contract. Update it as ablation results land. Keep the structure (P0/P1/P2/P3) — it's how we'll triage daily.

---

## Phase 0 — Pre-submission housekeeping (must land before Submission #1)

These came up across multiple reviewers; all are <1h fixes and non-controversial.

### P0.1 — Wire `audit_single_model` into `build_submission.sh`
**Risk it closes:** Disqualification from a state-dict that silently picks up an `ema_*` / `swa_*` key. Currently the audit only runs in the smoke test and never gates the submission zip.
**Fix:** Add an `audit_single_model "$CKPT_DIR"` invocation in `build_submission.sh` before the `zip` step; non-zero exit aborts.
**Effort:** 0.5h. **Owner:** anyone.
**Files:** `pcvr/build_submission.sh`.

### P0.2 — Wire `oob_rate_check` + `sequence_history_leak_probe` into `train.py`
**Risk it closes:** Validation AUC inflated by leaked future events; code review fails to reproduce. Both validators exist in `src/data.py` and are unit-tested but never called from the entry point.
**Fix:** After `assert_label_rate_sane(...)` in `train.py`, add `sequence_history_leak_probe(...)` (log + soft warn unless n_future_events > threshold). After eval loops, pull `_oob_stats` off the dataset and pass to `oob_rate_check(...)`.
**Effort:** **4–6h** (revised — integration + debugging on real-torch shapes was understated; budget for one iteration of "trainer doesn't expose `_oob_stats` cleanly" surgery). **Owner:** the teammate handling validators.
**Files:** `pcvr/train.py`, `pcvr/src/trainer.py` (expose dataset's `_oob_stats` if not already accessible).

### P0.3 — `infer.py`: use `cfg.seed` instead of hardcoded `set_seed(0)`
**Risk it closes:** Drift at 4-5 decimals if any inference path is order-sensitive.
**Fix:** Move `set_seed(...)` after `Config(**cfg_dict)` and pass `cfg.seed`.
**Effort:** 0.25h.
**Files:** `pcvr/infer.py`.

### P0.4 — `submission_manifest.json` sidecar
**Risk it closes:** Code review asks "which run produced this checkpoint?" — `pcvr/runs/` (per-run JSON) is gitignored, so the binding between checkpoint and config is invisible to the reviewer. We need a tracked artifact.
**Fix:** At save-time in `src/checkpoint.save_checkpoint`, write `submission_manifest.json` alongside other sidecars containing: config_hash, git rev-parse HEAD, checkpoint sha256 (from `audit_single_model`), wall-clock time, train_data path. `build_submission.sh` includes it in the zip.
**Effort:** 1h. **Files:** `pcvr/src/checkpoint.py`, `pcvr/build_submission.sh`.

### P0.5 — `Makefile` shortcuts
**Risk it closes:** Submission scramble at 16:00 AOE because someone forgot a flag. Lower priority, but cheap.
**Fix:** Add `Makefile` with `make test`, `make smoke`, `make submit` (smoke = `pytest -m smoke`; submit = `audit && build_submission.sh`).
**Effort:** 0.5h. **Files:** new `pcvr/Makefile`.

### P0.6 — `predictions.json` dry-run on synthetic data (NEW — promoted from Day 2)
**Risk it closes:** A user_id-format mismatch between training and the platform's eval set burns the whole Day-2 submission window. The check needs to happen before we ever depend on it during a real submission.
**Fix:** Add `pcvr/scripts/dry_run_predictions.py` that runs `infer.py` end-to-end against the synthetic `synth_data_root` fixture, writes `predictions.json`, and asserts: (a) format matches `{"predictions": {str: float, ...}}`; (b) all keys are JSON-strings (not bytes); (c) all values are finite + within [0, 1]; (d) `_check_user_id_format` warns or passes cleanly. Wire as a pytest test marker so `make smoke` runs it.
**Effort:** 1h. **Files:** new `pcvr/scripts/dry_run_predictions.py`, edit `pcvr/Makefile`.

**P0 total effort:** **6–9h** (revised — Day 0 + part of Day 1; the original "one afternoon" estimate was wrong. Run P0.1, P0.3, P0.5 in parallel with the longer P0.2 / P0.4 / P0.6).

---

## Phase 1 — Anchor baseline (Days 1–2)

**Goal:** an honest leaderboard AUC on the default config, against which every later experiment is judged.

| Day | Teammate A (lead) | Teammate B | Teammate C |
|----|----|----|----|
| 1 morning | **Profile a 1-epoch dry-train on real torch** (full data path, small batch). Record wall-clock-per-step + estimated full-epoch and full-train durations. This is the single most important Day 1 deliverable: it tells us if Submission #1 is even feasible by Day 2. | Land P0.1 + P0.4 + P0.5. Branch hygiene: announce branch ownership in README. | Land P0.2 + P0.3 + P0.6. |
| 1 afternoon | Fix any shape/OOM errors in smoke or full-data path. Confirm validators run cleanly on real data. | Continue if anything is open. | Run `make smoke` end-to-end. Confirm dry-run predictions.json passes hygiene checks. |
| 2 morning (start ≤ 06:00 AOE) | **Start full baseline training** at first light. If Day-1 profiling showed > 6h training time, budget for an overnight run on Day 1→2 instead. | Stand by to assist if training hits an issue; otherwise begin reading Stream B literature. | Stand by to assist; otherwise begin reading Stream C literature. |
| 2 afternoon | **Submission #1 by 14:00 AOE** (4-hour buffer to 18:00 leaderboard cutoff). Build zip, run audit. If training is still in flight, default to a checkpoint at 12:00 AOE rather than waiting. | **Submission #2: same model, different seed** (variance check — gives us a noise estimate for Day-3 stop signals). | Submission #3: **held in reserve** for Day-1/2 emergency re-submit. Don't burn it on a variant. |

**Hard rules for Day 2:**
- Always have a usable checkpoint at 12:00 AOE — even if it's mid-epoch. Don't gamble on training finishing late.
- The `user_id` format check is already validated in P0.6; do NOT delay submission to "double-check" the format.
- If P0 work isn't fully landed by morning of Day 2, slip Submission #1 to Day 3 — submitting a broken zip is strictly worse than submitting a day late.

**Why baseline-then-stop on Day 1–2:** without an anchor, Day 3+ ablations have nothing to compare against. The seed-variance submission tells us how big a real "improvement" needs to be to beat noise.

**Critical path:** P0 → smoke + profiling pass → train baseline (start ≤ Day 2 06:00 AOE) → audit + zip → Submission #1 by 14:00 AOE Day 2.

---

## Phase 2 — Parallel AUC iteration (Days 3–10)

Three long-lived feature branches: `stream-A`, `stream-B`, `stream-C`. Merge to `main` only when an experiment beats the current best AUC. `pcvr/runs/` (per-run JSON) is the source of truth — appended on every training-end, PR-reviewed before the corresponding branch merges.

### Submission allocation (daily)
- **Sub #1 (12:00–14:00 AOE):** the current best-known config (never regress; this is our leaderboard floor).
- **Sub #2 (14:00–15:30 AOE):** the leading-stream challenger.
- **Sub #3 (15:30–16:00 AOE buffer):** an ablation, OR an emergency re-submit slot if Sub #1 or #2 has issues.

Why this order: the platform's 19:00 AOE refresh is 3h after the cutoff. If Sub #1 reveals a regression, you have time to investigate but not re-submit. So Sub #1 has to be the "safe" config.

### Stream A — Sequence & Temporal Encoding (Owner: Teammate A)

**Hypothesis:** PCVR signal is dominated by recency/frequency in user behavior sequences. Better temporal encoding + longer history = AUC.

**Experiments (in this order):**
1. **A.1 Sparse Embedding Re-init (KuaiShou MultiEpoch trick).** Was in the starter kit's `dafault file/trainer.py`; we dropped it in v0 because not implemented in our trainer. Re-add: at end of every epoch ≥ `cfg.reinit_sparse_after_epoch`, reinitialize embedding rows whose `vocab_size > cfg.reinit_cardinality_threshold` and rebuild Adagrad state. **Effort: 4–6h.** Expected gain: meaningful (literature reports 0.001–0.003 on similar systems). Risk: low (training-time only).
2. **A.2 Continuous log-delta time encoding.** Replace `BUCKET_BOUNDARIES`-based `searchsorted` time embedding with `MLP(log(Δt + 1))` projected to `d_model`, additive with the existing per-token embedding. **Effort: 3–5h.** Expected gain: meaningful. Risk: low (isolated to dataset + model time-bucket path).
3. **A.3 Extended seq_max_lens + gradient checkpointing.** Bump `seq_c` and `seq_d` to 768 or 1024; enable `torch.utils.checkpoint.checkpoint` on `MultiSeqHyFormerBlock.forward`. **Effort: 4–6h.** Expected gain: marginal-to-meaningful. Risk: medium (memory budget tight; must profile).
4. **A.4 Combined: best length + best time encoding.** Confirm the gains compose.

**Stop signal for stream A (revised — sharper):**
- **Day 4 EOD checkpoint:** if A.1 alone hasn't moved AUC by ≥ +0.001 over baseline (above seed noise from Day-2 variance check), pivot A.2 priority to embedding-dim sweep IMMEDIATELY rather than continuing the planned A.2/A.3 sequence.
- **Day 7 EOD checkpoint:** if A.1 + A.2 combined haven't moved AUC by ≥ +0.002, pivot Teammate A to embedding-dim / `num_queries` sweep for the rest of the sprint.

The original Day-7 +0.002 threshold burned 5 × 3 = 15 submission slots before any pivot decision. Keeping a cheaper Day-4 gate lets us recover faster from a sterile hypothesis.

### Stream B — Feature engineering & cross features (Owner: Teammate B)

**Hypothesis:** HyFormer attends across sequences but lacks explicit target × context products feeding the classifier. Hand-crafted crosses move AUC.

**Experiments:**
1. **B.1 User-level aggregate features:** click-rate, avg dwell, recency. Add as `user_dense_feats_*` columns at preprocessing time, route through `user_dense_proj`. **Effort: 4h.**
2. **B.2 Item-level aggregates:** CTR, position-bias correction. **Effort: 3h.**
3. **B.3 Target × context cross features:** explicit hadamard between target-item embedding and top user-profile embedding before the classifier head (modify `PCVRHyFormer.forward` near the classifier). **Effort: 4–6h.** Risk: medium — adds parameters; verify no regression.
4. **B.4 Continuous-feature normalization + bucketing.** Apply `log1p` + standardization to dense features; ablate.

**Stop signal:** if no single feature group moves AUC > +0.001, stop adding features (Teammate B pivots to support stream A or C).

### Stream C — Loss & optimizer tuning (Owner: Teammate C)

**Hypothesis:** Imbalanced labels and a fixed LR schedule leave AUC on the table.

**Experiments:**
1. **C.1 Focal loss γ sweep:** `cfg.loss_type="focal"` × γ ∈ {0.5, 1.0, 2.0} × α ∈ {0.1, 0.25}. **Effort: 1h per trial × 6 trials.**
2. **C.2 LR schedule ablation:** cosine vs. linear-decay vs. constant-after-warmup. **Effort: 1h per trial × 3.**
3. **C.3 Optimizer betas:** AdamW(0.9, 0.98) vs. (0.9, 0.999) vs. (0.95, 0.99). **Effort: 1h per trial.**
4. **C.4 Effective-batch via gradient accumulation:** 256 → 1024 effective. **Effort: 2h.**

**GPU scheduling:** the slice is single-tenant; concurrent training across streams is not possible. **Stream C batches its sweeps overnight** (kicks off C.1 at end of one workday, harvests by next morning) so it never competes with Streams A/B during business hours. Specifically: a chained `for cfg in [...]: train.py --config $cfg` shell loop. Stream C's own daytime is for analysis + writeups.

**Stop signal:** Day 10 hard cutoff regardless of progress — Stream C feeds final-config selection.

### Daily cadence (all streams)
- **09:00 standup (15 min):** share yesterday's last AUC + today's planned experiment.
- **EOD:** commit run row to `pcvr/runs/` (per-run JSON) (PR-reviewed), update branch.
- **Merge to main:** any experiment beating current best AUC by ≥ +0.001 (above seed noise).

### Branch & coordination strategy
- `main` = stable, last-best config.
- Each teammate keeps one long-lived branch: `stream-A`, `stream-B`, `stream-C`.
- `pcvr/configs/` — per-experiment config files: `configs/streamA_v3_long_seq.py`, etc. One-line `from configs.baseline import Config`. No edits to `configs/baseline.py` after Day 2.
- **Per-run JSON files in `pcvr/runs/` (revised — was `pcvr/runs/` (per-run JSON)).** A single CSV under three concurrent appenders + git merges WILL silently drop or scramble rows; we observed this pattern before. Switch to one JSON file per training run: `pcvr/runs/<utc_iso>_<config_hash>_<run_id>.json` containing the same fields as the CSV row plus the full `asdict(Config)` dump. The trainer writes its own file at end-of-training (no shared file). Use a tiny helper `scripts/aggregate_runs.py` to materialize the equivalent of the CSV on demand for analysis. `pcvr/runs/` is gitignored except for a `.gitkeep`.
- No force-pushes. No rebase of merged branches.

---

## Phase 2.5 — Round-2 prep (additive, slip into Days 5–10)

These do **not** target Round-1 AUC. They prevent panic when the May 25 / 10×-data window opens. All low-risk and additive.

### P2.5.1 — Latency benchmark script (`pcvr/scripts/bench_infer.py`)
**Round-2 problem:** Latency is the leaderboard tiebreaker. Without a baseline number we can't tell which optimization to invest in.
**Build now because:** zero model-file risk; ports unchanged to Round 2; informs every other optimization.
**Effort:** 2–3h. **Owner:** any teammate during a day with extra capacity.
**File:** new `pcvr/scripts/bench_infer.py`.

### P2.5.2 — `torch.no_grad` → `torch.inference_mode()` swap in `infer.py`
**Round-2 problem:** marginal latency at scale.
**Build now because:** 1-line change; verifiable in minutes.
**Effort:** 0.5h.

### ~~P2.5.3 — Sharded parquet streaming via `pyarrow.dataset`~~ — **DEFERRED to Round 1→2 transition**
**Why deferred:** This change touches `data.py` during Days 5–10, the busiest iteration window. A regression here has 3× blast radius — corrupts baseline reproducibility AND blocks all three streams. Round 2 opens May 25, and we have May 24 (1 day after Round 1 deadline) to do this swap before the 10×-data window opens. Better to ship Round 1 with the v0 dataset and use the gap day for the swap.
**Re-add to roadmap:** the day after Round-1 deadline (2026-05-24), as the first Round-2 prep task.

### P2.5.4 — Embedding-table pruning to top-N by frequency
**Round-2 problem:** 1M-vocab × 64-dim × fp32 = 256MB per table; the upload cap is 100MB. Round 2's larger vocabulary makes this worse.
**Build now because:** preprocessing script + config flag; conservative pruning (keep top 500K) verifiable on Round-1 data.
**Effort:** 3–5h. Risk: low–medium (validate AUC delta offline before committing).

### P2.5.5 — Gradient checkpointing on HyFormer blocks
Already covered as part of Stream A.3. Counts here too — Round 2's richer data will demand longer sequences.

---

## Phase 3 — Lock-in (Days 13–14)

| Step | Owner | Action |
|----|----|----|
| Pick best config | All | Sort `pcvr/runs/` (per-run JSON) by `best_val_auc`. Pick the highest with ≥ 2-seed stability (run the candidate config under a 2nd seed if not already done). |
| Retrain on best | A | Full retrain. Log SHA256 of the resulting `model.pt` in `submission_manifest.json`. |
| Audit | B | `audit_single_model`. Run all 4 validators. Verify smoke test still passes. |
| Build zip | C | `bash build_submission.sh`. **Manually inspect** the resulting zip's tree. Verify the `infer.py` filename + `def main()` signature via `python -c 'import infer; import inspect; print(inspect.signature(infer.main))'`. |
| **Submission #1 Day 13 by 12:00 AOE** | A | 4-hour buffer before cutoff (in case audit fails or zip is malformed). |
| Submission #2 Day 13 | A | Second-best config as insurance. |
| Day 14 | All | Sub #1 = best (re-submit if needed). Sub #2 = held in reserve. Sub #3 = absolute last-ditch. |

**Why not push to the deadline:** on Day 13, the platform's 19:00 leaderboard refresh tells us if our score landed. If it didn't, Day 14 has 3 fresh slots to fix it. Submitting on Day 14 final hour means a single zip-format bug ends the run.

---

## Critical path

```
P0 housekeeping (Day 0)
  ↓
Smoke test on real torch (Day 1, Teammate A)
  ↓
Baseline training (Day 2)
  ↓
Submission #1 → first leaderboard AUC anchor (Day 2)
  ↓                 ┌─→ Stream A: sparse re-init + time encoding + long seq
Parallel iteration  ├─→ Stream B: feature crosses
(Days 3–10)         └─→ Stream C: loss/optim tuning
  ↓
Day 10 EOD: stop exploring
  ↓
Day 11–12: best-config retrain + 2-seed stability check
  ↓
Day 13: final audit + zip + Submission by 12:00 AOE
```

**The one chain that blocks everything:** Smoke test must pass on Day 1. If it fails, Day 2 baseline submission slips, Day 3 streams have no anchor, and the whole sprint compresses. Mitigation: P0 work is teammate-parallel so smoke debugging on Teammate A doesn't block validators or build script work on B and C.

---

## Anti-overfitting: held-out local validation split

**The risk this section closes:** With 3 submissions/day × 14 days, the team will tune against the public leaderboard signal. Day 13's "best config" pick from the run log will systematically prefer the most overfit run unless we have an independent local check.

**The protocol:**
- During the v0 dataset construction (`get_pcvr_data`), the latest 10% of row groups becomes the validation split. Carve out an additional **2% holdout** from the train portion (NOT contiguous with the validation split — pick a different time slice, e.g. the row groups from 30–32% of the time range). Call this `local_holdout`.
- **Never train on local_holdout.** Never tune any hyperparameter against it. It is consulted ONLY at lock-in (Day 12).
- Trainer logs `local_holdout_auc` to the run JSON alongside `best_val_auc`.
- At lock-in, the candidate set is the top-5 configs by `best_val_auc` from `pcvr/runs/`. Pick whichever of those has the **highest `local_holdout_auc`** — not the highest `best_val_auc`. If they disagree by > 0.005, that's evidence of validation-split overfitting and we should run a 2nd seed before final retrain.
- Wire this in P0 alongside P0.2 (cost: 1–2h on top of the validator wiring).

**Why this matters:** the same critic who flagged this points out that without it, day-by-day leaderboard chasing is exactly the wrong optimization process for "best generalization config." This single discipline costs 2h once and prevents the most likely Day-13 regret.

---

## Top 4 risks + mitigations

1. **Smoke test fails on real torch (Day 1).** Synthetic data may violate model constraints (e.g., `d_model % T == 0` where T = num_queries × num_seq_domains + num_ns). *Mitigation:* the smoke test config can be adjusted, NOT the source. If unresolved by Day 1 EOD, submit baseline with minimal config (`num_queries=1`, `d_model=64`) to anchor and continue debugging.

2. **No stream moves AUC significantly by Day 7.** *Mitigation:* mid-sprint review on Day 7. If all three streams are < +0.005 over baseline, pivot one teammate to a focused embedding-dim/depth sweep (`d_model ∈ {64, 96, 128}`, `num_hyformer_blocks ∈ {2, 3, 4}`), which historically is the lever that moves PCVR systems when feature/loss tweaks plateau.

3. **`pcvr/runs/` JSON-per-run data corruption / config drift.** Mostly mitigated by per-run files (one writer per file), but cross-branch races on git pulls can still cause issues. *Mitigation:* each JSON includes `config_hash` + git SHA + `submission_manifest.json` reference. Teammate C owns periodic `aggregate_runs.py` runs to spot anomalies.

4. **Leaderboard overfitting.** With 3 submissions/day × 14 days, daily tuning systematically prefers the most overfit run. *Mitigation:* the held-out local-validation protocol described above. Day-13 lock-in selects on `local_holdout_auc` first, leaderboard rank second.

---

## What we're deliberately NOT doing

- **Hydra / MLflow / W&B integration.** Too heavy for 14 days. `pcvr/runs/` (per-run JSON) + TensorBoard subdirs cover the need.
- **Model-file split.** Port stays monolithic until a 2nd model variant lands — premature splitting introduces import-cycle bugs in week 1.
- **Random hyperparameter sweep harness.** Tempting but the GPU is single-slice. Manual streaming with `pcvr/runs/` (per-run JSON) is faster than building a sweep tool, given the timeline.
- **Notebook directory.** EDA happens in `scripts/` if at all. Reconsider for Round 2.
- **Heavy feature engineering pipelines.** Stream B will add a few crosses; nothing that requires a separate preprocessing system.

---

## Open questions to revisit weekly

- Does focal loss actually help on this dataset, or is the positive rate high enough that BCE wins? (Stream C answers this.)
- Is the rankmixer NS tokenizer (5 user / 2 item tokens) the right choice, or does the GroupNSTokenizer with `ns_groups.json` give better AUC? (Brought up by AUC analyst — worth one experiment in Stream A or B.)
- Per-position vocab sizes are collapsed to `max(...)` in `_build_feature_specs` — does fixing this matter?
- Round 2 latency tiebreaker — what's our current end-to-end inference latency on a typical eval row? (P2.5.1 answers this.)

---

## Update protocol

Update this file directly when:
- A P0 item lands (mark ✅ with commit SHA).
- A stream stop-signal fires (note pivot decision).
- An open question is answered (move to `pcvr/ARCHITECTURE.md`).

Don't update for routine experiment results — those go in `pcvr/runs/` (per-run JSON).

— end —
