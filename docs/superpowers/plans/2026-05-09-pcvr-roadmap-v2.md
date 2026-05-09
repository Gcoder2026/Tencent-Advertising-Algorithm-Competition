# PCVR Roadmap v2 — post-merge reality (2026-05-09)

**Supersedes:** `2026-05-08-pcvr-roadmap-v1.md` (the original 14-day sprint roadmap, written assuming a 3-person team with sole project ownership). Roadmap v1's v0 anchor + 5-agent specialist findings + critic-integrated revisions remain valid where not contradicted here.

**Updated context (vs. v1):**
- **Today: 2026-05-09.** Round 1 deadline: 2026-05-23 AOE → **14 days remaining**.
- **Team-size correction:** roadmap v1 assumed 3 students × 3 streams. Actual: **you alone + one teammate (Gcoder2026) working in parallel on a separate branch.** Stream A/B/C breakdown collapses to one-stream-at-a-time for you, with the teammate as an independent contributor.
- **Cross-team flow established:** local `main` → push → `jw2333-null/taac-pcvr` (private solo). Local `pcvr-jw2333` → push → `Gcoder2026/.../pcvr-jw2333` (merged, visible to teammate). See [`pcvr/ARCHITECTURE.md`](../../../pcvr/ARCHITECTURE.md) → "Branches" section if you want me to add one.
- **Phase 1 of merge plan v1.1 applied** locally as 8 commits (37/37 tests pass on conda env `taac`). Ready-to-submit v0.5 zips archived under [`submissions/2026-05-09_v0.5_post-phase1/`](../../../submissions/2026-05-09_v0.5_post-phase1/) — **not yet submitted.**
- **Anchor scores:** your v0 = AUC **0.81144** (leaderboard). teammate's v1 = AUC 0.807 (per their README; submission filename was wrong, anchor is suspect — see merge plan v1.1).
- **Generalization gap diagnosis pending:** local valid AUC ~0.86 vs leaderboard 0.81144 = ~0.05 gap. Most likely sequence-history leakage; the wired `sequence_history_leak_probe` validator in `train.py` will report on next training run.

---

## What changes vs. v1

| Roadmap v1 assumption | v2 correction |
|---|---|
| 3 streams (A/B/C) running in parallel by 3 students | One stream at a time for you. Teammate contributes independently to their own `vN/` snapshots; you can opportunistically port single high-value items from each new vN they produce. |
| `experiments.csv` shared by 3 appenders | Single appender (you) on local main, plus teammate's separate registry on their branch. CSV is fine — corruption-via-concurrency risk gone. |
| `submissions/` was unspecified | Now structured at top level with one folder per submission attempt; see `submissions/README.md`. |
| Phase 2 ablations described abstractly | v2 commits to specific config files to write next. |

---

## Phase A — Anchor diagnosis (~3-4 days; **HIGH leverage**)

The 0.05 gen gap is the most expensive open question. Resolve before spending more submission slots.

### A1. Submit `v0.5_post-phase1` to capture leak-probe diagnostic
**What:** Upload `submissions/2026-05-09_v0.5_post-phase1/step1_train.zip`. Watch the platform log for `sequence-leak probe: {n_future_events: ..., n_sampled: ...}` and `OOB rate guard ...`.
**Why:** This is the cheapest diagnostic available. You learn (a) whether v0.5's micro-changes (valid_ratio 0.05, patience bumps, validator wiring) move the leaderboard AUC, AND (b) the actual leak-probe count on real data. If `n_future_events > 0` we have direct evidence of the sequence leakage hypothesis.
**Cost:** 1 submission slot.
**Expected outcome:** AUC within ±0.005 of 0.81144. Real value is the diagnostic, not a score bump.

### A2. Aggregation ablation on the existing checkpoint
**What:** Modify `pcvr/infer.py` to support `--aggregation {mean, max, last_by_ts}` (currently mean is hardcoded). Rebuild Step-3 zip × 2 variants. Submit each against the **same** published checkpoint.
**Why:** If the test set has multiple rows per user, the aggregation choice can move AUC by 0.005-0.020 without retraining. Burns 2 Step-3 slots but reuses the published checkpoint (no Step-1 re-run needed).
**Cost:** 2 submission slots.
**Effort:** 1h code + 1h zip-build + waiting for grader.

### A3. Local capacity diagnostic (NO submission slots)
**What:** Write three configs `pcvr/configs/diag_emb{32,64,128}.py`. Train each locally on the HF sample for 4k steps. Compare local valid AUC across the three.
**Why:** The 4k-step plateau in your previous training plot is the strongest signal we have that the model is capacity-bound (or not). If doubling `emb_dim` raises AUC, focus subsequent work on architecture (Phase B). If not, focus on data/features (Phase C).
**Cost:** 0 submission slots. ~3 GPU hours local (or local CPU-time if you don't have GPU access — slower but free).
**Effort:** 30 min config writing + GPU-time waiting.

**Phase A budget: 3 submission slots, 4 hours coding/waiting.**

---

## Phase B — Architecture levers (~5-6 days; gated on Phase A result)

If Phase A reveals capacity-bound:

### B1. `seq_encoder_type='transformer'` ablation
Already supported in `pcvr/configs/baseline.py` (`_ALLOWED_SEQ_ENCODERS = {"swiglu", "transformer", "longer"}`). Single config change. Run **locally first**; submit only if local AUC ≥ 0.81144 + 0.002.
**Cost:** 0-1 slots. **Effort:** 5 min config + GPU time.

### B2. `use_rope=True` ablation
RoPE is implemented in `pcvr/src/model.py:RoPEMultiheadAttention` but disabled by default. Toggle on. Same submit-rule as B1.
**Cost:** 0-1 slots.

### B3. Cold-restart sparse re-init (KuaiShou MultiEpoch)
**Conditional:** only do this if Phase A shows training completes ≥ 3 epochs before patience-based early stopping. The re-init fires at epoch boundaries; with 1-2 epoch runs it never executes. The teammate's `v2/trainer.py` has the implementation — port it back into our `src/trainer.py` (~3-4h).

### B4. Capacity bump
If Phase A's `emb_dim=128` beat `emb_dim=64` by ≥ 0.005 locally, build `pcvr/configs/v1_capacity.py` with `d_model=128, emb_dim=128`. Verify the `d_model % T == 0` constraint passes with the new T value.

---

## Phase C — Data + signal levers (~3-4 days; gated on Phase A result)

If Phase A shows the bottleneck is data/signal:

### C1. DIN-style target-attention query
Use the target item embedding as the cross-attention query for user-sequence attention, instead of the current learned NS tokens. Architectural change in `pcvr/src/model.py` (~5h, medium risk). High AUC ceiling in DIEN/DIN literature.

### C2. Multi-task click head (CTR auxiliary loss)
**Prerequisite:** verify the platform's data has a click label column (the `label_type` field has values 0/1/2 — type 1 may be a click event). Add a sigmoid head + auxiliary BCE loss with a small weight (e.g., 0.1×). ~4h.

### C3. Continuous log-Δt time encoding
Replace `BUCKET_BOUNDARIES`-based time embedding with `MLP(log(Δt + 1))`. ~3h. Was deferred from roadmap v1 → v2 candidates.

### C4. Cross features (target × top user-profile)
Hadamard product of target-item embedding and top user dense features, fed into the classifier head. ~4h.

---

## Phase D — Round-2 prep (interleave during Phase B/C; additive only)

Same as roadmap v1 Phase 2.5, no changes:
- Latency benchmark (`pcvr/scripts/bench_infer.py`).
- `torch.no_grad` → `torch.inference_mode`.
- Sharded parquet via `pyarrow.dataset` (deferred to 2026-05-24, the day after Round 1 closes).
- Embedding pruning if memory is tight.
- Gradient checkpointing on HyFormer blocks.

---

## Phase E — Lock-in (Days 12-14, 2026-05-21 to 2026-05-23)

### E1. Pick best config
Sort `submissions/*/NOTES.md` by leaderboard AUC. Top candidate is the lock-in target.

### E2. Stable retrain
Retrain with seed=42 on the full data window. Verify SHA256 of `model.pt` is logged in submission_manifest.json (need to wire this — see merge plan v1.1 P0.4).

### E3. Pre-flight audit
```bash
cd pcvr/
python -m src.audit pcvr/local_ckpt_dir   # standalone audit
python -m pytest tests/                    # 37/37 must pass
python build_step1_submission.py
python build_step3_submission.py
```

### E4. Submit by 12:00 AOE on Day 13 (2026-05-22)
4-hour buffer before the 16:00 cutoff. Sub #2 = backup config. Sub #3 = held in reserve.

---

## Critical path

```
A1 submit v0.5      (Day 1 — TODAY if you want)
  ↓
A3 capacity diag    (Days 1-2, parallel — no slots)
  ↓
A2 aggregation      (Day 2-3 — 2 slots)
  ↓ (decision: B vs C)
B1/B2/B3/B4   OR    C1/C2/C3/C4   (Days 3-10)
  ↓
D items interleaved (Days 5-12, additive)
  ↓
E1-E4 lock-in       (Days 12-14)
```

The **single highest-information action available right now** is A3 (local capacity diagnostic). It needs no submission slots and tells you which Phase (B vs C) to invest in.

The **single highest-information submission** is A1 (v0.5 with wired validators). The leak-probe output on real data is irreplaceable.

---

## Submission slot budget

| Used | Date | Score |
|---|---|---|
| 1 | 2026-05-08 (v0 baseline) | 0.81144 |
| **2** estimated | 2026-05-09 (Phase A1 + A2 = 3 slots) | TBD |

After Phase A: ~36 slots remain over 13 days = ~2.7/day average. Adequate budget for B/C without scrambling.

---

## What this roadmap does NOT decide

- Whether to merge `pcvr-jw2333` into the teammate's `main` (PR flow). Currently the branch lives as a parallel record on their repo; the teammate decides.
- Whether to publish `jw2333-null/taac-pcvr` (the solo backup repo) publicly. Recommended: stay private until Round 1 closes 2026-05-23.
- The teammate's roadmap. They have v3-v6 planned per their README; we should port any of their successful items as data lands, but their work is on a separate cadence.

---

## Update protocol

When a Phase item completes, append `✅ commit-SHA YYYY-MM-DD` next to it. When a submission lands, update `submissions/README.md`'s log table. When new evidence changes the priority order, write a roadmap-v3.

When in doubt: A3 first (it's free), then revisit this doc.
