# PCVR Merge Plan v1 — Integrating teammate's GitHub repo into `pcvr/`

**Date:** 2026-05-08. Inputs: 1 recon agent + 5 parallel specialist agents (model, trainer, data, submission-hygiene, devx) + verified two divergent claims directly against `pcvr/src/`.

**Anchor scores:** A (`pcvr/`) ≈ **0.86** AUC on platform Step-1. B (`Gcoder2026/Tencent-Advertising-Algorithm-Competition` v1) = **0.807** AUC. B's v2 unscored.

**Merge direction:** Option B = absorb selected parts of B into A. **Do not regress A's 0.86 baseline.**

---

## Critic-integrated revisions (v1.1, 2026-05-08)

A dedicated critic agent challenged this plan after v1 was drafted. Substantive findings, all integrated below:

1. **B's 0.807 anchor is unverified** (BLOCKER per critic). B's source code writes `result.json`, no rename in run.sh — direct verification confirmed. Implication: their leaderboard 0.807 may correspond to a hand-patched zip, not the committed code. **Action item for the user**: ping the teammate, ask whether the graded submission used the committed `submission_v1.zip` as-is or a renamed variant. Until answered, treat 0.807 as a soft signal, not a verifiable anchor.
2. **H2 must update both `get_pcvr_data` AND `_row_group_timestamps`** (data.py:777). Verified: both functions independently `sorted(glob(...))`, so they happen to agree today, but H2 in only one path makes the validator silently misleading. Effort revised: 1h → **2-3h**. Scope: extract a shared `_collect_row_groups_sorted_by_time()` helper.
3. **H3 valid_ratio 0.05 needs `patience` bumped to 7-8**. With step-level eval and halved val set, per-checkpoint AUC variance rises ~√2; current `patience=5` (or 2 in `first_submission`) would trigger spurious early termination. Bundle the bump with the ratio change.
4. **H4 (transformer) and M6 (RoPE) ablations should run LOCALLY first.** Validate against local AUC; only burn a leaderboard slot if local AUC exceeds 0.86 by ≥ 0.002. With 42 total slots over 14 days and roadmap streams already claiming most, ablations on a 4k-step plateau are not slot-worthy without local evidence.
5. **M1 cold-restart sparse re-init may never fire** with current step-based stopping. The re-init hook fires at epoch boundaries; if early-stopping triggers within 1-2 epochs (likely on platform-scale data), the 3-4h implementation buys nothing. **Verify the epoch count of the prior 0.86 run before implementing M1.**
6. **M2 (`seq_truncate='auto'`) verification promoted to H0** — see new H0 below. If any seq domain stores oldest-first, A is silently dropping the most recent user behavior, which is a systematic correctness bug not a tuning knob. The verification step (read one RG, check timestamp order per domain) is a 15-minute task.
7. **NEW H5: embedding-capacity ablation (local only).** The 0.86 AUC plateau at step 4k of a 26k-step run is the strongest diagnostic signal in the project right now. None of the existing H/M items address it. Halve/double `emb_dim` and `d_model`, compare 4k-step AUC locally — directly tells us whether we're capacity-bound (→ prioritize architecture work) or signal-bound (→ prioritize features/data).
8. **NEW M7: end-to-end reproducibility script.** Code review requires reproducing from a cold clone in the platform environment. A has Python build scripts but no single command that does data-prep → train → infer → zip. Add a `Makefile` target or a `scripts/repro.sh` that captures the full pipeline.

---

## Headline finding (CRITICAL)

**B's `infer.py` (both v1 and v2) writes `result.json`. The platform's rulebook mandates `predictions.json`.** Either the grader silently accepts `result.json`, B has an off-repo workaround that renames the file, or B's 0.807 was scored under conditions we can't reproduce. **A's `predictions.json` is correct per spec; do not regress.**

This single finding inverts what "merge B's stuff into A" should mean — most of B's submission hygiene is weaker than A's. Pull only the genuinely new ideas, not the infra patterns.

---

## What A already has that the recon implied was B-unique

After direct verification:

| Claimed B-only | Actual status in A | Action |
|---|---|---|
| RankMixerBlock parameter-free mixing | Identical implementation in `pcvr/src/model.py` (common starter-kit ancestor) | Skip — already in A |
| LongerEncoder adaptive top-K | Identical in A | Skip |
| RoPE asymmetric in CrossAttention | `RoPEMultiheadAttention` used in `TransformerEncoder` L561 and `CrossAttention` L261. Toggled by `cfg.use_rope` (default False) | Skip pull, **enable via config ablation** |
| `emb_skip_threshold` + zero-vector fallback | A has it (default 1M) | Skip |
| MultiSeqHyFormerBlock 4-domain orchestration | Identical in A | Skip |
| `get_sparse_params()` / `get_dense_params()` | A has both, **wired in `trainer.py:115-133`**, both optimizers stepped per training step | Skip — already wired |
| Self-contained checkpoint sidecars | A has all of B's plus `user_id_sample.json` and `submission_manifest.json` is in P0 | Skip |
| Step-level validation (`eval_every_n_steps`) | A has it, exposed in Config | Skip |
| File-system tensor sharing | A has it (set at module top of `data.py`) | Skip |
| 64-boundary time bucketing | A has it (`BUCKET_BOUNDARIES` constant) | Skip |
| Pre-allocated numpy buffers + fused 3D padding | A has it (port from same starter kit) | Skip |

**Bottom line:** Most of B's "standout" features are starter-kit features both projects share. The genuine deltas are smaller than the recon implied.

---

## HIGH priority pulls (do this batch first)

### H0. (NEW from critic) Verify per-domain seq-event timestamp order
**What:** Read one parquet file from training data (or HF sample once H1 lands). For each of `seq_a/b/c/d`'s timestamp column (`<prefix>_<ts_fid>`), compute the diff of consecutive events in a few rows. If diffs are systematically negative, the domain stores newest-first (current `head` truncation is correct). If positive, the domain is oldest-first and `head` truncation drops the most-recent events — `tail` truncation is needed.
**Why:** The result determines whether M2 (`seq_truncate='auto'`) is a P0 correctness bug fix or a P2 polish item. Cheap to know.
**Effort:** 15 min.
**Risk:** None (read-only).
**Files:** new `pcvr/scripts/check_seq_order.py` (single-file analysis script, gitignored).

### H1. `tools/prepare_hf_sample.py` + `data_sample_1000/schema.json`
**What:** Copy B's HuggingFace download script + the committed reference schema that documents the actual platform's 46/14/10/4-domain feature layout (109 fids).
**Why:** A's `tests/conftest.py` uses a 50-row toy schema (2/2/1/2). A schema-mismatch bug only surfaces on platform; pulling B's reference schema lets the smoke test catch it locally.
**Effort:** 30 min (copy 2 files; no integration unless we want to wire `prepare_hf_sample.py` into `make smoke`).
**Risk:** Near-zero. Both files are read-only references.
**Files:** new `pcvr/tools/prepare_hf_sample.py`, new `pcvr/data_sample_1000/schema.json`.

### H2. Timestamp-sorted row-group split (in BOTH paths) — **REVISED per critic**
**What:** Sort row groups by RG-level min timestamp BEFORE the train/valid split, AND update `_row_group_timestamps` to use the same sort, so the validator and training-split use the SAME RG ordering.
**Why:** A's `assert_time_split_monotonic` checks the order but doesn't *fix* the split. Today both functions independently call `sorted(glob(*.parquet))` — they happen to agree, masking the real risk. If RGs within files aren't time-ordered, the validator passes but the split is wrong. **Critical:** if H2 only fixes `get_pcvr_data`, the validator becomes a confidence-builder for the wrong order. Extract a shared `_collect_row_groups_sorted_by_time()` and have both call it.
**Effort:** **2-3h** (revised from 1h — must touch both code paths + add tests).
**Risk:** Low (after the shared-helper refactor; medium if rushed).
**Files:** `pcvr/src/data.py` (both `get_pcvr_data` and `_row_group_timestamps` + new shared helper).

### H3. `valid_ratio: 0.05` + bump `patience` (was 0.10 + patience 5/2) — **REVISED per critic**
**What:** Halve the val split → ~5% more training data, AND bump `patience` from 5 → 8 (and `first_submission` from 2 → 4) to compensate for ~√2× higher per-checkpoint AUC variance.
**Why:** Cutting val data without raising patience causes spurious early termination on noisy dips. Bundle the changes.
**Effort:** 5 min. Two fields on `configs/baseline.py` and `configs/first_submission.py`.
**Risk:** Low after the patience bump.
**Files:** `pcvr/configs/baseline.py`, `pcvr/configs/first_submission.py`.

### H4. `seq_encoder_type='transformer'` ablation — **LOCAL FIRST per critic**
**What:** Run one ablation with `transformer` encoder vs A's default `swiglu`. Pure config change.
**Why:** B's v2 moved to transformer (rationale: SwiGLU has no intra-sequence attention, only per-position FFN). For ordered behavior sequences, attention should win.
**Effort:** 5 min config; ~1 GPU hour for a LOCAL training run on the HF sample (after H1).
**Risk:** Low.
**Submission rule:** Do NOT submit unless local AUC exceeds 0.86 baseline by ≥ 0.002. With ~42 total leaderboard slots and roadmap streams competing for them, an unvalidated ablation is not worth a slot.
**Files:** new `pcvr/configs/ablation_transformer.py` inheriting baseline + `seq_encoder_type='transformer'`.

### H5. (NEW from critic) Embedding-capacity diagnostic — local only
**What:** Run 3 short local training runs at `(emb_dim, d_model) ∈ {(32, 32), (64, 64), (128, 128)}`. Compare 4k-step local AUC.
**Why:** A's 0.86 AUC plateaus at step 4k of a 26k-step run. This is THE diagnostic signal in the project right now. None of the other H/M items address it. The result tells us whether the model is capacity-bound (→ prioritize H4 transformer + M6 RoPE + R1 DIN) or signal-bound (→ prioritize M2 truncation + R2 click head). Without this, every other architectural pull is unguided.
**Effort:** 5 min config × 3; ~3 GPU hours local; no submission slot burn.
**Risk:** None — local only.
**Files:** new `pcvr/configs/diag_emb{32,64,128}.py`.

---

## MEDIUM priority pulls (post first iteration cycle)

### M1. Cold-restart sparse re-init (KuaiShou MultiEpoch) — **CONDITIONAL per critic**
**What:** Implement `model.reinit_high_cardinality_params(threshold)` + trainer hook that snapshots Adagrad state by `data_ptr()`, calls reinit, rebuilds Adagrad, restores state for low-cardinality params only.
**Pre-check:** **Verify how many full epochs A actually completes** in a typical platform run. The re-init fires at epoch boundaries; if early stopping triggers within 1-2 epochs (likely on platform-scale data), this implementation buys nothing. Read the prior 0.86 run's log: how many epochs ran before patience exhausted? If < 3, **skip M1 entirely**.
**Why (if M1 still applies):** Literature reports +0.001-0.003 from this technique. B's trainer has the implementation.
**Effort:** 3-4h.
**Risk:** Medium. Conditional on epoch count.
**Files:** `pcvr/src/trainer.py`, `pcvr/configs/baseline.py`.

### M2. `seq_truncate='auto'` policy
**What:** Per-domain: inspect timestamp sort order once; if newest-first, keep head; if oldest-first, keep tail. Currently A always keeps head (`values[s:s+ul]`), which throws away the most-recent events for chronological domains.
**Why:** Correctness, not tuning. If any seq domain stores oldest-first and is longer than `max_len`, A is silently discarding the most predictive signal.
**Effort:** 2h. ~30 lines in `_convert_batch`, one new `__init__` param.
**Risk:** Low. AUC impact is non-zero but unknown direction until we know each domain's storage order.
**Files:** `pcvr/src/data.py`.

### M3. Wire OOB rate guard + sequence-history leak probe into `train.py`
**What:** Both validators exist in A's `src/data.py` and have unit tests, but `train.py` only calls 2 of 4. Add the calls. (This is an A-internal task surfaced by the comparison — not technically pulled from B.)
**Effort:** 1h.
**Risk:** None.
**Files:** `pcvr/train.py`.

### M4. `_strip_module_prefix()` defensive guard
**What:** When loading `model.pt`, strip a leading `module.` from any key (DDP wrapper artifact). One-liner in `checkpoint.load_state_dict`.
**Why:** A doesn't use DDP today, but Round 2's larger data may force multi-GPU training. Cheap defensive add.
**Effort:** 10 min.
**Risk:** None.
**Files:** `pcvr/src/checkpoint.py`.

### M5. TensorBoard per-step train-loss logging
**What:** A's trainer logs valid AUC + valid LogLoss per epoch; add per-step `Loss/train` like B's. Helps diagnose long-run dynamics (e.g., whether the loss spike at step 5k from our screenshot has a systematic cause).
**Effort:** 5 lines in `trainer._train_step`.
**Files:** `pcvr/src/trainer.py`.

### M6. Try `use_rope=True` ablation — **LOCAL FIRST per critic**
**What:** A has RoPE implementation but it's off by default. Toggle on as one ablation. (Already in code; just config.)
**Effort:** 5 min config + 1 GPU hour LOCAL.
**Submission rule:** Same as H4 — submit only if local AUC ≥ 0.86 + 0.002.
**Files:** new `pcvr/configs/ablation_rope.py`.

### M7. (NEW from critic) End-to-end reproducibility entrypoint
**What:** Add a `Makefile` target `make repro` (or `scripts/repro.sh`) that runs the full pipeline from a cold clone: `tools/prepare_hf_sample.py` (after H1 lands) → `python train.py --config first_submission` → `python infer.py` → `python build_step3_submission.py`. Verifies the submission can be reproduced top-to-bottom with one command.
**Why:** Code review on the best submission is mandatory. Without a single-command reproduction path, the reviewer's environment may differ subtly from ours and the run may fail or score differently — disqualifying us regardless of AUC.
**Effort:** 1h.
**Risk:** None.
**Files:** new `pcvr/Makefile` (or extend existing) + ensure all upstream commands work in sequence.

---

## ROADMAP additions (NOT immediate code; add to roadmap v2)

These came from B's roadmap (v3-v6 in their README) and are genuinely orthogonal to what's already in A's roadmap:

### R1. DIN-style target-attention query (B's v6)
Use the target item's embedding as the query for user-sequence cross-attention, instead of the current learned NS tokens. This is a fundamentally different attention pattern than A's HyFormer queries. Stream A or B addition; ~5h; medium risk; well-documented +0.001-0.005 in DIEN/DIN literature.

### R2. Multi-task click head (CTR auxiliary, B's v4)
Add a sigmoid head for click prediction alongside the conversion head, with a small auxiliary loss. **Prerequisite:** verify the platform's data has a click-event column we can derive a label from (B doesn't say). Stream C addition; ~4h once data confirmed.

### R3. `ARCHITECTURE.md` rule block
B's README has the explicit "no ensembling, 3 subs/day" rules in plain text. A's `ARCHITECTURE.md` references these operationally but should have a top-level constraints block so they survive the sprint as audit fixtures.

---

## Anti-patterns from B — explicitly DON'T pull

| Pattern in B | Why not |
|---|---|
| Versioned monorepo (`v1/`, `v2/` self-contained snapshots) | Code duplication. A has proper git history + tags + `submission_manifest.json` for the same purpose. |
| Pre-built submission/evaluation zips committed to git | Not reproducibility — opposite. Binary artifacts in git create implicit coupling between code and artifact, no rebuild guarantee. A's `build_step{1,3}_submission.py` is correct. |
| `torch.nan_to_num(nan=0, posinf=1, neginf=0)` at inference | Silently converts NaN/Inf to valid probabilities. Hides model pathologies. A's `allow_nan=False` + raise-on-non-finite is correct. |
| CLI-flag-driven config | A's typed `@dataclass` Config with `__post_init__` validation is strictly stronger (typed, IDE-friendly, validated, JSON-serializable). |
| `result.json` output filename | **Wrong filename per spec.** A uses `predictions.json` correctly. |
| No test suite | A has 35/35 passing pytest. Pulling B would mean dropping tests — never. |
| No requirements.txt with pins | A pins. |

---

## Suggested merge order — **REVISED v1.1 per critic**

```
Day-of-merge sequence:

PHASE 0 — Diagnostics (no code merge, just observation, ~45 min):
  ▸ Read the prior 0.86 platform run's log: how many epochs completed
    before patience exhausted? (Determines whether M1 applies at all.)
  ▸ ASK TEAMMATE: did your graded 0.807 submission use the committed
    submission_v1.zip as-is, or a renamed variant?
    (Determines whether B's anchor is verifiable.)

PHASE 1 — Core merges (low-risk, ~5h):
  1. H0  Verify per-domain seq-event timestamp order            [15 min]
         If any domain is oldest-first → M2 promoted to P0.
  2. H1  prepare_hf_sample.py + data_sample_1000/schema.json    [30 min]
  3. M3  Wire OOB + leak validators into train.py                [1h]
  4. M4  _strip_module_prefix in checkpoint.load_state_dict      [10 min]
  5. M5  TensorBoard per-step Loss/train                         [5 lines]
  6. H2  Timestamp-sorted RG split (BOTH paths)                  [2-3h]
  7. H3  valid_ratio 0.05 + patience bump 5→8 / 2→4             [5 min]
  8. M7  Makefile `make repro` end-to-end target                 [1h]
  --- run full pytest, confirm 35/35 still pass ---
  --- if H0 flagged any oldest-first domain, do M2 NOW (correctness fix) ---
  --- rebuild step1_train.zip; this is the moment for an optional v0.5 submission ---

PHASE 2 — Local ablations (no submission slot burn, ~5h GPU):
  9. H5  configs/diag_emb{32,64,128}.py — capacity diagnostic
 10. H4  configs/ablation_transformer.py — encoder ablation
 11. M6  configs/ablation_rope.py — RoPE ablation
  --- compare 4k-step local AUC across {capacity × encoder × RoPE}
      to choose ONE candidate that beats local 0.86 by ≥ 0.002 ---
  --- only THEN submit the winner ---

PHASE 3 — Conditional larger work (only if Phase 2 indicates a path):
 12. M1  Cold-restart sparse re-init                              [3-4h]
         GATED on Phase-0 epoch-count check: skip if < 3 epochs typical.
 13. M2  seq_truncate='auto'                                      [2h]
         Already done in Phase 1 if H0 flagged it.

Roadmap additions (no code, just doc updates):
 14. R1, R2, R3 added to roadmap v2 next time we update it.
     R2 (multi-task click head) requires confirming a click-label
     column exists in the platform data.
```

**Revised total effort:** Phase 1 ≈ 5h merge work + 1h verification = 6h focused day. Phase 2 ≈ 5h GPU time, 0 submission slots. Phase 3 ≈ 0-6h depending on Phase 0 + Phase 2 outcomes. None of it regresses A's 0.86 baseline if Phase 1 + 2 are completed before any new submission.

---

## What this merge does NOT decide

- **Whether to actually run the transformer encoder ablation** — that's a strategic call about which submission slot to spend on it.
- **Whether B's v2 has anything we missed** — v2 wasn't scored, so it's an unverified hypothesis. We could read v2 more deeply if you want.
- **What to do about `result.json`** — if you have channels with the teammate, ask whether that filename actually graded; the answer informs whether to trust their 0.807 anchor at all.

---

## Update protocol

Apply pulls in the suggested order. After each, update this file with ✅ + commit SHA. After the H batch lands, run `pytest pcvr/tests/` and rebuild `step1_train.zip`. Once all H+M items land, this merge plan is closed and rolled into roadmap v2.
