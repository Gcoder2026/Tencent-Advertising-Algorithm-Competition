# PCVR Merge Plan v1 — Integrating teammate's GitHub repo into `pcvr/`

**Date:** 2026-05-08. Inputs: 1 recon agent + 5 parallel specialist agents (model, trainer, data, submission-hygiene, devx) + verified two divergent claims directly against `pcvr/src/`.

**Anchor scores:** A (`pcvr/`) ≈ **0.86** AUC on platform Step-1. B (`Gcoder2026/Tencent-Advertising-Algorithm-Competition` v1) = **0.807** AUC. B's v2 unscored.

**Merge direction:** Option B = absorb selected parts of B into A. **Do not regress A's 0.86 baseline.**

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

### H1. `tools/prepare_hf_sample.py` + `data_sample_1000/schema.json`
**What:** Copy B's HuggingFace download script + the committed reference schema that documents the actual platform's 46/14/10/4-domain feature layout (109 fids).
**Why:** A's `tests/conftest.py` uses a 50-row toy schema (2/2/1/2). A schema-mismatch bug only surfaces on platform; pulling B's reference schema lets the smoke test catch it locally.
**Effort:** 30 min (copy 2 files; no integration unless we want to wire `prepare_hf_sample.py` into `make smoke`).
**Risk:** Near-zero. Both files are read-only references.
**Files:** new `pcvr/tools/prepare_hf_sample.py`, new `pcvr/data_sample_1000/schema.json`.

### H2. Timestamp-sorted row-group split in `get_pcvr_data()`
**What:** Sort row groups by RG-level min timestamp BEFORE the train/valid split, instead of relying on glob order.
**Why:** A's `assert_time_split_monotonic` checks the order but doesn't *fix* the split. If files are glob-sorted but RGs within files aren't time-ordered, the validator passes (because it reads RGs in glob order anyway) but the actual split is wrong. B reads RG min timestamp from parquet metadata (free, no row scan) and sorts before splitting.
**Effort:** 1h. Replace ~15 lines in `pcvr/src/data.py:get_pcvr_data`.
**Risk:** Low. Adds a sort step; doesn't change the dataset contract.
**Files:** `pcvr/src/data.py`.

### H3. `valid_ratio: 0.05` (was 0.10)
**What:** Halve the validation split → ~5% more training data.
**Why:** B uses 0.05 in production; the leaderboard is the ground truth, local AUC is a noisier estimate either way. We have the time-split validator + (after H2) actually time-sorted split, so leakage risk is bounded.
**Effort:** 5 min. One field on `configs/baseline.py` and `configs/first_submission.py`.
**Risk:** Low. Worst case: noisier early-stop signal.
**Files:** `pcvr/configs/baseline.py`.

### H4. `seq_encoder_type='transformer'` ablation
**What:** Run one ablation with `transformer` encoder vs A's default `swiglu`. Pure config change.
**Why:** B's v2 moved to transformer (rationale: SwiGLU has no intra-sequence attention, only per-position FFN). For ordered behavior sequences, attention should win. Costs one submission slot to test.
**Effort:** 5 min config; ~1 GPU hour for a training run.
**Risk:** Low — it's an ablation, not a forced change.
**Files:** new `pcvr/configs/ablation_transformer.py` inheriting baseline + `seq_encoder_type='transformer'`.

---

## MEDIUM priority pulls (post first iteration cycle)

### M1. Cold-restart sparse re-init (KuaiShou MultiEpoch)
**What:** Implement `model.reinit_high_cardinality_params(threshold)` (already exists in starter kit's model.py; should already be in A's port — verify) + trainer hook that snapshots Adagrad state by `data_ptr()`, calls reinit, rebuilds Adagrad, restores state for low-cardinality params only.
**Why:** A's `Config` previously had `reinit_sparse_after_epoch` fields, removed in P0 fix-up because trainer didn't implement them. B's trainer has the implementation. Pull it back. Literature reports +0.001-0.003 on similar systems.
**Effort:** 3-4h. ~50 lines in `trainer.py`, re-add 2 fields to `Config`.
**Risk:** Medium. Optimizer state surgery; benchmark before committing.
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

### M6. Try `use_rope=True` ablation
**What:** A has RoPE implementation but it's off by default. Toggle on as one ablation. (Already in code; just config.)
**Effort:** 5 min config + 1 GPU hour.
**Files:** new `pcvr/configs/ablation_rope.py`.

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

## Suggested merge order

```
Day-of-merge sequence:

  1. H1  prepare_hf_sample.py + data_sample_1000/schema.json    [30 min, near-zero risk]
  2. M3  Wire OOB + leak validators into train.py                [1h, A-internal]
  3. H3  valid_ratio 0.05                                         [5 min]
  4. H2  Timestamp-sorted RG split in get_pcvr_data               [1h]
  5. M5  TensorBoard per-step Loss/train                          [5 lines]
  6. M4  _strip_module_prefix in checkpoint.load_state_dict       [10 min]
  --- run full pytest, confirm 35/35 still pass ---
  --- rebuild step1_train.zip; if you want to submit a v0.5,
      this is the moment ---

Then create ablation configs, NOT yet in run.sh:
  7. H4  configs/ablation_transformer.py
  8. M6  configs/ablation_rope.py

Larger work for after first ablation results land:
  9. M1  Cold-restart sparse re-init                              [3-4h]
 10. M2  seq_truncate='auto'                                      [2h]

Roadmap additions (no code, just doc updates):
 11. R1, R2, R3 added to roadmap v2 next time we update it.
```

Total H+M effort: ~10 hours, achievable in one focused day. None of it touches the v0 scaffold's invariants (audit, predictions.json hygiene, single-model rule, deterministic seed).

---

## What this merge does NOT decide

- **Whether to actually run the transformer encoder ablation** — that's a strategic call about which submission slot to spend on it.
- **Whether B's v2 has anything we missed** — v2 wasn't scored, so it's an unverified hypothesis. We could read v2 more deeply if you want.
- **What to do about `result.json`** — if you have channels with the teammate, ask whether that filename actually graded; the answer informs whether to trust their 0.807 anchor at all.

---

## Update protocol

Apply pulls in the suggested order. After each, update this file with ✅ + commit SHA. After the H batch lands, run `pytest pcvr/tests/` and rebuild `step1_train.zip`. Once all H+M items land, this merge plan is closed and rolled into roadmap v2.
