# PCVR Roadmap v3 — architecture edition (2026-05-10)

**Supersedes:** `2026-05-09-pcvr-roadmap-v2.md`. v2's direction-only cards are tightened here to architecture-level cards: each upgrade specifies the **module** it lives in, the **interface contract** it must preserve, the **test that proves it works**, the **rollback path**, and the **integration boundary** (what it MUST NOT touch). The HOW remains implementer-free; the WHERE and the CONTRACT are now fixed.

**Why v3 exists:** the teammate's three experiments since the v0 anchor (v3: bundled changes; v4: continuous time encoding; v5: longer training) all regressed (-0.005 / -0.014 / -0.014). The pattern is bundled changes, missing test contracts, and silent interface drift. v3 gives every upgrade an architecture frame so future attempts cannot drift outside boundaries without an explicit test failure.

---

## Updated context

- **Today: 2026-05-10.** Round 1 deadline: 2026-05-23 AOE → **13 days remaining**.
- **Anchor: pcvr v0 = AUC 0.81144** (leaderboard). Roadmap forward-plan starts from this anchor; v3/v4/v5 (below) are negative results, not new baselines.
- **Cross-team flow:** local `main` → `jw2333-null/taac-pcvr` (private solo). Local `pcvr-jw2333` → `Gcoder2026/.../main` (merged; canonical team default branch).
- **Open diagnosis:** local valid AUC ~0.86 → leaderboard 0.81144 = ~0.05 generalization gap. Most likely sequence-history leakage; wired `sequence_history_leak_probe` will report on next training run.

## Prior attempts that regressed — data points, not new baselines

| Run | Bundle | AUC | Δ vs 0.81144 | Architecture lesson |
|---|---|---|---|---|
| v3 | transformer + RoPE + focal loss (3 changes at once) | 0.806681 | −0.005 | Bundling prevents isolating the bad lever. **Future attempts on B1 / B2 must be done one at a time.** |
| v4 | continuous log-Δt time encoding | 0.797723 | −0.014 | Either implementation diverged from baseline semantics OR the bucketed encoding was carrying signal. **C3 needs a pre-training numerical test (see card).** |
| v5 | longer training only | 0.797711 | −0.014 | Pure-time-budget changes don't help — model has converged at v0 architecture. **"Train longer" is not a lever; capacity/architecture is.** Validates A3's importance. |

---

## Picking up a card — discipline (read this first)

**Pick ONE card. Read all 12 fields. Implement. Run the test contract. Only then submit. Never bundle.**

This is the discipline that the v3/v4/v5 regressions came from skipping. Single-lever isolation is the structural fix: each card has a test contract designed to FAIL if you bundle two changes together. If the test passes, the change is well-scoped; if it doesn't, you're touching more than one lever.

### Working tree convention

All upgrades land in the **`pcvr/` codebase** (the layered architecture you're currently looking at). **Do NOT create new `vN/` snapshot directories** for upgrades — that was the pattern before the merge. Now the discipline is: one git branch per upgrade off `main`, implemented inside `pcvr/`. The existing `v1/`, `v2/`, `v3/`, `v4/`, `v5/` directories remain as historical record only; do not edit or extend them.

### Before submitting any card — run this checklist

For every upgrade you pick up:

- [ ] I read all 12 fields of the card, not just the "Direction" and "Why".
- [ ] I am changing exactly ONE lever (not bundling B1+B2, not bundling encoder + loss, not bundling data + model).
- [ ] I wrote the card's **Test contract** as a real pytest test in `pcvr/tests/` (or extended an existing test file).
- [ ] The test passes locally — with the project's conda env activated, `python -m pytest pcvr/tests/ -v` shows my new test name as PASSED.
- [ ] I diffed the **Integration** field's listed files against my working branch and confirmed I did NOT touch them.
- [ ] I have a clear **Rollback** — flipping the new config flag back returns me to the prior anchor with no other side effects.

If any box is unchecked, do not build the submission zip. The single most common failure pattern in this project is implementing without writing the test contract — that's what produced the v4 regression. Don't skip step 3 and 4.

### Submission discipline

For each Step-1 submission, write a `NOTES.md` in the corresponding `submissions/<date>_<tag>/` folder explaining:
- which card(s) you picked,
- which lever changed,
- what the test contract assertion was and that it passed,
- which config file you used.

Without `NOTES.md`, the leaderboard score is unattributable; the next person can't tell what was different.

---

## How to read an upgrade card

Every item below uses the same 12-field structure:

```
Direction          — what kind of change (one sentence)
Why                — the hypothesis
─── Architecture frame ───
Module             — file(s) / class(es); module-level, not line-level
Interface          — what shape / API must be preserved
Test contract      — the specific assertion that proves correctness (NOT the implementation)
Rollback           — how to undo cleanly
Integration        — what existing modules MUST NOT be modified
─── (already there in v2) ───
Stage 1 zip        — what changes in step1_train.zip
Stage 3 zip        — what changes in step3_infer.zip
Success criterion  — the concrete number / signal we look for
Watch out for      — known blockers, prior regression cross-references
Reference          — paper or standard technique
```

The Architecture frame is the new contribution. It defines BOUNDARIES — implementers get freedom inside them, but cannot drift outside without an explicit test failure.

---

## Phase A — Diagnostics (do these BEFORE any architectural work)

The 0.05 generalization gap is the most expensive open question. Phase A resolves it before submission slots are spent on architecture or feature work.

### A1. Submit v0.5 — capture validator diagnostics on real platform data

**Direction:** diagnostic submission using the pre-built v0.5 zips to surface `sequence_history_leak_probe` and `oob_rate_check` outputs on platform data.

**Why:** the 0.05 gap (local 0.86 → leaderboard 0.81144) is most likely sequence-history leakage. The validators are wired and will print to Step-1 log on the platform; this is the only way to confirm or rule out leakage on real data. AUC is secondary.

#### Architecture frame
- **Module:** `submissions/2026-05-09_v0.5_post-phase1/` — pre-built zips, upload as-is; no code changes.
- **Interface:** platform grader contract — `step1_train.zip` runs training end-to-end; `step3_infer.zip` reads `MODEL_OUTPUT_PATH` checkpoint and writes `predictions.json`.
- **Test contract:** pytest assertion that `sequence_history_leak_probe` writes a log line containing `n_future_events=` and `oob_rate_check` writes a line containing `oob_rate=`; both must be parseable floats.
- **Rollback:** no code was changed; nothing to revert — v0 checkpoint remains the live anchor.
- **Integration:** `pcvr/src/data.py`, `pcvr/src/trainer.py`, `pcvr/infer.py` must NOT be modified for this submission.

**Stage 1 zip:** pre-built; upload `step1_train.zip` unchanged.
**Stage 3 zip:** pre-built; upload `step3_infer.zip` after Step 2 publishes the new checkpoint.
**Success criterion:** Step-1 log contains `n_future_events=` value; if `> 0` → leakage confirmed; if `= 0` → distribution shift is the gap cause. Leaderboard AUC within ±0.005 of 0.81144. **Also record from the same log: the actual number of epochs completed before early-stop** (gates B3 — sparse re-init is wasted if early-stop fired inside epoch 1).
**Watch out for:** AUC drop > 0.005 below anchor means v0.5 changes regressed somewhere; diff v0 vs v0.5 before re-submitting.
**Reference:** standard train/test temporal-split leakage audit.

### A2. Aggregation-strategy ablation on published v0 checkpoint

**Direction:** inference-only experiment — expose aggregation strategy as a `Config` enum field, submit two non-mean variants without retraining.

**Why:** `infer.py:main` hardcodes mean aggregation. If the eval set has multiple rows per user, `max` or `last_by_ts` may shift AUC by 0.005–0.020 at zero training cost. Cheapest experiment in the roadmap.

#### Architecture frame
- **Module:** `pcvr/infer.py:main` (aggregation loop); `pcvr/configs/baseline.py:Config` (new field). No changes to `pcvr/src/`.
- **Interface:** `Config` gains `score_agg: str` validated against `{"mean", "max", "last_by_ts"}` in `__post_init__`. `infer.py:main` reads `cfg.score_agg` and dispatches via a `_AGGREGATORS` dict mapping each name to a callable `(scores: list, timestamps: list) -> float` — no inline `if/elif` branches.
- **Test contract:** pytest assertion that for each strategy, calling the aggregator with a two-row input for the same `user_id` returns a single finite float, and that `last_by_ts` returns the score from the row with the larger timestamp.
- **Rollback:** `score_agg` defaults to `"mean"`; removing the two new strategy configs restores baseline behavior without touching model/training code.
- **Integration:** `pcvr/src/trainer.py`, `pcvr/src/data.py`, `pcvr/src/model.py` must NOT be modified. Stage-1 zip is unchanged.

> **Downstream dependency:** A2 establishes the strategy-dispatch refactor in `infer.py:main`. Cards C1, C2, C4 (which say "small loader update" for Stage 3 zip) all assume A2 has landed first. **A2 must precede C1/C2/C4 chronologically** even if Phase A is otherwise complete before Phase C starts.

**Stage 1 zip:** unchanged — reuses published v0 checkpoint.
**Stage 3 zip:** ships two variants (`step3_infer_max.zip`, `step3_infer_last_by_ts.zip`), each with a separate config file.
**Success criterion:** a non-mean strategy beats mean by ≥ 0.002 AUC → adopt it as new default. Both variants must produce `predictions.json` with all user_ids present and all scores finite.
**Watch out for:** `last_by_ts` requires a timestamp field in the eval batch; verify before building. If absent, submit `max` only (1 slot).
**Reference:** standard retrieval/ranking aggregation ablation.

### A3. Local capacity diagnostic — emb_dim × d_model sweep

**Direction:** local-only training sweep across three isolated configs; compare 4k-step valid AUC to determine whether v0 is capacity-bound or signal-bound.

**Why:** v0 training plot shows valid AUC plateau at ~4k steps out of 26k. v5 (longer training, same arch) also plateaued at 0.797711, independently confirming the ceiling is architectural. This diagnostic determines whether Phase B (architecture) or Phase C (data/signal) is the higher-ROI path.

#### Architecture frame
- **Module:** `pcvr/configs/` — three new config files (`cap_32.py`, `cap_64.py`, `cap_128.py`), each a standalone `Config` subclass; `pcvr/src/trainer.py:Trainer` read-only.
- **Interface:** each config sets `(emb_dim, d_model)` to `(32, 32)`, `(64, 64)`, `(128, 128)`; other fields match `baseline.py`; `eval_every_n_steps=500` so AUC is captured before 4k steps; `patience=999` to prevent early-stop before the 4k mark.
- **Test contract:** pytest assertion that all three configs instantiate without error, `Config.__post_init__` passes for each, a single training-step with each config runs without exception and emits a log line matching `val_auc=\d+\.\d+`.
- **Rollback:** configs are additive new files; deleting them leaves the codebase unchanged; no shared state between runs.
- **Integration:** `pcvr/src/trainer.py`, `pcvr/src/data.py`, `pcvr/infer.py`, `pcvr/configs/baseline.py` must NOT be modified. Each run uses its own `ckpt_dir`.

**Stage 1 zip:** not produced.
**Stage 3 zip:** not produced.
**Success criterion:** if `cap_128` 4k-step valid AUC ≥ `cap_64` + 0.005 → capacity-bound → execute Phase B. If delta < 0.005 across all three → signal-bound → execute Phase C.
**Watch out for:** CPU-only environment makes each run slow — run `cap_32` first to calibrate. `d_model % num_heads == 0` constraint must hold; `cap_32` with `d_model=32` satisfies this for `num_heads=4`.
**Reference:** standard model-capacity ablation. v5 negative result independently validates the relevance of this diagnostic.

---

## Phase B — Architecture levers (if A3 says capacity-bound)

### B1. Self-attention sequence encoder

**Direction:** swap `SwiGLUEncoder` for `TransformerEncoder` via a single config flag, leaving all adjacent modules untouched.

**Why:** position-to-position attention over behavior sequences captures "clicked X right after viewing Y" co-occurrence signals that the position-independent SwiGLU FFN structurally cannot model.

#### Architecture frame
- **Module:** `pcvr/src/model.py` — `create_sequence_encoder()` factory; `MultiSeqHyFormerBlock.seq_encoders`; `PCVRHyFormer.__init__()` passes `seq_encoder_type` through.
- **Interface:** `SwiGLUEncoder` and `TransformerEncoder` both return `(output: Tensor[B, L, D], key_padding_mask: Tensor[B, L])`. `MultiSeqHyFormerBlock` calls the encoder generically; no caller edits required.
- **Test contract:** with `seq_encoder_type="transformer"` and `"swiglu"`, assert `model(batch).shape == (B, action_num)` for identical batch; assert encoder output shape `(B, L, D)` is identical between both variants on fixed random input; **AND assert the two encoder outputs are NOT element-wise close** (`not torch.allclose(swiglu_out, transformer_out, atol=1e-3)`) — otherwise the swap is a no-op (e.g., an identity passthrough returning the right shape would pass a shape-only check). Must pass before any training run.
- **Rollback:** set `seq_encoder_type="swiglu"`. State-dict keys diverge between variants (no shared checkpoint); rollback is config-only.
- **Integration:** `MultiSeqHyFormerBlock`, `CrossAttention`, `_run_multi_seq_blocks`, `MultiSeqQueryGenerator`, all NS tokenizers must NOT be modified.

**Stage 1 zip:** new experiment config with `seq_encoder_type="transformer"`. No source changes — factory and both encoder classes already exist.
**Stage 3 zip:** unchanged — encoder type reconstructed from `train_config.json` sidecar at load time.
**Success criterion:** local valid AUC > 0.81144 + 0.002 vs SwiGLU baseline under identical training budget.
**Watch out for:** v3 bundled this with RoPE + focal loss and regressed −0.005. **Run in strict isolation.** Memory: `TransformerEncoder` adds QKV projections per block per sequence — verify batch fits GPU before scaling `num_hyformer_blocks`. `d_model % num_heads == 0` must hold.
**Reference:** Vaswani et al. 2017.

### B2. Rotary position encoding (RoPE)

**Direction:** enable RoPE in attention paths via `use_rope=True` config flag, with no structural model changes.

**Why:** `TransformerEncoder` and `LongerEncoder` use `RoPEMultiheadAttention` which is permutation-invariant when `use_rope=False`; RoPE injects temporal order into Q/K dot-products at zero parameter cost.

#### Architecture frame
- **Module:** `pcvr/src/model.py` — `PCVRHyFormer.__init__()` conditionally constructs `self.rotary_emb`; `_run_multi_seq_blocks()` passes cos/sin to each block; `RoPEMultiheadAttention.forward()` applies rotation when not None.
- **Interface:** `use_rope: bool` field in `Config` (already present, default `False`). RoPE cache buffers are `persistent=False` → absent from saved checkpoints.
- **Test contract:** with `use_rope=True`, assert `model(batch).shape == (B, action_num)`; assert `rotary_emb is not None`; assert Q and K tensors differ from non-RoPE case on same batch (rotation actually fires).
- **Rollback:** set `use_rope=False`. No state-dict key differences — buffers are non-persistent.
- **Integration:** `CrossAttention`, `LongerEncoder`, `MultiSeqHyFormerBlock` must NOT be modified — they already accept optional rope args.

**Stage 1 zip:** new config with `use_rope=True`. No source changes.
**Stage 3 zip:** unchanged — `use_rope` read from `train_config.json`.
**Success criterion:** RoPE-on vs RoPE-off (holding `seq_encoder_type` fixed), local AUC delta ≥ +0.002.
**Watch out for:** only meaningful when an attention encoder is active. **DO NOT enable RoPE while `seq_encoder_type="swiglu"`** — in that mode RoPE only touches `CrossAttention` (KV-side only), weaker and harder to interpret. v3 bundled this; isolate.
**Reference:** Su et al. 2021 (RoFormer).

### B3. Cold-restart sparse re-initialization (KuaiShou MultiEpoch)

**Direction:** add an epoch-boundary trainer hook that calls `model.reinit_high_cardinality_params()`, resetting only high-cardinality embeddings while preserving low-cardinality optimizer state.

**Why:** rare-ID embeddings accumulate gradient noise and overfit faster than dense parameters; periodic Xavier re-init breaks the overfit cycle without disturbing well-learned low-cardinality or time embeddings.

#### Architecture frame
- **Module:** `pcvr/src/trainer.py` — epoch-end hook site; `pcvr/src/model.py` — `PCVRHyFormer.reinit_high_cardinality_params(threshold)` already implemented; `pcvr/configs/baseline.py` — add `reinit_every_n_epochs: int = 0` (0 = disabled) and `reinit_cardinality_threshold: int = 10000`.
- **Interface:** hook fires at end of epoch N if `N % reinit_every_n_epochs == 0 and N > 0`. Calls `model.reinit_high_cardinality_params(cfg.reinit_cardinality_threshold)`; resets corresponding Adagrad accumulator slots. Preserves all dense and low-cardinality optimizer state.
- **Test contract:** after one synthetic epoch, assert `not torch.equal(pre_snapshot_high, post_snapshot_high)` (high-cardinality embedding weights changed exactly — not approximately) **AND** `torch.equal(pre_snapshot_low, post_snapshot_low)` (low-cardinality embedding weights are bitwise unchanged); assert Adagrad accumulator for the reinitialized param's `data_ptr()` was zeroed (i.e., `state_sum.abs().sum().item() == 0` post-reinit). An implementer who reinits ALL embeddings would pass the high-card assertion but fail the low-card assertion — that's the intended failure mode.
- **Rollback:** set `reinit_every_n_epochs=0`. No architecture changes; state-dict unaffected.
- **Integration:** `PCVRHyFormer` must NOT be modified — `reinit_high_cardinality_params` is already present. Only `trainer.py` and `baseline.py` change.

**Stage 1 zip:** new trainer hook + two config fields. Source change confined to `trainer.py` and `baseline.py`.
**Stage 3 zip:** unchanged — reinit is training-only.
**Success criterion:** local valid AUC improves ≥ 0.001 vs without reinit, over ≥ 3 full epochs.
**Watch out for:** **prerequisite — verify v0 training completed ≥ 3 epochs** before investing. If early-stop fires inside epoch 1 (likely on full-scale data), the hook never executes. v5's plateau at the same training horizon as v0 suggests early-stop did fire; check the v0 log first.
**Reference:** KuaiShou "MultiEpoch: Reusing Training Data" re-init trick.

### B4. Capacity bump

**Direction:** scale `d_model` and `emb_dim` via config, conditioned on A3 confirming capacity-bound behavior.

**Why:** v0 AUC plateaus at ~4k steps of 26k, consistent with a capacity ceiling; more parameters give the model representational headroom — but only if A3 confirms the bottleneck is capacity, not signal.

#### Architecture frame
- **Module:** `pcvr/configs/baseline.py` — `d_model`, `emb_dim`, `num_heads` fields; `pcvr/src/model.py` — `PCVRHyFormer.__init__()` divisibility check `d_model % T == 0` (T = `num_queries * num_sequences + num_ns`).
- **Interface:** config change only. Divisibility constraint enforced at construction with a clear error message. No other interface changes.
- **Test contract:** construct `PCVRHyFormer` with doubled `d_model` (128); assert no ValueError; assert `model(batch).shape == (B, action_num)` for the larger config.
- **Rollback:** revert config to `d_model=64`, `emb_dim=64`. Checkpoints NOT cross-compatible — rollback means retraining.
- **Integration:** no source changes. `infer.py` reconstructs from `train_config.json` sidecar.

**Stage 1 zip:** new config with larger `d_model`/`emb_dim`/`num_heads`. No source changes.
**Stage 3 zip:** unchanged — model loader reads dimensions from checkpoint sidecar.
**Success criterion:** doubling capacity, local AUC at step 4k ≥ 0.81144 + 0.005 (confirming capacity-bound). If not, signal-bound — stop and execute Phase C.
**Watch out for:** divisibility — with `rank_mixer_mode="full"`, increasing `d_model` may require adjusting `num_queries` or `num_ns` to keep T a valid divisor. 19 GiB GPU cap; estimate parameter count before training.
**Reference:** scaling laws; standard industrial-recsys practice.

---

## Phase C — Data / signal levers (if A3 says signal-bound)

### C1. DIN-style target-attention query

**Direction:** replace `MultiSeqQueryGenerator`'s FFN-derived query tokens with queries derived from the target item embedding, so each HyFormer block focuses on history relevant to the current target.

**Why:** current queries summarize user history globally regardless of which item is being scored; target-aware queries dramatically improve attention relevance — dominant single-lever in CTR/CVR literature (+0.005–0.015 in comparable systems).

#### Architecture frame
- **Module:** `pcvr/src/model.py` — `MultiSeqQueryGenerator`; `PCVRHyFormer.forward()` at step 3 (query generation); a new `DINQueryGenerator` class alongside.
- **Interface:** `DINQueryGenerator` produces identical output signature to `MultiSeqQueryGenerator`: list of `(B, Nq, D)` tensors of length S. Takes the same args plus `item_ns: (B, num_item_ns, D)` already computed at step 1. Swap is gated by `use_din_query: bool` (new config field). Downstream code unchanged.
- **Test contract:** assert `DINQueryGenerator(batch).shape` matches `MultiSeqQueryGenerator` output shape; assert query tensor values differ between the two generators on a batch where two items differ (target-awareness actually fires).
- **Rollback:** set `use_din_query=False`. State-dict keys for `query_generator.*` differ — rollback requires retraining.
- **Integration:** `MultiSeqHyFormerBlock`, `CrossAttention`, `_run_multi_seq_blocks`, all NS tokenizers must NOT be modified.

**Stage 1 zip:** new `DINQueryGenerator` in `model.py` + config field + wiring in `PCVRHyFormer.__init__()` and `forward()`.
**Stage 3 zip:** small loader update — `query_generator.*` state-dict keys change; confirm `use_din_query` is written to sidecar so `infer.py` reconstructs correctly.
**Success criterion:** local valid AUC ≥ 0.81144 + 0.003.
**Watch out for:** the DIN query must use mean-pooled or first item-NS token as the query seed — not raw int features. Gradient through `item_ns` must NOT be blocked; it shares the item-tokenizer parameters. Medium implementation complexity; test output shape strictly.
**Reference:** Zhou et al. 2018 (DIN); Zhou et al. 2019 (DIEN).

### C2. Multi-task click head (CTR auxiliary loss)

**Direction:** add a second sigmoid head on the shared `output_proj` predicting clicks (`label_type==1`), weighted at 0.1× the main conversion loss.

**Why:** the platform's `label_type` field emits click labels alongside conversion labels; supervising on both gives the shared encoder more gradient signal per row and acts as regularization — standard ESMM-style.

#### Architecture frame
- **Module:** `pcvr/src/model.py` — add `self.click_clsfier` when `use_aux_click=True`; **BOTH `forward()` AND `predict()` must be updated** (inference uses `predict()`, not `forward()` — verify at `pcvr/infer.py` call site); `pcvr/src/trainer.py` — loss computation; `pcvr/src/data.py` — `_convert_batch()` must emit `click_label` (currently only `label = (label_type == 2)`); `pcvr/configs/baseline.py` — add `use_aux_click: bool = False` and `aux_click_weight: float = 0.1`.
- **Interface:** `PCVRHyFormer.forward()` AND `predict()` both return single `(B, action_num)` logit when `use_aux_click=False` (unchanged). When `True`, BOTH methods return tuple `((B, 1), (B, 1))` = `(cvr_logits, click_logits)`. Trainer unpacks `forward()`'s tuple; inference unpacks `predict()`'s tuple and uses `cvr_logits[0]` only.
- **Test contract:** assert `model.forward(batch)` returns tuple `((B,1),(B,1))` when `use_aux_click=True`; assert `model.predict(batch)` ALSO returns the same tuple structure under the same flag (NOT just `forward`); assert conversion AUC computed only from `cvr_logits`.
- **Rollback:** set `use_aux_click=False`. Click head state-dict keys are additive (`click_clsfier.*`) — removing requires `strict=False` load or retrain.
- **Integration:** NS tokenizers, sequence encoders, `_run_multi_seq_blocks` must NOT be modified — click head attaches only at the classifier layer.

**Stage 1 zip:** new click head + click label extraction in `data.py` + aux loss in `trainer.py` + `predict()` updated alongside `forward()` + two config fields.
**Stage 3 zip:** `infer.py` must unpack `predict()`'s tuple and use `cvr_logits` only when `use_aux_click=True` is read from the config sidecar. Verify at the line that today reads `logits, _ = model.predict(inp)`.
**Success criterion:** conversion AUC ≥ 0.81144 + 0.001 with `aux_click_weight=0.1`.
**Watch out for:** **PREREQUISITE — before any implementation, count `label_type` values on training data:**
```bash
python -c "import pyarrow.parquet as pq; import numpy as np; t=pq.read_table('<train_parquet>',columns=['label_type']); v,c=np.unique(t['label_type'].to_pylist(),return_counts=True); print(dict(zip(v,c)))"
```
If `label_type==1` count is near zero, skip this card. Aux weight > 0.3 risks conversion AUC regression — start at 0.1.
**Reference:** Ma et al. 2018 (ESMM — Entire Space Multi-task Model).

### C3. Continuous log-Δt time encoding **(prior v4 regression — extra care)**

**Direction:** replace the 64-bucket `searchsorted` time embedding with a continuous `log(1 + Δt)` linear projection, preserving the `(B, L, D)` additive injection point in `_embed_seq_domain`.

**Why:** the 64-bucket scheme loses within-bucket ordinal information; continuous encoding preserves the full recency signal that dominates PCVR.

#### Architecture frame
- **Module:** `pcvr/src/data.py` — `_convert_batch()` time bucketing block; `BUCKET_BOUNDARIES`; `NUM_TIME_BUCKETS`; `pcvr/src/model.py` — replace `self.time_embedding = nn.Embedding(num_time_buckets, d_model)` with `ContinuousTimeEncoder(d_model)` linear projection; `_embed_seq_domain()` addition point uses the new encoder; `pcvr/configs/baseline.py` — add `use_continuous_time: bool = False`.
- **Interface:** when `use_continuous_time=False`, existing path unchanged. When `True`, `_convert_batch` emits `{domain}_log_time_delta: Tensor[B, L]` (float32) instead of `{domain}_time_bucket: Tensor[B, L]` (int64). `ContinuousTimeEncoder.forward(x: Tensor[B,L]) → Tensor[B,L,D]` — `nn.Linear(1, d_model)` applied per position.
- **Test contract (v4-regression guard, MANDATORY before training):**
  1. For a known Δt = 300s, assert `log(1+300) ≈ 5.71` is what the encoder receives.
  2. Two Δt values that previously mapped to the same bucket (e.g., 305s and 595s, both bucket 21) produce **distinct** encoder outputs (`not torch.allclose(out_305, out_595, atol=1e-3)`).
  3. **Assert the OUTPUT vector for padding positions is exactly the zero vector** — `torch.equal(encoder(padding_mask_input), torch.zeros(B, L, D))` where padding positions are masked OUT before the linear projection. This is an OUTPUT assertion on the model's encoder, not a hint to the implementer. A buggy implementation that passes `log(1)=0` through `nn.Linear(1, D)` with nonzero bias produces a non-zero output vector and MUST fail this test.
  4. Assert `ContinuousTimeEncoder(batch).shape == (B, L, D)`.
  5. **Assert `infer.py:_run()` correctly routes the new batch key.** With `use_continuous_time=True` in the config sidecar, assert that `_run()` consumes the float `{domain}_log_time_delta` key from the batch and feeds it to the model's `ContinuousTimeEncoder`. The current `_run()` hardcodes `{domain}_time_bucket` lookup with a silent fallback-to-zeros — that fallback path is what silently broke v4 and MUST be exercised by this test.

  All five assertions must pass in a pytest unit test BEFORE any training run. **This is the structural fix that surfaces the v4-style bug at unit-test time.**

- **Rollback:** set `use_continuous_time=False`. State-dict keys differ (`time_embedding` vs `time_encoder.*`) — rollback requires retraining; bucketed path fully preserved behind the flag.
- **Integration:** `MultiSeqHyFormerBlock`, all attention modules, NS tokenizers must NOT be modified. Change confined to `_embed_seq_domain` and `_convert_batch`.

**Stage 1 zip:** modified `_convert_batch` (emits float `log_time_delta` instead of int `time_bucket`), new `ContinuousTimeEncoder` class, wired in `_embed_seq_domain` and `__init__`, new config field.
**Stage 3 zip:** `infer.py` reconstructs from sidecar; `use_continuous_time` field triggers `ContinuousTimeEncoder` construction. **AND `infer.py:_run()` must branch on `cfg.use_continuous_time` to read the correct batch key** (`{domain}_log_time_delta` vs `{domain}_time_bucket`) — the silent zero-fallback in the current `_run()` is the v4 bug surface.
**Success criterion:** local valid AUC ≥ 0.81144 + 0.001.
**Watch out for:** **v4 regressed −0.014.** The v4 implementation likely either (a) failed to zero-out padding positions, or (b) changed the dataset path in a way that broke `label` derivation. The four-assertion test contract above is **mandatory** before training. Do not bundle this with any other change.
**Reference:** Time2Vec (Kazemi et al. 2019); continuous-time encoding in sequential recommendation.

### C4. Target × user-profile cross features

**Direction:** add explicit Hadamard-product interaction between mean-pooled item NS tokens and mean-pooled user NS tokens at the classifier head, before `self.clsfier`.

**Why:** `PCVRHyFormer` attends over sequences but never computes explicit pairwise interactions between target-item and user-profile embeddings at the prediction layer; Hadamard products inject these classical CTR signals cheaply.

#### Architecture frame
- **Module:** `pcvr/src/model.py` — `PCVRHyFormer.forward()` and `predict()` after `output_proj` and before `clsfier`; a new `CrossFeatureLayer(d_model)` module; `pcvr/configs/baseline.py` — add `use_cross_features: bool = False`.
- **Interface:** when `use_cross_features=False`, path unchanged. When `True`, the cross vector `(B, D)` is added to `output_proj(...)` before `clsfier`. `clsfier` input shape `(B, D)` is unchanged — no downstream edits.
- **Test contract:** assert `model(batch).shape == (B, action_num)` with `use_cross_features=True`; assert the cross vector has non-zero values for a non-trivial batch (interaction is actually computed).
- **Rollback:** set `use_cross_features=False`. `cross_layer.*` state-dict keys are additive — use `strict=False` or retrain.
- **Integration:** NS tokenizers, sequence encoders, `_run_multi_seq_blocks`, `output_proj` must NOT be modified.

**Stage 1 zip:** new `CrossFeatureLayer` + wiring + config field.
**Stage 3 zip:** small loader update — confirm `use_cross_features` written to `train_config.json`.
**Success criterion:** local valid AUC ≥ 0.81144 + 0.001.
**Watch out for:** mean-pooling discards per-token structure — start with exactly one Hadamard pair (mean(user_ns) × mean(item_ns)) before adding more. Multiple cross terms without L2 regularization will overfit. Do not start with more than 2-3 pairs.
**Reference:** PNN (Qu et al. 2016); FM-style feature interactions.

---

## Phase D — Round-2 prep (interleave during Phases B/C; additive only)

Every Phase D card must include in its test contract: **"Round-1 AUC unchanged (within ±0.001) on the existing v0 checkpoint."** If the optimization changes outputs, it's not Round-2 prep — it's a different upgrade.

### D1. Inference latency benchmark

**Direction:** standalone timing script that measures v0 checkpoint throughput (ms/row) as a reproducible baseline.

**Why:** Round 2 applies a latency tiebreaker on 10× data; without a numeric baseline, no D2/D3/D4/D5 optimization can be declared a win.

#### Architecture frame
- **Module:** new `pcvr/bench_latency.py` standalone script; read-only imports from `infer.py` building blocks (`PCVRParquetDataset`, `PCVRHyFormer`, DataLoader).
- **Interface:** CLI args `--ckpt-dir`, `--eval-data`, `--batch-size`, `--n-rows`; emits `{"ms_per_row": float, "rows": int, "batch_size": int}` to stdout. Same inputs → reproducible to ±5% (warm GPU, 3-run median).
- **Test contract:** two runs on identical inputs produce median ms/row within ±5% of each other; Round-1 AUC unchanged (v0 checkpoint not modified).
- **Rollback:** delete `bench_latency.py`. No other file touched.
- **Integration:** `infer.py`, `src/model.py`, `src/data.py` — read-only imports only.

**Stage 1 zip:** unchanged.
**Stage 3 zip:** unchanged — benchmark is local-only tooling.
**Success criterion:** baseline `ms_per_row` captured + committed to `docs/` before any D2-D5 work; re-run after each optimization confirms reduction.
**Watch out for:** GPU warm-up variance — discard the first batch. CPU fallback unrepresentative of the platform.

### D2. `torch.no_grad` → `torch.inference_mode`

**Direction:** replace `@torch.no_grad()` on `_run()` in `infer.py` with `@torch.inference_mode()`.

**Why:** `inference_mode` disables the autograd engine entirely (no version counter increments, no `grad_fn` tracking) — 5-15% latency reduction on embedding-heavy forward passes.

#### Architecture frame
- **Module:** `pcvr/infer.py` → `_run()` inner function (decorator).
- **Interface:** `_run()` signature and call site unchanged; predictions dict output identical.
- **Test contract:** `predictions.json` output is bit-identical to `no_grad` output on same v0 checkpoint and eval shard (JSON keys + values within 1e-7); D1 benchmark shows ≥ 3% ms/row reduction.
- **Rollback:** revert the single decorator line.
- **Integration:** `src/model.py`, `src/data.py` — not touched.

**Stage 1 zip:** unchanged.
**Stage 3 zip:** `infer.py` (one line change).
**Success criterion:** D1 ms/row decreases ≥ 3%; AUC unchanged on leaderboard.
**Watch out for:** any downstream `.detach()` or `.numpy()` calls — `inference_mode` tensors are already detached; double-detach safe but verify `.cpu().numpy()` still works.
**Reference:** PyTorch `torch.inference_mode` docs.

### D3. Sharded parquet streaming via `pyarrow.dataset`

**Direction:** replace `pq.ParquetFile` + per-file `iter_batches` in `PCVRParquetDataset.__iter__` with a `pyarrow.dataset.dataset()` scanner that partitions across shards for multi-worker DataLoader.

**Why:** Round-2 data is 10×; current `pq.ParquetFile` loop re-opens files per worker without shard awareness, creating I/O contention.

#### Architecture frame
- **Module:** `pcvr/src/data.py` → `PCVRParquetDataset.__iter__` and `__init__` (file-list construction block).
- **Interface:** `__init__` signature unchanged; `__iter__` still yields `Dict[str, Any]` batches of the same schema. Worker-shard assignment must replicate the current `i % num_workers == worker_id` semantics.
- **Test contract:** on a 100-row local parquet sample, **byte-identical** batch output (tensor values, shape, order with `shuffle=False`) between old `pq.ParquetFile` path and new `pyarrow.dataset` path; Round-1 AUC unchanged (±0.001) on v0 checkpoint replay.
- **Rollback:** feature-flag `use_arrow_dataset: bool = False`; old path remains live behind the flag.
- **Integration:** `infer.py`, `trainer.py`, `model.py` — not touched.

**Stage 1 zip:** `src/data.py` (new streaming path behind flag).
**Stage 3 zip:** `src/data.py` (same change; flag set in config).
**Success criterion:** ≥ 20% I/O throughput gain on a 10-shard local benchmark; byte-identical output verified.
**Watch out for:** `row_group_range` slicing logic must translate to fragment filtering. **Defer until after Round 1 closes (2026-05-24).** High blast radius.

### D4. Embedding-table pruning via `allowed_id_sets.json`

**Direction:** preprocessing step that computes top-N most-frequent IDs per high-cardinality int feature and writes `allowed_id_sets.json`; `PCVRParquetDataset` reads it at construction to remap rare IDs to 0.

**Why:** high-cardinality features waste embedding-table rows on near-zero-frequency IDs; pruning reduces model size and speeds up embedding lookup — critical for Round-2 latency.

#### Architecture frame
- **Module:** new `pcvr/scripts/build_allowed_ids.py` (preprocessing); `pcvr/src/data.py` → `PCVRParquetDataset.__init__` and `_convert_batch` (consume sidecar).
- **Interface:** `allowed_id_sets.json` schema: `{feature_id: [allowed_int, ...]}`. `PCVRParquetDataset.__init__` accepts optional `allowed_ids_path: Optional[str] = None`; when provided, builds per-feature lookup mask applied in `_convert_batch` before vocab-size clipping. Model code (`model.py`) and `infer.py` not touched — vocab sizes already read from dataset properties.
- **Test contract:** with `allowed_ids_path=None`, output identical to current; with path provided, OOB IDs are mapped to 0 and reported vocab sizes reflect pruning; Round-1 AUC unchanged (±0.001) on v0.
- **Rollback:** pass `allowed_ids_path=None`. Sidecar file unused.
- **Integration:** `model.py` must NOT be modified. `infer.py` adds one line to pass `allowed_ids_path` from checkpoint dir; `trainer.py` similar.

**Stage 1 zip:** `src/data.py` + `scripts/build_allowed_ids.py`.
**Stage 3 zip:** `infer.py` (pass sidecar path) + `src/data.py`.
**Success criterion:** embedding-table parameter count reduced ≥ 30% on top-3 high-cardinality features; AUC unchanged; D1 latency decreases.
**Watch out for:** pruning threshold (top-N vs frequency cutoff) tuned on training data only — never on eval data.

### D5. Gradient checkpointing on HyFormer blocks

**Direction:** add `use_grad_ckpt: bool` config flag to `PCVRHyFormer` that wraps each `MultiSeqHyFormerBlock.forward` call in `torch.utils.checkpoint.checkpoint`, trading recomputation for VRAM.

**Why:** Round-2 longer sequences will exhaust VRAM at current batch sizes without checkpointing; this reclaims VRAM for larger batches or deeper stacks.

#### Architecture frame
- **Module:** `pcvr/src/model.py` → `PCVRHyFormer.__init__` (store flag) and `forward` (wrap `MultiSeqHyFormerBlock.forward` call); `MultiSeqHyFormerBlock.forward` signature unchanged.
- **Interface:** `__init__` gains `use_grad_ckpt: bool = False`; all other constructor args and `forward` I/O unchanged. Config gains matching field.
- **Test contract:** (1) with `use_grad_ckpt=False`, forward output bit-identical to current code; (2) build a model with fresh weights at `use_grad_ckpt=True`, run a forward pass, then build an identical model at `use_grad_ckpt=False` with the same state_dict copy, run the same forward — assert `torch.allclose(out_ckpt, out_noeck, atol=1e-5)` (proves wrapping doesn't change numerics); (3) Round-1 AUC unchanged on v0 replay loaded with `use_grad_ckpt=False` (the v0 sidecar predates this flag, so `infer.py` must default to `False` when the field is absent from `train_config.json`); (4) VRAM peak reduced ≥ 25% on 2-block stack at `seq_len=256`.
- **Rollback:** set `use_grad_ckpt=False`. Wrapping code unreachable.
- **Integration:** `infer.py` — checkpointing is training-only (disabled in `model.eval()` mode); no infer-side change. `trainer.py` passes flag from config.

**Stage 1 zip:** `src/model.py` + `configs/baseline.py`.
**Stage 3 zip:** unchanged.
**Success criterion:** VRAM peak ≥ 25% lower vs non-checkpointed; training loss curve statistically identical; AUC unchanged.
**Watch out for:** `torch.utils.checkpoint.checkpoint` requires all inputs to be tensors — the `list` args (`q_tokens_list`, `seq_tokens_list`) must be unpacked into positional tensors or wrapped. Use `use_reentrant=False` (PyTorch ≥ 2.0).
**Reference:** PyTorch `torch.utils.checkpoint` docs.

---

## Phase E — Lock-in (Days 12-14, 2026-05-21 to 2026-05-23)

Phase E is the sprint mechanic, not an upgrade. Concrete enough to be a runbook.

### E1. Pick best config
Sort `submissions/*/NOTES.md` by leaderboard AUC. Top candidate is the lock-in target.

### E2. Stable retrain
Retrain the chosen config with the established seed on the full training window. Verify the resulting model's SHA256 is recorded in `submission_manifest.json` sidecar.

### E3. Pre-flight audit
```bash
cd pcvr/
python -m pytest tests/                # 37/37 must pass
python -m src.audit <ckpt_dir>          # single-model rule check
python build_step1_submission.py
python build_step3_submission.py
```

### E4. Submit by 12:00 AOE on Day 13 (2026-05-22)
4-hour buffer before the 16:00 cutoff. Sub #2 = backup config. Sub #3 = held in reserve.

---

## Critical path

```
A1 submit v0.5 (1 slot, captures probe diagnostic)
  ↓
A3 capacity diagnostic (0 slots, local) — parallel with A1
  ↓
A2 aggregation ablation (2 slots, reuses ckpt)
  ↓
                  if capacity-bound  →  Phase B (B1, B2, B3, B4) — one lever at a time
A3's verdict —
                  if signal-bound    →  Phase C (C1, C2, C3, C4) — C3 needs the 4-assertion test
  ↓
Phase D interleaved throughout (additive, low risk, must pass "AUC unchanged" test)
  ↓
E1-E4 lock-in (Days 12-14)
```

The **single highest-information action available** is A3 (local capacity diagnostic). 0 slots, decides Phase B vs Phase C.

The **single highest-information submission** is A1 (v0.5). Wired validator output on real data is irreplaceable.

---

## Submission slot budget

| Used | Date | Score |
|---|---|---|
| 1 | 2026-05-08 | pcvr v0 = **0.81144** (anchor) |
| 3 | 2026-05-10 (teammate v3/v4/v5) | 0.806681 / 0.797723 / 0.797711 (all regressed) |
| 0-3 estimated | 2026-05-10/11 (Phase A1 + A2) | TBD |

Slots used so far: 4 of ~42. Remaining over 13 days = ~3/day budget. Adequate for B/C if Phase A is done before splurging.

---

## What this roadmap does NOT decide

- Specific implementation choices inside each upgrade. The architecture frame defines BOUNDARIES; the HOW is implementer-free as long as the test contract passes.
- Which Phase B / C items to do FIRST within their phase. Choose based on Phase A's local AUC results.
- The teammate's roadmap. Their v3/v4/v5 cycle ran and regressed; future teammate work either ports a single high-value piece from their attempts or adopts cards directly from this roadmap.
- Whether to publish `jw2333-null/taac-pcvr` publicly. Stay private until Round 1 closes 2026-05-23.

---

## Update protocol

When a Phase item completes, append `✅ <commit-SHA> <date>` next to it. When a submission lands, update `submissions/README.md`'s log table. When the architecture frame proves wrong (e.g., a test contract is too lax), edit this doc in place — small revisions over rewriting v4.

When in doubt: A3 first (free), then A1, then revisit.
