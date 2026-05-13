# PCVR Roadmap v2 — post-merge reality (originally 2026-05-09, refreshed 2026-05-10)

**Supersedes:** `2026-05-08-pcvr-roadmap-v1.md` (the original 14-day sprint roadmap, written assuming a 3-person team with sole project ownership). Roadmap v1's v0 anchor + 5-agent specialist findings + critic-integrated revisions remain valid where not contradicted here.

**Purpose of this doc:** keep the team aligned on **direction** for each upgrade. Implementation detail is intentionally left flexible — whoever picks up an item (you, teammate, or a fresh Claude Code session) gets to choose the exact approach, as long as the direction in the upgrade card matches.

---

## Updated context

- **Today: 2026-05-10.** Round 1 deadline: 2026-05-23 AOE → **13 days remaining**.
- **Team-size correction:** roadmap v1 assumed 3 students × 3 streams. Actual: **you alone + one teammate (Gcoder2026)** working in parallel. The streams collapse to one-thing-at-a-time for you; the teammate runs their own track and we port single high-value items as they emerge.
- **Cross-team flow established:** local `main` → push → `jw2333-null/taac-pcvr` (private solo). Local `pcvr-jw2333` → push → `Gcoder2026/.../main` (merged; now the canonical default branch on the team repo).
- **Phase 1 of merge plan v1.1 applied** — 37/37 tests pass on conda env `taac`. v0.5 zips archived under `submissions/2026-05-09_v0.5_post-phase1/` — **not yet submitted**.
- **Anchor scores:** v0 = AUC **0.81144** (leaderboard). Teammate's v1 = AUC 0.807.
- **Open diagnosis:** local valid AUC ~0.86 → leaderboard 0.81144 = ~0.05 generalization gap. Most likely sequence-history leakage; wired `sequence_history_leak_probe` will report on the next training run.

## Recent milestones (2026-05-10 refresh)

- ✅ **GitHub attribution fix** — all 45 of your commits rewritten via `git filter-repo` from a malformed email to the correct GitHub-privacy format. Force-pushed to both remotes with `--force-with-lease`. Contributors page now shows `jw2333-null: 45`, `Gcoder2026: 6`.
- ✅ **Teammate sync confirmed** before the rewrite (no work overwritten).
- ✅ **Local git config corrected** — future commits attributed automatically.
- 🟨 **Stale SHA references** in older docs (e.g. `c565b36` in merge plan v1.1) are now orphaned by the rewrite. Treat them as historical labels.

---

## How to read an upgrade card

Every upgrade item below uses the same seven fields. The first five are about WHERE this upgrade goes and WHY. The last two name the artifacts the upgrade ultimately ships — important because the team submits TWO zips per round (training + inference), and not every upgrade touches both.

```
Direction        — what kind of change (architecture / data / loss / training / inference)
Why              — the hypothesis (rooted in PCVR or general ML wisdom)
Success criterion — the concrete number that must move for this to count
Watch out for    — known blockers, gating conditions, risks
Reference        — published technique or paper (when applicable)
Stage 1 zip      — what changes in step1_train.zip
Stage 3 zip      — what changes in step3_infer.zip
```

**The four common patterns for the zip-impact pair:**

| Upgrade type | Stage 1 zip | Stage 3 zip |
|---|---|---|
| Config-only ablation | new config | unchanged |
| Training-side change | source + config change | unchanged OR small loader fix |
| Inference-side change | unchanged | source + config change |
| Memory / latency optimization | maybe training, maybe inference, never both | maybe training, maybe inference, never both |

---

## Phase A — Anchor diagnosis (do these BEFORE any architectural work)

The 0.05 generalization gap is the most expensive open question. Resolve it before spending submission slots on architecture or feature work.

### A1. Submit v0.5 to capture validator diagnostics on real data

**Direction:** diagnostic submission against the canonical platform data.

**Why:** the validators we wired in Phase 1 (`sequence_history_leak_probe`, `oob_rate_check`) report on real data only when training runs end-to-end on the platform. This submission is the cheapest way to learn whether the 0.05 gap is leakage (probe will return `n_future_events > 0`) or distribution shift (probe will return 0). The leaderboard score is a secondary signal.

**Success criterion:** validator output is captured in the platform's Step-1 log. AUC within ±0.005 of 0.81144 is acceptable — the diagnostic is the real prize, not a score bump.

**Watch out for:** burns one submission slot; if AUC drops > 0.005 below the anchor, the v0.5 changes regress somewhere and need to be unwound.

**Stage 1 zip:** already built and archived under `submissions/2026-05-09_v0.5_post-phase1/step1_train.zip`. Upload as-is.

**Stage 3 zip:** already built and archived under the same folder. Upload after Step 2 (Export Model) publishes the new checkpoint.

### A2. Aggregation-strategy ablation on the published checkpoint

**Direction:** inference-side experiment — vary how per-user scores are aggregated.

**Why:** `infer.py` currently aggregates by `mean` across multiple rows per `user_id`. If the test set has multiple rows per user, the choice (mean / max / last-by-timestamp) can move AUC by 0.005–0.020 with zero retraining. Cheapest "free AUC" experiment in the entire roadmap.

**Success criterion:** a non-mean aggregation beats current mean by ≥ 0.002 → adopt it. Otherwise: keep mean and close the question.

**Watch out for:** burns 2 Step-3 slots. Reuses the published v0 checkpoint (no Step-1 re-run needed).

**Stage 1 zip:** unchanged.

**Stage 3 zip:** ships a new inference variant with the aggregation strategy selectable via config; submit each variant separately.

### A3. Local capacity diagnostic — is the model under-parameterized?

**Direction:** local-only experiment — sweep model capacity at training time and compare convergence speed and ceiling.

**Why:** v0's training plot shows valid AUC plateau at ~step 4k out of 26k. That's a strong signal the model is either capacity-bound (more parameters would help) or signal-bound (no amount of parameters helps without better features). Distinguishing the two is the SINGLE highest-information action available — it determines whether to invest in Phase B (architecture) or Phase C (data/signal).

**Success criterion:** doubling capacity moves local 4k-step valid AUC by ≥ 0.005 → capacity-bound → execute Phase B next. Else → signal-bound → execute Phase C next.

**Watch out for:** local environment is CPU-torch unless you set up a GPU-enabled env first. Runs slower but free.

**Stage 1 zip:** not produced (no submission).

**Stage 3 zip:** not produced.

**Phase A budget: up to 3 submission slots, ~4 hours of waiting.**

---

## Phase B — Architecture levers (if Phase A says capacity-bound)

### B1. Self-attention sequence encoder

**Direction:** swap the current per-position FFN sequence encoder for one that uses attention across sequence positions.

**Why:** position-to-position relationships in user behavior sequences (e.g. "clicked X right after viewing Y") carry signal that a position-independent encoder structurally cannot capture. Attention is the canonical way to model these.

**Success criterion:** AUC improves over the SwiGLU baseline by a meaningful margin on local valid (check before any leaderboard submission).

**Watch out for:** attention is heavier than per-position FFN. Memory and latency may force trade-offs (shorter sequences, smaller batch) that erode some of the gain.

**Reference:** Vaswani et al. 2017 (standard transformer self-attention).

**Stage 1 zip:** new experiment config selecting the transformer encoder. Existing training code already supports it.

**Stage 3 zip:** unchanged — the encoder choice rides with the checkpoint metadata.

### B2. Rotary position encoding (RoPE)

**Direction:** add positional encoding to whichever attention paths exist in the model.

**Why:** any attention-based encoder is permutation-invariant by default. RoPE injects positional information via rotation in the query/key space, capturing order without inflating parameter count. Particularly relevant alongside B1.

**Success criterion:** RoPE-on vs RoPE-off, local AUC delta ≥ 0.002. Compose with B1 if both apply.

**Watch out for:** only meaningful where attention is actually used. If the SwiGLU encoder is kept (B1 not applied), RoPE only affects the cross-attention path → marginal effect.

**Reference:** Su et al. 2021 (RoFormer).

**Stage 1 zip:** new config enabling RoPE.

**Stage 3 zip:** unchanged.

### B3. Cold-restart sparse re-initialization

**Direction:** training-loop intervention — periodically reset high-cardinality embedding tables.

**Why:** embeddings for rare long-tail IDs collect noise across epochs and overfit faster than embeddings for frequent IDs. Periodic reset (with optimizer-state preservation for low-cardinality params, so they don't lose accumulated signal) breaks the overfit pattern. The teammate's `v2/trainer.py` has a working implementation we can study.

**Success criterion:** AUC improves by ≥ 0.001–0.003 with re-init enabled.

**Watch out for:** the re-init fires at epoch boundaries. If training early-stops within 1–2 epochs (likely on full-scale data), the hook never executes — implementation effort is wasted. Verify the v0 training reached ≥ 3 full epochs before scheduling this.

**Reference:** KuaiShou "MultiEpoch: Reusing Training Data" trick.

**Stage 1 zip:** training-side code change (new trainer hook + config field).

**Stage 3 zip:** unchanged — re-init is a training-only concern; the final model is still a single state_dict.

### B4. Capacity bump

**Direction:** scale the model — larger embeddings, larger hidden dimension.

**Why:** if Phase A diagnoses capacity-bound, more parameters give the model representational headroom for finer-grained patterns. The diagnostic literally tells us this.

**Success criterion:** scaled-up model beats current baseline by ≥ 0.005 AUC locally.

**Watch out for:** 19 GiB GPU is the hard constraint; the model also has a structural divisibility constraint between `d_model` and the total token count `T`. Verify the new size fits both before training.

**Stage 1 zip:** new config with the larger dimensions.

**Stage 3 zip:** unchanged — model loader handles arbitrary sizes from the checkpoint's training-config sidecar.

---

## Phase C — Data + signal levers (if Phase A says signal-bound)

### C1. DIN-style target-attention query

**Direction:** architectural change to the cross-attention path — replace learned "NS query" tokens with queries derived from the target item embedding.

**Why:** the current model uses generic learned tokens to query over user sequences, which effectively does "summarize this user's history" regardless of what we're predicting. DIN replaces this with target-aware queries — the model focuses on past behaviors relevant to the CURRENT target item. In CTR/CVR literature this is one of the strongest single architectural levers.

**Success criterion:** local AUC improves by ≥ 0.003 (literature reports +0.005–0.015 in comparable systems).

**Watch out for:** changes the cross-attention path in `PCVRHyFormer`. Medium implementation complexity; non-trivial regression risk if the query shape is wrong.

**Reference:** Zhou et al. 2018 (DIN); Zhou et al. 2019 (DIEN).

**Stage 1 zip:** training-side model change + new config.

**Stage 3 zip:** small loader update — model architecture change may shift state-dict keys.

### C2. Multi-task click head (CTR auxiliary loss)

**Direction:** training-side multi-task supervision — add a second sigmoid head predicting clicks (label_type==1) alongside the main conversion head (label_type==2).

**Why:** the platform's `label_type` column has values 0/1/2 (exposure / click / conversion). Right now we only supervise on the conversion signal. Adding click as an auxiliary task gives the shared encoder more gradient information per row and acts as implicit regularization. Standard multi-task setup; ESMM is the canonical reference for the conversion-rate variant.

**Success criterion:** AUC on the conversion head improves by ≥ 0.001 with auxiliary loss enabled (typical aux-loss weight: 0.1× the main loss).

**Watch out for:** prerequisite is confirming the training data actually emits click labels — `label_type == 1` rows must exist in non-trivial numbers. Aux-loss weight tuning matters; too high and conversion AUC regresses.

**Reference:** Ma et al. 2018 (ESMM — Entire Space Multi-task Model).

**Stage 1 zip:** training-side model change (new head) + loss change + new config for the aux-loss weight.

**Stage 3 zip:** typically unchanged — inference reads only the conversion-head logit. The click head exists in the model but is ignored at scoring.

### C3. Continuous log-Δt time encoding

**Direction:** feature-encoding change — replace bucketed time deltas with a continuous function of the delta.

**Why:** the current 64-bucket searchsorted-based time embedding throws away the continuous structure of time (within-bucket ordinal information is lost — "5 minutes ago" and "50 minutes ago" can land in similar buckets). Recency-and-frequency signals are dominant in PCVR; a continuous encoding preserves them.

**Success criterion:** local AUC improves by ≥ 0.001.

**Watch out for:** changes both the dataset path and the model's time-embedding consumer. Must verify the synthetic-data fixture still produces sensible sequences after the change (the conftest fixture was already once silently bugged on this axis — see merge plan v1.1).

**Stage 1 zip:** training-side dataset + model change + new config.

**Stage 3 zip:** model architecture change visible at load — inference loader updates accordingly.

### C4. Target × user-profile cross features

**Direction:** feature engineering near the classifier head — add explicit pairwise interactions between target-item features and user-profile features.

**Why:** the HyFormer backbone attends over sequences but doesn't explicitly compute target-item-vs-user-profile interaction terms at the prediction head. Hadamard products of selected features inject these classical CTR-system signals.

**Success criterion:** local AUC improves by ≥ 0.001–0.003.

**Watch out for:** adds parameters at the head; can overfit if many cross terms are added without regularization. Start with a small number of high-confidence pairs.

**Stage 1 zip:** training-side model change (new layer near the classifier) + config.

**Stage 3 zip:** small loader update — model architecture change.

---

## Phase D — Round-2 prep (interleave during Phases B/C; additive only)

Round 2 (May 25 – Jun 24) has 10× the data and a stricter latency tiebreaker. These items are insurance: they don't have to deliver AUC improvements; they just have to not regress AUC while preparing the system for Round 2 scale.

### D1. Inference latency benchmark

**Direction:** observability tool — measure baseline inference latency on the v0 checkpoint.

**Why:** Round 2's tiebreaker is latency. We can't optimize what we don't measure; a baseline number is the prerequisite for every other latency-optimization decision.

**Success criterion:** a script that reports inference latency in ms/row, run once, baseline value recorded.

**Stage 1 zip:** unchanged.

**Stage 3 zip:** unchanged — the benchmark is a separate local script, not submitted.

### D2. `inference_mode` swap

**Direction:** inference-side micro-optimization — swap `torch.no_grad()` for `torch.inference_mode()`.

**Why:** `inference_mode` skips view tracking in addition to grad computation, slightly faster than `no_grad`. Free per-batch win; matters cumulatively at 10× data.

**Success criterion:** zero AUC change, measurable latency reduction.

**Stage 1 zip:** unchanged.

**Stage 3 zip:** inference-code change (a context-manager swap; mechanically trivial).

### D3. Sharded parquet streaming

**Direction:** data-loading rewrite for scale — swap full-file reads for sharded streaming via `pyarrow.dataset`.

**Why:** at 10× data, current full-file parquet scans may exhaust memory or stall. Sharded streaming reads only what's needed per shard.

**Success criterion:** identical AUC on Round 1 data + no OOM / no stall on Round 2 data.

**Watch out for:** high blast-radius change to the dataset path. Defer until after Round 1 closes (2026-05-24).

**Stage 1 zip:** training-side data-loading change.

**Stage 3 zip:** inference-side data-loading change if eval data is also multi-file.

### D4. Embedding-table pruning

**Direction:** size optimization — keep only the top-N most-frequent IDs per high-cardinality feature.

**Why:** at 1M vocab × 64 dim × fp32, each embedding table is ~256 MB. Multiple tables push past the 100 MB upload cap. Pruning to top-500K typically preserves the bulk of the signal at half the size.

**Success criterion:** AUC delta ≥ −0.002 after pruning; zip stays under 100 MB cap.

**Watch out for:** aggressive pruning can drop tail-ID information unexpectedly; validate on a local holdout first.

**Stage 1 zip:** training-side embedding-table change + new config flag.

**Stage 3 zip:** unchanged — model loader handles the smaller tables transparently.

### D5. Gradient checkpointing on HyFormer blocks

**Direction:** training-side memory optimization — checkpoint activations within HyFormer blocks to free VRAM.

**Why:** enables longer `seq_max_lens` within the same 19 GiB GPU budget. Longer sequences capture more behavior history.

**Success criterion:** longer sequences become possible without OOM; AUC ideally improves with longer history.

**Stage 1 zip:** training-side trainer code change + config flag.

**Stage 3 zip:** unchanged — gradient checkpointing is a training-only concern.

---

## Phase E — Lock-in (Days 12–14, 2026-05-21 to 2026-05-23)

Phase E is the sprint mechanic, not an upgrade. Concrete enough to be a runbook.

### E1. Pick best config
Sort `submissions/*/NOTES.md` by leaderboard AUC. Top candidate is the lock-in target.

### E2. Stable retrain
Retrain the chosen config with the established seed on the full training window. Verify the resulting model's SHA256 is recorded somewhere durable (e.g. `submission_manifest.json` sidecar in the checkpoint).

### E3. Pre-flight audit

```bash
cd pcvr/
python -m pytest tests/                # 37/37 must pass
python -m src.audit <ckpt_dir>          # single-model rule check
python build_step1_submission.py        # archive the Stage-1 zip
python build_step3_submission.py        # archive the Stage-3 zip
```

### E4. Submit by 12:00 AOE on Day 13 (2026-05-22)
4-hour buffer before the 16:00 cutoff. Sub #2 = backup config. Sub #3 = held in reserve.

---

## Critical path

```
A1 submit v0.5 (1 slot, captures probe diagnostic)
  ↓
A3 capacity diagnostic (0 slots, local) — runs in parallel
  ↓
A2 aggregation ablation (2 slots, reuses ckpt)
  ↓
                  if capacity-bound  →  Phase B (B1, B2, B3, B4)
A3's verdict —
                  if signal-bound    →  Phase C (C1, C2, C3, C4)
  ↓
Phase D interleaved throughout (additive, low risk)
  ↓
E1–E4 lock-in (Days 12–14)
```

The **single highest-information action available right now** is A3 (local capacity diagnostic). It needs no submission slots and tells you which Phase (B vs C) to invest in.

The **single highest-information submission** is A1 (v0.5 with wired validators). The validator outputs on real data are irreplaceable.

---

## Submission slot budget (refreshed 2026-05-10)

| Used | Date | Score |
|---|---|---|
| 1 | 2026-05-08 (v0 baseline) | 0.81144 |
| 0 | 2026-05-09 | (no submissions; merge work + email rewrite consumed the day) |
| 0–3 estimated | 2026-05-10 (Phase A1 + A2 = up to 3 slots) | TBD |

Slots used so far: **1 of ~42**. Slots remaining over 13 days = ~3/day budget. Phase A can plausibly run today; Phases B/C still have plenty of runway.

---

## What this roadmap does NOT decide

- The specific implementation choices for each upgrade. Whoever picks up an item gets to choose libraries, file organization, and exact APIs — as long as the direction in the upgrade card holds.
- Which Phase B / C items to do FIRST within their phase. Choose based on local AUC results from earlier items.
- The teammate's roadmap. They have v3-v6 planned per their README; if their `v3+` ships a successful upgrade, we port the single high-value piece — we don't try to replicate their whole branch.
- Whether to publish `jw2333-null/taac-pcvr` publicly. Stay private until Round 1 closes 2026-05-23.

---

## Update protocol

When a Phase item completes, append `✅ <new-SHA> <date>` next to it. When a submission lands, update `submissions/README.md`'s log table. When evidence (e.g. local AUC results) changes the priority order, edit this doc — don't write a v3 yet; small revisions in place are fine until the post-Phase-A pivot.

When in doubt: A3 first (it's free), then revisit this doc.
