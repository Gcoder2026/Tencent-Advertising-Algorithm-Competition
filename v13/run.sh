#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v13.3 GPU-contention mitigations (escalated after four failures on
# the shared Angel pool):
#
# Failure progression:
#   v13:    OOM at model-load (.to(device)) — only 326 MiB allocated
#   v13.1:  OOM during forward (F.scaled_dot_product_attention) —
#           PyTorch had 10.9 GiB allocated, then needed another 1024 MiB
#           for the seq_c/seq_d attention score matrix.
#   v13.2:  Silent kill ~57s into training (no Python traceback —
#           kernel-level OOM kill bypassing PyTorch). Likely host RAM
#           or cgroup memory limit, since v13.1's GPU mitigations
#           reduced GPU pressure but host pressure is independent.
#
# v13.3 final mitigation: drop batch_size 256 -> 128. This halves
# every per-batch activation tensor (attention scores, embedding
# lookup outputs, transformer hidden states, gradients) AND halves
# DataLoader pinned-memory traffic. It is the single most impactful
# memory reduction available without changing model architecture.
#
# Active mitigations (carried over from v13.1/v13.2):
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - --num_workers 4 (was 8) — host-RAM headroom
#   - --buffer_batches 10 (was default 20) — smaller shuffle buffer
#   - --seq_max_lens 128/128/256/256 (was 256/256/512/512) — cuts
#     attention activation memory 4x on long sequences
#   - --batch_size 128 (was 256) — halves all activation memory
#
# Net AUC expectation vs v6 (0.81287):
#   Cyclical time lever:    +0.005 to +0.011
#   Shorter seq lens:       -0.005 to -0.007
#   Smaller batch (256->128): ~neutral (-0.001 to +0.002)
#   Net:                    ~0 to +0.006 -> 0.813 to 0.819
# Lower optimistic ceiling than original v13 but should actually
# complete on this contended pool.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- v13 config: v6 recipe + ROBUST cyclical time features ----
#
# Lever: row-level cyclical time embeddings (hour-of-day + day-of-week +
# month-of-year) added additively to user_dense_tok. Community consensus
# on 小红书 across multiple independent posters says this lever gives
# +0.011 AUC on this competition.
#
# Why v10 failed and v13 should not:
#   v10 only added (hour + day_of_week) and used flag-conditional module
#   construction. Hypothesis for v10's eval-step crash: state-dict
#   mismatch between train (modules present) and eval (modules absent
#   because the flag wasn't honored on the eval container).
# v13 structural fixes:
#   1) Always construct hour/day/month embedding modules — flag only
#      controls APPLICATION at forward time, not construction. Eliminates
#      any state-dict drift between training and inference.
#   2) Add month-of-year (the third cyclical channel from 小红书 posts).
#   3) Defensive feature derivation in infer.py: if dataset doesn't emit
#      hour_of_day/day_of_week/month_of_year, derive them from the
#      timestamp field. Guarantees features ALWAYS reach the model.
#
# Recipe base = v6 (our 0.81287 best):
#   - transformer seq encoder, num_queries=2
#   - rankmixer NS tokenizer with 5 user + 2 item tokens
#   - seq_max_lens 256/256/512/512
#   - reinit_sparse_after_epoch=1 (the MultiEpoch KuaiShou trick)
#   - emb_skip_threshold=1000000
#   - DEFAULT num_epochs/patience/valid_ratio (do NOT override — the
#     platform's defaults are what produced v6's good run)
#
# Expected outcome:
#   - Best case (community consensus holds): +0.011 → 0.823
#   - Realistic with platform variance: +0.005 to +0.011 → 0.818-0.823
#   - Worst case (regression): ~ -0.005 within variance → no harm
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --seq_encoder_type transformer \
    --seq_truncate auto \
    --seq_max_lens seq_a:128,seq_b:128,seq_c:256,seq_d:256 \
    --batch_size 128 \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 1 \
    --use_cyclical_time \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
