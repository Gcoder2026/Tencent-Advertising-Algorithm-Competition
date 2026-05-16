#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v13.2 GPU-contention mitigations (escalated after three OOM failures
# on the shared Angel pool):
#
# Failure progression:
#   v13:    OOM at model-load (.to(device)) — only 326 MiB allocated
#   v13.1:  OOM during forward (F.scaled_dot_product_attention) —
#           PyTorch had 10.9 GiB allocated, then tried for another
#           1024 MiB for the seq_c/seq_d (length 512) attention score
#           matrix and failed. v13.1's host-side mitigations DID work
#           (model loaded), but attention activation memory is the new
#           wall.
#
# v13.2 mitigations (in order of impact):
#
#   - seq_max_lens 256/256/512/512 -> 128/128/256/256 (v1's recipe)
#       Attention memory = O(batch * heads * seq_len^2). Halving the
#       long sequences (512 -> 256) cuts seq_c/seq_d attention memory
#       4x. v1 used these exact lengths and fit comfortably. Cost in
#       AUC: estimated -0.005 (v6 vs v1 delta from this lever alone).
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       Carried over from v13.1. Reduces fragmentation. Confirmed
#       useful from v13.1's progress past the model-load OOM.
#   - --num_workers 4 (was 8)
#       Carried over from v13.1. Host-RAM headroom.
#   - --buffer_batches 10 (was default 20)
#       Carried over from v13.1. Smaller shuffle buffer.
#
# Net AUC expectation vs v6 (0.81287):
#   Cyclical time lever:   +0.005 to +0.011
#   Shorter seq lens:      -0.005 to -0.007
#   Net:                   ~0 to +0.006 -> 0.813 to 0.819
# Optimistic ceiling now lower than original v13, but training will
# actually complete on this contended pool.

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
    --batch_size 256 \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 1 \
    --use_cyclical_time \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
