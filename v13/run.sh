#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v13.1 GPU-contention mitigations (added after two OOM failures on the
# shared Angel pool):
#
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       Reduces fragmentation when the resident-set ramps. The PyTorch
#       OOM error message explicitly recommends this for our exact case.
#   - --num_workers 4 (was 8)
#       Each DataLoader worker carries its own buffer + pinned host RAM.
#       On a shared node where neighbors are pressuring host memory,
#       4 workers leaves more headroom for the model itself. Throughput
#       impact is small on this size of training set; we are
#       GPU-compute bound not data-loader bound.
#   - --buffer_batches 10 (was default 20)
#       Halves the shuffle buffer's pinned-memory footprint.
#
# Neither change touches the model architecture, so v13.1 results are
# directly comparable to a hypothetical successful v13 run.

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
    --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
    --batch_size 256 \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 1 \
    --use_cyclical_time \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
