#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v13.4: drop datetime64 from the cyclical-time path
#
# Failure progression:
#   v13:    OOM at model-load (.to(device)) — only 326 MiB allocated
#   v13.1:  OOM during forward (F.scaled_dot_product_attention) —
#           PyTorch had 10.9 GiB allocated.
#   v13.2:  Silent kill ~57s into training (no Python traceback)
#   v13.3:  Silent kill again, even with batch_size=128 and 16x less
#           attention memory than v6. THIS is what shifts the diagnosis
#           from "platform contention" to "v13 has a code bug".
#
# Strongest suspect: the dataset.py datetime64 conversion that runs in
# every DataLoader worker:
#     ts64 = timestamps.astype('datetime64[s]')
#     month = (ts64.astype('datetime64[M]').astype(np.int64) % 12)
# numpy datetime64 has platform-dependent behaviour on edge values
# (nulls, sentinels, NaT). A worker that segfaults inside numpy
# native code triggers exactly the silent-kill pattern we see — the
# trainer hangs/exits without a Python traceback.
#
# v13.4 strips this completely:
#   - Drop month_of_year emission, embedding application, infer.py
#     fallback. month_embedding module is STILL constructed in the
#     model so the state-dict shape stays stable across train/eval.
#   - Cyclical time path is now pure integer arithmetic only:
#       hour_of_day = (ts % 86400) // 3600
#       day_of_week = (ts // 86400) % 7
#     Both run on plain int64, no numpy dtype gymnastics.
#   - 小红书 community claims +0.011 from "time features"; dropping
#     month likely costs little because the training set spans only
#     weeks not years, so month is near-constant anyway.
#
# Memory mitigations from v13.1/v13.2/v13.3 are RETAINED as safety
# belt (we still saw real GPU OOMs early in the sequence, so the pool
# IS contended even if not the proximate cause of v13.2/v13.3):
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - --num_workers 4 (was 8) — host-RAM headroom
#   - --buffer_batches 10 (was default 20)
#   - --seq_max_lens 128/128/256/256 (was 256/256/512/512)
#   - --batch_size 128 (was 256)
#
# Net AUC expectation vs v6 (0.81287):
#   hour + day-of-week lever:  +0.003 to +0.008 (slightly less than
#                              full hour+day+month lever)
#   Shorter seq lens:          -0.005 to -0.007
#   Smaller batch (256->128):  ~neutral
#   Net:                       ~-0.002 to +0.001 -> 0.811 to 0.814
# Lower upside than original v13 plan but at least training should
# complete this time.

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
