#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v8 config: v6/v7 recipe + C1 DIN target-attention ----
# Single lever vs v6 (team's best 0.81287): --use_din_query replaces the
# global mean-pool in MultiSeqQueryGenerator with target-aware attention
# in DINQueryGenerator. Output signature unchanged so downstream HyFormer
# blocks are unaffected. Expected +0.005 to +0.015 per Zhou 2018 DIN.
# Notably: --use_continuous_time is OFF — v7 proved it hurt -0.0074;
# the 65-bucket discrete time embedding has regularizing benefit we
# don't want to lose. All other knobs match v6's empirical winner.
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
    --use_din_query \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 1 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
