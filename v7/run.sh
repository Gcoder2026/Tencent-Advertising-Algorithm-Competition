#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v7 config: v1 flat layout + C3 continuous time encoding ----
# Single lever vs v1: --use_continuous_time replaces the discrete 65-bucket
# time embedding with a bias-free Linear(1, d_model) projection of
# log(1 + (label_ts - event_ts)). Padding emits exactly zero (no v4-style
# LayerNorm-bias leak). All other knobs match v1 EXCEPT we trust the
# platform's default training schedule (num_epochs=999, patience=5, etc.)
# rather than the original --num_epochs 3 --patience 2, since the v6 result
# showed that longer training of ~11 epochs is what helped.
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
    --use_continuous_time \
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
