#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v2 config: same as v1 but seq_encoder_type swiglu -> transformer ----
# v1 used SwiGLU (no attention) for the per-domain sequence encoder; this version
# switches to standard Pre-LN self-attention so behavior sequences actually get
# sequential modelling. Cost: ~2x sequence-encoder compute, well inside the
# 1800s inference budget (v1 ran in 381s).
# Independent change shipped with v2: dataset.py now does a timestamp-sorted
# train/valid split (see get_pcvr_data) so local val AUC is temporally causal.
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
    --num_epochs 3 \
    --patience 2 \
    --valid_ratio 0.05 \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 999 \
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
