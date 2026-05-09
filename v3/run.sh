#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v3 config: multi-task head + focal loss + longer training ----
# Building on v2 (transformer encoder, temporal-causal val split). The major
# changes are:
#
#   1. action_num 1 -> 2: model now has two binary heads. Head 0 = conversion
#      (label_type==2, the leaderboard target; infer.py reads this head only);
#      head 1 = click (label_type in {1,2}), trained as auxiliary supervision
#      via aux_click_weight. Free signal that previously was dropped into the
#      negative bucket.
#
#   2. loss_type bce -> focal (alpha=0.25, gamma=2.0) on the conversion head.
#      Addresses the ~12% positive imbalance; alpha=0.25 is a more
#      conservative pos-class downweight than the default 0.1.
#
#   3. Training length: num_epochs 3 -> 5, patience 2 -> 3, valid_ratio
#      0.05 -> 0.10. v2 inference ran in 261s, suggesting the transformer
#      converged early; more epochs + a less aggressive early-stop give
#      it room to use the new auxiliary signal, and a 2x larger val set
#      makes early-stop decisions less noisy.
#
# Inference cost: barely changed (one extra Linear column in the head);
# still well under the 1800s budget.
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
    --num_epochs 5 \
    --patience 3 \
    --valid_ratio 0.10 \
    --action_num 2 \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --aux_click_weight 0.3 \
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
