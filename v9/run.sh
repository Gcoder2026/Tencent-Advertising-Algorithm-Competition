#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v9 config: v6's exact recipe + dropout regularization bump ----
#
# Strategy: v6 produced 0.81287 with v1's PLAIN code (no continuous time,
# no DIN, no other architectural changes I tried). The val→test gap is
# the dominant problem: local val converges to ~0.86, platform AUC sits
# at ~0.81. That 0.05 gap is the documented overfitting signature.
#
# v9 is v6's EXACT recipe + one targeted change: --dropout_rate 0.05
# (5x increase from the 0.01 default). Pure config — zero new code,
# zero new trainable parameters. Same flat v1 codebase that has worked
# every time the platform actually ran it.
#
# v6's recipe (reconstructed from the platform Args dump):
#   * NO --num_epochs / --patience / --valid_ratio overrides
#     → train.py defaults apply: num_epochs=999, patience=5, valid_ratio=0.1
#   * --seq_encoder_type transformer (not swiglu)
#   * --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 (doubled)
#   * --reinit_sparse_after_epoch 1 (sparse cold-restart each epoch, ON)
#   * --emb_skip_threshold 1000000 (skip very-high-card embeddings)
#
# Expected outcome:
#   v9 ≥ 0.815  dropout reg narrowed the val/test gap → real climb
#   v9 ≈ 0.812  dropout neutral → try weight_decay next as v10
#   v9 < 0.81   dropout too aggressive → try smaller (0.025) as v10
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
    --dropout_rate 0.05 \
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
