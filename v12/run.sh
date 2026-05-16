#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v12 config: v6 recipe + capacity bump (d_model & emb_dim 64->128) ----
#
# Pure config, zero new code. v1 codebase byte-identical except run.sh.
#
# Rationale: after 11 submissions, pattern shows our 0.808-0.813 ceiling
# is dominated by run-to-run variance. Architectural additions all
# regressed; hyperparameter tweaks land within noise of each other.
# Top 50 needs 0.83 — +0.018 we can't reliably get with the current
# architecture size.
#
# Capacity bump is the last untapped pure-config lever:
#   --d_model 64 -> 128      (4x dense capacity per layer)
#   --emb_dim 64 -> 128      (2x sparse embedding width)
#
# T constraint check: T = num_queries*num_sequences + num_ns =
#                         2*4 + 8 = 16. d_model=128 % 16 = 0  ✓
# Dense params: ~2.5M -> ~10M
# Sparse params: ~237M -> ~474M
# Total params: ~240M -> ~485M
#
# Inference time will roughly double (was 381s for v6 → expect ~700s).
# Still under the 1800s budget.
#
# Risk: bigger model could either help (real capacity gain) or hurt
# (already overfitting val, more capacity overfits more). It's a real
# coin flip. But it's the only pure-config lever we haven't pulled.
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
    --d_model 128 \
    --emb_dim 128 \
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
