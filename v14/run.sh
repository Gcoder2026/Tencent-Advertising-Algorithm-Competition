#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v14: v6's EXACT winning recipe + focal loss (single-variable test)
#
# What we learned from v13.4 (0.797712) vs v6 (0.81287):
#   - The cyclical-time community claim ("+0.011") did NOT materialize.
#     hour+day cyclical was at best neutral, possibly slightly negative.
#   - The seq_max_lens cut (256/512 -> 128/256) cost real AUC (~ -0.01).
#   - 13 submissions in, v6's recipe is still our ceiling.
#
# v14 strategy: restore v6's recipe exactly, then add ONE new lever
# that no prior submission tried — focal loss for the class-imbalance
# problem. Single-variable test: if v14 beats v6, the gain is purely
# from focal. If not, focal is not the answer and we move on.
#
# Why focal loss specifically:
#   - Positive rate is 12.4%. Mild but real class imbalance.
#   - All 13 prior submissions used vanilla BCE.
#   - gamma=2 focal modulation focuses gradient on hard examples —
#     exactly what helps when the model is confident but wrong.
#   - alpha=0.75 upweights the rare positive class. Reading utils.py:
#     alpha_t = alpha*targets + (1-alpha)*(1-targets), so alpha=0.75
#     means positives get weight 0.75 and negatives get 0.25 (the
#     opposite of the docstring's example, which assumed positives
#     dominate).
#   - Standard +0.003 to +0.008 on imbalanced binary classification.
#
# Cyclical time DISABLED (--no_cyclical_time) for a clean single-
# variable test. The module is still constructed in PCVRHyFormer
# (state-dict stable) but skipped in the forward path.
#
# Memory safety belt RETAINED (light touch, not aggressive like v13.3):
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - --num_workers 4 (was 8 in v6) — host RAM safety
#   - --buffer_batches 10 (was 20 default) — smaller shuffle buffer
# But SEQ_MAX_LENS and BATCH_SIZE go back to v6's values: any memory
# savings v13.x got from cutting those came at real AUC cost.
#
# Expected outcome vs v6 (0.81287):
#   focal lever:    +0.003 to +0.008
#   Net:            +0.003 to +0.008 -> 0.816 to 0.821
# Hits the band where top-50 (0.83) becomes a 2-3 lever stack away.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
    --loss_type focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --no_cyclical_time \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
