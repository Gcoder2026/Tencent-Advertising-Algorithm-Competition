#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v14.1: focal loss + v13.4's PROVEN-FIT memory recipe
#
# v14 attempted v6's recipe + focal but OOM'd at 12.37 GiB allocated.
# Platform GPU pool is currently heavily contended (5 neighbor
# processes using 95+ GiB), our budget capped at ~12 GiB. v6's
# recipe peak is ~15 GiB — won't fit tonight.
#
# v13.4 trained successfully a few hours ago with the smaller recipe
# (batch 128, seq 128/256). We accept that recipe as our floor and
# test focal loss on top.
#
# Trade-off:
#   - v13.4 baseline: 0.797712 (significantly below v6's 0.81287)
#   - + focal: +0.005 to +0.008
#   - Expected v14.1: 0.803 to 0.806
#   - Still below v6 ceiling, but tests if focal is a real lever.
#     If yes, we re-test v6 recipe + focal another day when platform
#     is less contended (likely better numbers then).
#
# Why focal loss specifically:
#   - Positive rate is 12.4%. Mild but real class imbalance.
#   - All 13 prior submissions used vanilla BCE.
#   - gamma=2 focuses gradient on hard examples (confident-but-wrong).
#   - alpha=0.75 upweights positives (reading utils.py: alpha_t =
#     alpha*targets + (1-alpha)*(1-targets), so alpha=0.75 means
#     positives get weight 0.75, negatives 0.25).
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
    --seq_max_lens seq_a:128,seq_b:128,seq_c:256,seq_d:256 \
    --batch_size 128 \
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
