#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# v15: minimum-compromise path back to v6's big-recipe regime
#
# Pattern from 15 prior submissions is unambiguous:
#   BIG recipe (seq 256/512, batch 256):    0.808 - 0.813 cluster
#   SMALL recipe (seq 128/256, batch 128):  0.7977 cluster
#                                           (4 parts per million spread!)
#
# Every "feature" we tried at the small recipe moved AUC by literally
# zero parts per thousand:
#   focal vs BCE at small recipe:   0.797711 vs 0.797712  (1 ppm)
#   cyclical vs no cyclical:        0.797712 vs 0.797715  (3 ppm)
#
# The ONLY lever that moves AUC is the memory recipe. The +0.015 from
# small -> big is bigger than any feature lever we've ever tested.
#
# v15 strategy: stop adding features. Get back to the big recipe with
# minimum compromise that still fits the platform's current ~12 GB
# pod budget. The only trim is the LONG sequence lengths (seq_c,
# seq_d) from 512 -> 384. Attention memory scales as seq_len^2, so
# the long-sequence cut is the highest-leverage memory saver per
# lost AUC point.
#
# Memory math:
#   v6 big:           batch 256, seq 256/256/512/512 -> ~15 GB (OOMs now)
#   v14 OOM:          batch 256, seq 256/256/512/512 -> died at 12.37 GiB
#   v15:              batch 256, seq 256/256/384/384 -> ~10-11 GB peak
#   v13.4 small:      batch 128, seq 128/128/256/256 -> ~10 GB (fits)
#
# Long-seq cut effect: 1 - 384^2 / 512^2 = 44% less attention memory
# on seq_c/d layers. Combined with seq_a/b unchanged, total attention
# memory drops ~35% vs v6's recipe.
#
# Expected AUC vs v6 (0.81287):
#   - seq_c/d 512 -> 384: estimated -0.002 to -0.004
#   - Everything else identical to v6 (BCE loss, no cyclical, batch 256)
#   - Predicted v15: 0.808 - 0.811
# Lands back in the proven big-recipe cluster, not the 0.797 floor.
#
# No new feature lever in v15. We've established that no lever moves
# AUC at the small recipe. Until we confirm v15 lands back in the
# 0.81 cluster, adding a lever is just adding noise. v16 stacks a
# single lever on top of v15's confirmed baseline.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --seq_encoder_type transformer \
    --seq_truncate auto \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:384,seq_d:384 \
    --batch_size 256 \
    --emb_skip_threshold 1000000 \
    --reinit_cardinality_threshold 0 \
    --reinit_sparse_after_epoch 1 \
    --no_cyclical_time \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
