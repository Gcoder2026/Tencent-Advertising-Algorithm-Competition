#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data_sample_1000"

if [[ ! -f "${DATA_DIR}/demo_1000.parquet" || ! -f "${DATA_DIR}/schema.json" ]]; then
    python3 "${SCRIPT_DIR}/tools/prepare_hf_sample.py" \
        --source /private/tmp/taac_demo_1000.parquet \
        --out_dir "${DATA_DIR}" \
        --row_group_size 200 \
        --download_if_missing
fi

export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${SCRIPT_DIR}/outputs/sample_ckpt}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${SCRIPT_DIR}/outputs/sample_logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${SCRIPT_DIR}/outputs/sample_tf_events}"

bash "${SCRIPT_DIR}/run.sh" \
    --data_dir "${DATA_DIR}" \
    --schema_path "${DATA_DIR}/schema.json" \
    --num_epochs 1 \
    --patience 1 \
    --batch_size 64 \
    --num_workers 0 \
    --buffer_batches 1 \
    --valid_ratio 0.2 \
    --seq_max_lens seq_a:32,seq_b:32,seq_c:64,seq_d:64 \
    --d_model 32 \
    --emb_dim 16 \
    --num_hyformer_blocks 1 \
    --num_heads 4 \
    --seq_encoder_type swiglu \
    --reinit_sparse_after_epoch 999 \
    "$@"
