"""Platform Step-1 entry point — thin wrapper that:
1. Resolves env vars onto a Config.
2. Builds dataset (with validators).
3. Builds the model.
4. Runs Trainer.

Env vars take precedence over Config defaults.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Make sibling dirs importable regardless of how the platform invokes us.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from configs.baseline import Config
from src.utils import set_seed, create_logger
from src.data import (
    PCVRParquetDataset, get_pcvr_data, NUM_TIME_BUCKETS,
    assert_time_split_monotonic, assert_label_rate_sane,
)
from src.model import PCVRHyFormer
from src.trainer import Trainer


def _build_feature_specs(schema, vocab_sizes):
    out = []
    for fid, offset, length in schema.entries:
        vs = max(vocab_sizes[offset:offset + length])
        out.append((vs, offset, length))
    return out


def _resolve_paths(cfg: Config) -> Config:
    cfg.data_dir = os.environ.get("TRAIN_DATA_PATH", cfg.data_dir)
    cfg.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", cfg.ckpt_dir)
    cfg.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH", cfg.tf_events_dir)
    cfg.log_dir = cfg.log_dir or cfg.ckpt_dir or "."
    if not cfg.schema_path:
        cfg.schema_path = os.path.join(cfg.data_dir, "schema.json")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline", help="config module name in configs/")
    args, _unknown = p.parse_known_args()

    mod = __import__(f"configs.{args.config}", fromlist=["Config"])
    cfg: Config = mod.Config()  # type: ignore
    cfg = _resolve_paths(cfg)

    Path(cfg.ckpt_dir or ".").mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir or ".").mkdir(parents=True, exist_ok=True)
    if cfg.tf_events_dir:
        Path(cfg.tf_events_dir).mkdir(parents=True, exist_ok=True)
    create_logger(os.path.join(cfg.log_dir, "train.log"))
    set_seed(cfg.seed)
    logging.info(f"Config: {json.dumps(asdict(cfg), indent=2, default=str)}")

    # ---- Validators (fail loud before training starts) ----
    assert_time_split_monotonic(cfg.data_dir, valid_ratio=cfg.valid_ratio)
    assert_label_rate_sane(cfg.data_dir, sample_size=100_000)

    # ---- Data ----
    train_loader, valid_loader, ds = get_pcvr_data(
        data_dir=cfg.data_dir, schema_path=cfg.schema_path,
        batch_size=cfg.batch_size, valid_ratio=cfg.valid_ratio,
        train_ratio=cfg.train_ratio, num_workers=cfg.num_workers,
        buffer_batches=cfg.buffer_batches, seed=cfg.seed,
        seq_max_lens=cfg.seq_max_lens,
    )

    # Snapshot 100 user_ids for inference-time format check.
    sample_user_ids = []
    for b in train_loader:
        sample_user_ids.extend([str(u) for u in b["user_id"]])
        if len(sample_user_ids) >= 100:
            break
    sample_user_ids = sample_user_ids[:100]

    # ---- NS groups ----
    if cfg.ns_groups_json and os.path.exists(cfg.ns_groups_json):
        with open(cfg.ns_groups_json, "r", encoding="utf-8") as f:
            ns_cfg = json.load(f)
        u_idx = {fid: i for i, (fid, _, _) in enumerate(ds.user_int_schema.entries)}
        i_idx = {fid: i for i, (fid, _, _) in enumerate(ds.item_int_schema.entries)}
        user_ns = [[u_idx[f] for f in fids] for fids in ns_cfg["user_ns_groups"].values()]
        item_ns = [[i_idx[f] for f in fids] for fids in ns_cfg["item_ns_groups"].values()]
        ns_path = cfg.ns_groups_json
    else:
        user_ns = [[i] for i in range(len(ds.user_int_schema.entries))]
        item_ns = [[i] for i in range(len(ds.item_int_schema.entries))]
        ns_path = None

    user_specs = _build_feature_specs(ds.user_int_schema, ds.user_int_vocab_sizes)
    item_specs = _build_feature_specs(ds.item_int_schema, ds.item_int_vocab_sizes)

    model = PCVRHyFormer(
        user_int_feature_specs=user_specs,
        item_int_feature_specs=item_specs,
        user_dense_dim=ds.user_dense_schema.total_dim,
        item_dense_dim=ds.item_dense_schema.total_dim,
        seq_vocab_sizes=ds.seq_domain_vocab_sizes,
        user_ns_groups=user_ns,
        item_ns_groups=item_ns,
        d_model=cfg.d_model, emb_dim=cfg.emb_dim,
        num_queries=cfg.num_queries, num_hyformer_blocks=cfg.num_hyformer_blocks,
        num_heads=cfg.num_heads, seq_encoder_type=cfg.seq_encoder_type,
        hidden_mult=cfg.hidden_mult, dropout_rate=cfg.dropout_rate,
        seq_top_k=cfg.seq_top_k, seq_causal=cfg.seq_causal,
        action_num=cfg.action_num,
        num_time_buckets=NUM_TIME_BUCKETS if cfg.use_time_buckets else 0,
        rank_mixer_mode=cfg.rank_mixer_mode,
        use_rope=cfg.use_rope, rope_base=cfg.rope_base,
        emb_skip_threshold=cfg.emb_skip_threshold,
        seq_id_threshold=cfg.seq_id_threshold,
        ns_tokenizer_type=cfg.ns_tokenizer_type,
        user_ns_tokens=cfg.user_ns_tokens,
        item_ns_tokens=cfg.item_ns_tokens,
    )

    writer = None
    if cfg.tf_events_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(cfg.tf_events_dir)
        except Exception as e:
            logging.warning(f"TensorBoard writer disabled: {e}")

    trainer = Trainer(
        cfg=cfg, model=model,
        train_loader=train_loader, valid_loader=valid_loader,
        ckpt_out_dir=cfg.ckpt_dir, schema_path=cfg.schema_path,
        ns_groups_path=ns_path, user_id_sample=sample_user_ids,
        experiments_csv=os.path.join(_HERE, "experiments.csv"),
        writer=writer,
    )
    best_dir = trainer.train()
    logging.info(f"Training finished; best={best_dir}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
