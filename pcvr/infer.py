"""Platform Step-3 entry — mandated filename. Reads checkpoint, runs forward
on EVAL_DATA_PATH, writes predictions.json. Owns predictions.json hygiene.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _resolve_paths() -> Dict[str, str]:
    return {
        "model_output_path": os.environ.get("MODEL_OUTPUT_PATH", "./model_output"),
        "eval_data_path": os.environ.get("EVAL_DATA_PATH", "./eval_data"),
        "eval_result_path": os.environ.get("EVAL_RESULT_PATH", "./eval_result"),
    }


def _check_user_id_format(observed: List[str], ckpt_dir: str) -> None:
    sidecar = os.path.join(ckpt_dir, "user_id_sample.json")
    if not os.path.isfile(sidecar):
        logging.warning("No user_id_sample.json in checkpoint; skipping format check")
        return
    with open(sidecar, "r", encoding="utf-8") as f:
        sample = json.load(f).get("sample", [])
    if not sample:
        return
    nonempty_observed = [o for o in observed[:50] if o]
    if not nonempty_observed:
        logging.warning("user_id format check: no non-empty observed user_ids; skipping")
        return
    train_is_digit = all(s.lstrip("-").isdigit() for s in sample if s)
    eval_is_digit = all(o.lstrip("-").isdigit() for o in nonempty_observed)
    if train_is_digit != eval_is_digit:
        # Soft-warn rather than raise: a wrong heuristic should not kill an
        # otherwise-valid submission. Code review will see this in the log.
        logging.warning(
            f"user_id format possibly mismatched: training samples digit-like="
            f"{train_is_digit}, eval observed digit-like={eval_is_digit}. "
            f"Train sample: {sample[:5]}; eval sample: {nonempty_observed[:5]}. "
            f"Verify the grader's join behavior against the leaderboard score."
        )


def main() -> None:
    import torch
    import torch.utils.data as tud
    from configs.baseline import Config
    from src.utils import set_seed
    from src.checkpoint import find_ckpt_dir, load_state_dict, resolve_ns_groups_path
    from src.data import PCVRParquetDataset, NUM_TIME_BUCKETS
    from src.model import PCVRHyFormer, ModelInput

    set_seed(0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = _resolve_paths()
    ckpt_dir = find_ckpt_dir(p["model_output_path"])
    logging.info(f"Using checkpoint: {ckpt_dir}")

    schema_path = os.path.join(ckpt_dir, "schema.json")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"schema.json missing in {ckpt_dir} - submission build is broken")
    with open(os.path.join(ckpt_dir, "train_config.json"), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})

    eval_ds = PCVRParquetDataset(
        parquet_path=p["eval_data_path"],
        schema_path=schema_path,
        batch_size=cfg.batch_size,
        seq_max_lens=cfg.seq_max_lens,
        shuffle=False, buffer_batches=0,
        clip_vocab=True, is_training=False,
    )

    user_specs = []
    for fid, offset, length in eval_ds.user_int_schema.entries:
        vs = max(eval_ds.user_int_vocab_sizes[offset:offset + length])
        user_specs.append((vs, offset, length))
    item_specs = []
    for fid, offset, length in eval_ds.item_int_schema.entries:
        vs = max(eval_ds.item_int_vocab_sizes[offset:offset + length])
        item_specs.append((vs, offset, length))

    ns_path = resolve_ns_groups_path(cfg.ns_groups_json, ckpt_dir)
    if ns_path:
        with open(ns_path, "r", encoding="utf-8") as f:
            ns_cfg = json.load(f)
        u_idx = {fid: i for i, (fid, _, _) in enumerate(eval_ds.user_int_schema.entries)}
        i_idx = {fid: i for i, (fid, _, _) in enumerate(eval_ds.item_int_schema.entries)}
        user_ns = [[u_idx[f] for f in fids] for fids in ns_cfg["user_ns_groups"].values()]
        item_ns = [[i_idx[f] for f in fids] for fids in ns_cfg["item_ns_groups"].values()]
    else:
        user_ns = [[i] for i in range(len(eval_ds.user_int_schema.entries))]
        item_ns = [[i] for i in range(len(eval_ds.item_int_schema.entries))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCVRHyFormer(
        user_int_feature_specs=user_specs, item_int_feature_specs=item_specs,
        user_dense_dim=eval_ds.user_dense_schema.total_dim,
        item_dense_dim=eval_ds.item_dense_schema.total_dim,
        seq_vocab_sizes=eval_ds.seq_domain_vocab_sizes,
        user_ns_groups=user_ns, item_ns_groups=item_ns,
        d_model=cfg.d_model, emb_dim=cfg.emb_dim,
        num_queries=cfg.num_queries, num_hyformer_blocks=cfg.num_hyformer_blocks,
        num_heads=cfg.num_heads, seq_encoder_type=cfg.seq_encoder_type,
        hidden_mult=cfg.hidden_mult, dropout_rate=cfg.dropout_rate,
        seq_top_k=cfg.seq_top_k, seq_causal=cfg.seq_causal,
        action_num=cfg.action_num,
        num_time_buckets=NUM_TIME_BUCKETS if cfg.use_time_buckets else 0,
        rank_mixer_mode=cfg.rank_mixer_mode, use_rope=cfg.use_rope, rope_base=cfg.rope_base,
        emb_skip_threshold=cfg.emb_skip_threshold, seq_id_threshold=cfg.seq_id_threshold,
        ns_tokenizer_type=cfg.ns_tokenizer_type,
        user_ns_tokens=cfg.user_ns_tokens, item_ns_tokens=cfg.item_ns_tokens,
        use_continuous_time=cfg.use_continuous_time,
    ).to(device)
    model.load_state_dict(load_state_dict(ckpt_dir), strict=True)
    model.eval()

    loader = tud.DataLoader(eval_ds, batch_size=None, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    score_sum: Dict[str, float] = {}
    score_cnt: Dict[str, int] = {}
    observed_uids: List[str] = []

    @torch.no_grad()
    def _run() -> None:
        for batch in loader:
            uids = [str(u) for u in batch["user_id"]]
            if len(observed_uids) < 50:
                observed_uids.extend(uids[:max(0, 50 - len(observed_uids))])
            seq_domains = batch["_seq_domains"]
            seq_data = {d: batch[d].to(device, non_blocking=True) for d in seq_domains}
            seq_lens = {d: batch[f"{d}_len"].to(device, non_blocking=True) for d in seq_domains}
            seq_tb = {}
            seq_ltd = {}
            for d in seq_domains:
                key = f"{d}_time_bucket"
                if key in batch:
                    seq_tb[d] = batch[key].to(device, non_blocking=True)
                else:
                    B, _, L = batch[d].shape
                    seq_tb[d] = torch.zeros(B, L, dtype=torch.long, device=device)
                # C3 test contract assertion #5: _run() must route the float
                # log_time_delta key to the model. If use_continuous_time=True
                # and this key is missing, the model raises RuntimeError —
                # no silent fallback to zeros (the v4 regression surface).
                key_ltd = f"{d}_log_time_delta"
                if key_ltd in batch:
                    seq_ltd[d] = batch[key_ltd].to(device, non_blocking=True)
            inp = ModelInput(
                user_int_feats=batch["user_int_feats"].to(device, non_blocking=True),
                item_int_feats=batch["item_int_feats"].to(device, non_blocking=True),
                user_dense_feats=batch["user_dense_feats"].to(device, non_blocking=True),
                item_dense_feats=batch["item_dense_feats"].to(device, non_blocking=True),
                seq_data=seq_data, seq_lens=seq_lens, seq_time_buckets=seq_tb,
                seq_log_time_delta=seq_ltd if seq_ltd else None,
            )
            logits, _ = model.predict(inp)
            probs = torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy()
            for u, pv in zip(uids, probs):
                score_sum[u] = score_sum.get(u, 0.0) + float(pv)
                score_cnt[u] = score_cnt.get(u, 0) + 1
    _run()

    if not score_sum:
        raise RuntimeError(f"No predictions produced from {p['eval_data_path']}")

    _check_user_id_format(observed_uids, ckpt_dir)

    eps = 1e-7
    preds: Dict[str, float] = {}
    n_multi = 0
    for u, total in score_sum.items():
        c = score_cnt[u]
        if c > 1:
            n_multi += 1
        s = total / c
        if not np.isfinite(s):
            raise ValueError(f"Non-finite score {s} for user_id={u}")
        preds[u] = float(min(1 - eps, max(eps, s)))
    if n_multi:
        logging.warning(f"{n_multi}/{len(preds)} users had >1 row; aggregated by mean")

    os.makedirs(p["eval_result_path"], exist_ok=True)
    out = os.path.join(p["eval_result_path"], "predictions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f, allow_nan=False)
    logging.info(f"Wrote {len(preds)} predictions -> {out}")


if __name__ == "__main__":
    main()
