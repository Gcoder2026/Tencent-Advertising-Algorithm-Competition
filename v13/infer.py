"""Platform inference entry point for TAAC2026 PCVRHyFormer.

The Angel evaluation platform calls ``main()`` with no arguments. The expected
paths are provided through environment variables:

- ``MODEL_OUTPUT_PATH``: published checkpoint directory
- ``EVAL_DATA_PATH``: hidden evaluation parquet directory
- ``EVAL_RESULT_PATH``: output directory for predictions
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
from model import ModelInput, PCVRHyFormer


def _env_path(name: str, required: bool = True) -> Optional[Path]:
    value = os.environ.get(name)
    if value:
        return Path(value)
    if required:
        raise ValueError(f"{name} is not set")
    return None


def _find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def _find_parquet_path(root: Path) -> Path:
    if root.is_file() and root.suffix == ".parquet":
        return root
    direct = sorted(root.glob("*.parquet"))
    if direct:
        return root

    recursive = sorted(root.rglob("*.parquet"))
    if not recursive:
        raise FileNotFoundError(f"No parquet files found under {root}")
    parents = {p.parent for p in recursive}
    if len(parents) == 1:
        return next(iter(parents))
    raise ValueError(
        f"Found parquet files in multiple subdirectories under {root}; "
        "please upload an infer.py that selects the intended evaluation split.")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_seq_max_lens(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if not value:
        return {"seq_a": 128, "seq_b": 128, "seq_c": 256, "seq_d": 256}
    result: Dict[str, int] = {}
    for pair in str(value).split(","):
        if not pair.strip():
            continue
        key, raw = pair.split(":")
        result[key.strip()] = int(raw.strip())
    return result


def _feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vocab_size, offset, length))
    return specs


def _load_ns_groups(
    cfg: Dict[str, Any],
    ckpt_dir: Path,
    dataset: PCVRParquetDataset,
) -> Tuple[List[List[int]], List[List[int]]]:
    ns_path_raw = cfg.get("ns_groups_json")
    ns_path: Optional[Path] = None
    if ns_path_raw:
        candidate = Path(str(ns_path_raw))
        if candidate.is_absolute() and candidate.exists():
            ns_path = candidate
        else:
            candidate = ckpt_dir / candidate.name
            if candidate.exists():
                ns_path = candidate

    if ns_path is None:
        user_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]
        return user_groups, item_groups

    ns_cfg = _load_json(ns_path)
    user_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
    }
    item_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
    }
    user_groups = [
        [user_fid_to_idx[fid] for fid in fids]
        for fids in ns_cfg["user_ns_groups"].values()
    ]
    item_groups = [
        [item_fid_to_idx[fid] for fid in fids]
        for fids in ns_cfg["item_ns_groups"].values()
    ]
    return user_groups, item_groups


def _default_train_config() -> Dict[str, Any]:
    """Fallback matching the first submitted ``run.sh`` config."""
    return {
        "d_model": 64,
        "emb_dim": 64,
        "num_queries": 2,
        "num_hyformer_blocks": 2,
        "num_heads": 4,
        "seq_encoder_type": "swiglu",
        "hidden_mult": 4,
        "dropout_rate": 0.01,
        "seq_top_k": 50,
        "seq_causal": False,
        "action_num": 1,
        "use_time_buckets": True,
        "rank_mixer_mode": "full",
        "use_rope": False,
        "rope_base": 10000.0,
        "emb_skip_threshold": 1000000,
        "seq_id_threshold": 10000,
        "ns_tokenizer_type": "rankmixer",
        "user_ns_tokens": 5,
        "item_ns_tokens": 2,
        "seq_max_lens": "seq_a:128,seq_b:128,seq_c:256,seq_d:256",
        "seq_truncate": "auto",
        "batch_size": 256,
        "use_cyclical_time": True,
    }


def _build_model(
    cfg: Dict[str, Any],
    dataset: PCVRParquetDataset,
    ckpt_dir: Path,
) -> PCVRHyFormer:
    user_groups, item_groups = _load_ns_groups(cfg, ckpt_dir, dataset)
    model_args = {
        "user_int_feature_specs": _feature_specs(
            dataset.user_int_schema, dataset.user_int_vocab_sizes),
        "item_int_feature_specs": _feature_specs(
            dataset.item_int_schema, dataset.item_int_vocab_sizes),
        "user_dense_dim": dataset.user_dense_schema.total_dim,
        "item_dense_dim": dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_groups,
        "item_ns_groups": item_groups,
        "d_model": int(cfg.get("d_model", 64)),
        "emb_dim": int(cfg.get("emb_dim", 64)),
        "num_queries": int(cfg.get("num_queries", 2)),
        "num_hyformer_blocks": int(cfg.get("num_hyformer_blocks", 2)),
        "num_heads": int(cfg.get("num_heads", 4)),
        "seq_encoder_type": cfg.get("seq_encoder_type", "swiglu"),
        "hidden_mult": int(cfg.get("hidden_mult", 4)),
        "dropout_rate": float(cfg.get("dropout_rate", 0.01)),
        "seq_top_k": int(cfg.get("seq_top_k", 50)),
        "seq_causal": bool(cfg.get("seq_causal", False)),
        "action_num": int(cfg.get("action_num", 1)),
        "num_time_buckets": NUM_TIME_BUCKETS if cfg.get("use_time_buckets", True) else 0,
        "rank_mixer_mode": cfg.get("rank_mixer_mode", "full"),
        "use_rope": bool(cfg.get("use_rope", False)),
        "rope_base": float(cfg.get("rope_base", 10000.0)),
        "emb_skip_threshold": int(cfg.get("emb_skip_threshold", 1000000)),
        "seq_id_threshold": int(cfg.get("seq_id_threshold", 10000)),
        "ns_tokenizer_type": cfg.get("ns_tokenizer_type", "rankmixer"),
        "user_ns_tokens": int(cfg.get("user_ns_tokens", 5)),
        "item_ns_tokens": int(cfg.get("item_ns_tokens", 2)),
        "use_cyclical_time": bool(cfg.get("use_cyclical_time", True)),
    }
    return PCVRHyFormer(**model_args)


def _make_model_input(batch: Dict[str, Any], device: torch.device) -> ModelInput:
    seq_domains = batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = batch[domain].to(device, non_blocking=True)
        seq_lens[domain] = batch[f"{domain}_len"].to(device, non_blocking=True)
        seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(
            device, non_blocking=True)

    # v13.4: defensive cyclical-time field derivation, pure integer
    # arithmetic only (no datetime64). If the dataset emitted them
    # (preferred path), use directly. Otherwise derive from the
    # timestamp field — guarantees the cyclical features ALWAYS reach
    # the model even if the eval-side dataset version differs from
    # training. month_of_year is intentionally left None: the model's
    # forward path skips the application when month is missing, while
    # the embedding module is still constructed (state-dict stable).
    if "hour_of_day" in batch:
        hour_of_day = batch["hour_of_day"].to(device, non_blocking=True)
        day_of_week = batch["day_of_week"].to(device, non_blocking=True)
    elif "timestamp" in batch:
        ts = batch["timestamp"].to(device, non_blocking=True)
        hour_of_day = ((ts % 86400) // 3600).long()
        day_of_week = ((ts // 86400) % 7).long()
    else:
        hour_of_day = None
        day_of_week = None

    return ModelInput(
        user_int_feats=batch["user_int_feats"].to(device, non_blocking=True),
        item_int_feats=batch["item_int_feats"].to(device, non_blocking=True),
        user_dense_feats=batch["user_dense_feats"].to(device, non_blocking=True),
        item_dense_feats=batch["item_dense_feats"].to(device, non_blocking=True),
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        month_of_year=None,
    )


def _write_result(
    result_dir: Path,
    scores: List[float],
    user_ids: List[Any],
    item_ids: List[Any],
    elapsed: float,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "time": elapsed,
        "score": scores,
        "scores": scores,
        "prediction": scores,
        "predictions": scores,
        "user_id": user_ids,
    }
    if item_ids:
        result["item_id"] = item_ids

    with (result_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _iter_env_debug(names: Iterable[str]) -> str:
    pairs = [f"{name}={os.environ.get(name)}" for name in names]
    return ", ".join(pairs)


def main() -> None:
    start = time.time()
    infer_dir = _env_path("EVAL_INFER_PATH", required=False)
    model_dir = _env_path("MODEL_OUTPUT_PATH", required=True)
    data_root = _env_path("EVAL_DATA_PATH", required=True)
    result_dir = _env_path("EVAL_RESULT_PATH", required=True)

    print("Inference env:", _iter_env_debug([
        "EVAL_INFER_PATH", "MODEL_OUTPUT_PATH", "EVAL_DATA_PATH", "EVAL_RESULT_PATH"
    ]))
    if infer_dir:
        print(f"Inference code directory: {infer_dir}")

    model_path = _find_file(model_dir, "model.pt")
    schema_path = _find_file(model_dir, "schema.json")
    try:
        cfg = _load_json(_find_file(model_dir, "train_config.json"))
    except FileNotFoundError:
        cfg = _default_train_config()
        print("train_config.json not found; using first-submission defaults")

    seq_max_lens = _parse_seq_max_lens(cfg.get("seq_max_lens"))
    seq_truncate = cfg.get("seq_truncate", "auto")
    parquet_path = _find_parquet_path(data_root)
    batch_size = int(os.environ.get("INFER_BATCH_SIZE", cfg.get("infer_batch_size", 512)))

    print(f"Model path: {model_path}")
    print(f"Schema path: {schema_path}")
    print(f"Parquet path: {parquet_path}")
    print(f"batch_size={batch_size}, seq_max_lens={seq_max_lens}, seq_truncate={seq_truncate}")

    dataset = PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=False,
        seq_truncate=str(seq_truncate),
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, dataset, model_dir).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.eval()

    scores: List[float] = []
    user_ids: List[Any] = []
    item_ids: List[Any] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            model_input = _make_model_input(batch, device)
            logits, _ = model.predict(model_input)
            prob = torch.sigmoid(logits.squeeze(-1)).detach().cpu().float()
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
            scores.extend(prob.tolist())
            user_ids.extend(batch.get("user_id", []))
            item_ids.extend(batch.get("item_id", []))
            if batch_idx % 100 == 0:
                print(f"infer batches={batch_idx + 1}, rows={len(scores)}")

    elapsed = time.time() - start
    _write_result(result_dir, scores, user_ids, item_ids, elapsed)
    print(f"Wrote {len(scores)} predictions to {result_dir / 'result.json'}")
    print(f"Inference elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
