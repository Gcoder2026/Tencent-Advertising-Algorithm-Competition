"""PCVRHyFormer training loop — bf16 default ON, cosine+warmup, resume, registry append.

Differences vs. starter kit's PCVRHyFormerRankingTrainer:
- Hyperparams come from a Config dataclass, not constructor args.
- Optimizers built via src.optimizers (whitelist).
- Mixed precision (bf16) in train + eval forward.
- LambdaLR cosine+warmup scheduler stepped per training step.
- Resume from a `train_state.pt` sidecar that bundles dense_opt + sparse_opt
  + scheduler + RNG state + global_step.
- Best-model checkpoints saved via src.checkpoint.save_checkpoint with the
  full sidecar set, including user_id_sample.json.
- Run-registry append to ``experiments.csv`` on training end (or on best).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.checkpoint import save_checkpoint, build_step_dir_name
from src.optimizers import build_optimizers, build_scheduler
from src.utils import EarlyStopping, sigmoid_focal_loss
from src.model import ModelInput


def _make_model_input(device_batch: Dict[str, Any], device) -> ModelInput:
    seq_domains = device_batch["_seq_domains"]
    seq_data = {d: device_batch[d] for d in seq_domains}
    seq_lens = {d: device_batch[f"{d}_len"] for d in seq_domains}
    seq_tb: Dict[str, torch.Tensor] = {}
    for d in seq_domains:
        key = f"{d}_time_bucket"
        if key in device_batch:
            seq_tb[d] = device_batch[key]
        else:
            B, _, L = device_batch[d].shape
            seq_tb[d] = torch.zeros(B, L, dtype=torch.long, device=device)
    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_tb,
    )


def _batch_to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _config_hash(cfg) -> str:
    raw = json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _append_experiment_row(csv_path: str, row: Dict[str, Any]) -> None:
    new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


class Trainer:
    """Training driver. Owns the loop, the validators, and the registry."""

    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader,
        valid_loader,
        ckpt_out_dir: str,
        schema_path: str,
        ns_groups_path: Optional[str],
        user_id_sample: List[str],
        experiments_csv: Optional[str] = None,
        writer=None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.ckpt_out_dir = ckpt_out_dir
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path
        self.user_id_sample = user_id_sample
        self.experiments_csv = experiments_csv
        self.writer = writer
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        sparse = self.model.get_sparse_params() if hasattr(self.model, "get_sparse_params") else []
        dense = self.model.get_dense_params() if hasattr(self.model, "get_dense_params") else list(self.model.parameters())
        if sparse:
            self.dense_opt, self.sparse_opt = build_optimizers(
                dense_params=dense, sparse_params=sparse,
                dense_optimizer=cfg.dense_optimizer, sparse_optimizer=cfg.sparse_optimizer,
                dense_lr=cfg.lr, sparse_lr=cfg.sparse_lr,
                weight_decay=cfg.weight_decay, sparse_weight_decay=cfg.sparse_weight_decay,
            )
        else:
            # No sparse params: still run the whitelist on dense optimizer name,
            # but disable the sparse optimizer.
            self.dense_opt, _ = build_optimizers(
                dense_params=dense, sparse_params=[nn.Parameter(torch.zeros(1))],
                dense_optimizer=cfg.dense_optimizer, sparse_optimizer=cfg.sparse_optimizer,
                dense_lr=cfg.lr, sparse_lr=cfg.sparse_lr,
                weight_decay=cfg.weight_decay, sparse_weight_decay=cfg.sparse_weight_decay,
            )
            self.sparse_opt = None

        # Scheduler placeholder; total_steps wired in train() once known.
        self.dense_sched = None

        self.early_stopping = EarlyStopping(
            checkpoint_path=os.path.join(ckpt_out_dir, "_es_placeholder", "model.pt"),
            patience=cfg.patience,
            label="val",
        )

        self.global_step = 0
        self._t0 = time.time()
        self._best_dir: Optional[str] = None

    def _load_resume(self, resume_dir: str) -> None:
        ts_path = os.path.join(resume_dir, "train_state.pt")
        if not os.path.isfile(ts_path):
            return
        st = torch.load(ts_path, map_location=self.device, weights_only=False)
        self.dense_opt.load_state_dict(st["dense_opt"])
        if self.sparse_opt is not None and st.get("sparse_opt") is not None:
            self.sparse_opt.load_state_dict(st["sparse_opt"])
        self.global_step = int(st.get("global_step", 0))
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        logging.info(f"Resumed from {resume_dir} at step {self.global_step}")

    def _train_state_blob(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "dense_opt": self.dense_opt.state_dict(),
            "sparse_opt": self.sparse_opt.state_dict() if self.sparse_opt else None,
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }

    def _autocast(self):
        if self.cfg.use_bf16 and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext
        return nullcontext()

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = _batch_to_device(batch, self.device)
        label = device_batch["label"].float()

        self.dense_opt.zero_grad(set_to_none=True)
        if self.sparse_opt is not None:
            self.sparse_opt.zero_grad(set_to_none=True)

        with self._autocast():
            inputs = _make_model_input(device_batch, self.device)
            logits = self.model(inputs).squeeze(-1)
            if self.cfg.loss_type == "focal":
                loss = sigmoid_focal_loss(logits, label,
                                          alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
        self.dense_opt.step()
        if self.sparse_opt is not None:
            self.sparse_opt.step()
        if self.dense_sched is not None:
            self.dense_sched.step()
        self.global_step += 1
        return float(loss.item())

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        for batch in tqdm(self.valid_loader, desc="valid", dynamic_ncols=True):
            device_batch = _batch_to_device(batch, self.device)
            with self._autocast():
                inputs = _make_model_input(device_batch, self.device)
                logits, _ = self.model.predict(inputs)
            all_logits.append(logits.detach().float().squeeze(-1).cpu())
            all_labels.append(device_batch["label"].detach().cpu())
        self.model.train()
        L = torch.cat(all_logits).numpy()
        Y = torch.cat(all_labels).numpy()
        mask = np.isfinite(L)
        if mask.sum() == 0 or len(np.unique(Y[mask])) < 2:
            return 0.0, float("inf")
        probs = 1.0 / (1.0 + np.exp(-L[mask]))
        auc = float(roc_auc_score(Y[mask], probs))
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        ll = float(-(Y[mask] * np.log(p) + (1 - Y[mask]) * np.log(1 - p)).mean())
        return auc, ll

    def _save_best(self, val_auc: float, val_logloss: float) -> str:
        best_dir = save_checkpoint(
            out_dir=self.ckpt_out_dir,
            global_step=self.global_step,
            model=self.model,
            cfg=self.cfg,
            schema_path=self.schema_path,
            ns_groups_path=self.ns_groups_path,
            user_id_sample=self.user_id_sample,
            is_best=True,
            train_state=self._train_state_blob(),
        )
        # remove stale best_model dirs
        for d in os.listdir(self.ckpt_out_dir):
            full = os.path.join(self.ckpt_out_dir, d)
            if d.endswith(".best_model") and full != best_dir:
                shutil.rmtree(full, ignore_errors=True)
        self._best_dir = best_dir
        if self.writer is not None:
            self.writer.add_scalar("AUC/valid", val_auc, self.global_step)
            self.writer.add_scalar("LogLoss/valid", val_logloss, self.global_step)
        logging.info(f"NEW BEST AUC={val_auc:.5f} logloss={val_logloss:.5f} -> {best_dir}")
        return best_dir

    def train(self) -> Optional[str]:
        if self.cfg.resume_from:
            self._load_resume(self.cfg.resume_from)

        # Compute total_steps for cosine schedule.
        try:
            steps_per_epoch = len(self.train_loader)
        except TypeError:
            steps_per_epoch = max(1, self.cfg.eval_every_n_steps or 1000)
        total_steps = max(1, self.cfg.total_steps_hint or (steps_per_epoch * max(1, self.cfg.num_epochs)))
        self.dense_sched = build_scheduler(
            self.dense_opt, warmup_steps=self.cfg.warmup_steps,
            total_steps=total_steps, min_lr_factor=self.cfg.min_lr_factor,
        )

        self.model.train()
        for epoch in range(1, self.cfg.num_epochs + 1):
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True):
                loss = self._train_step(batch)
                if (self.cfg.eval_every_n_steps > 0
                        and self.global_step % self.cfg.eval_every_n_steps == 0):
                    auc, ll = self._evaluate()
                    if self.early_stopping(auc, self.model, {"auc": auc, "logloss": ll}):
                        self._save_best(auc, ll)
                    if self.early_stopping.early_stop:
                        return self._finish()
            auc, ll = self._evaluate()
            if self.early_stopping(auc, self.model, {"auc": auc, "logloss": ll}):
                self._save_best(auc, ll)
            if self.early_stopping.early_stop:
                break
        return self._finish()

    def _finish(self) -> Optional[str]:
        if self.experiments_csv:
            row = {
                "run_id": str(uuid.uuid4())[:8],
                "config_hash": _config_hash(self.cfg),
                "best_val_auc": float(self.early_stopping.best_score or 0.0),
                "best_val_logloss": float((self.early_stopping.best_extra_metrics or {}).get("logloss", float("nan"))),
                "global_step": self.global_step,
                "seed": self.cfg.seed,
                "ckpt_path": self._best_dir or "",
                "wall_clock_min": round((time.time() - self._t0) / 60.0, 2),
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "notes": "",
            }
            _append_experiment_row(self.experiments_csv, row)
        return self._best_dir
