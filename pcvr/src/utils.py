"""Shared utilities — seed pinning, losses, logger setup.

`set_seed` pins every non-platform-bound RNG and sets two env vars:
- ``PYTHONHASHSEED`` (NB: must ALSO be exported in run.sh BEFORE python starts;
  setting from inside Python is best-effort and primarily helps subprocesses).
- ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (required by torch.use_deterministic_algorithms
  with cuBLAS GEMMs).
"""
from __future__ import annotations

import os
import random
import logging
import time
from datetime import timedelta
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Pin Python/NumPy/Torch RNGs and configure deterministic env vars.

    Call this at the start of every entry point (train.py, infer.py).
    Note: PYTHONHASHSEED must also be exported BEFORE python starts via
    run.sh / prepare.sh — setting it here is a fallback for subprocesses.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass


def sigmoid_focal_loss(
    logits,
    targets,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
):
    """Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)."""
    import torch
    import torch.nn.functional as F
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_w = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_w * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class _ElapsedFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self._start = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = round(record.created - self._start)
        prefix = f"{time.strftime('%x %X')} - {timedelta(seconds=elapsed)}"
        msg = record.getMessage()
        msg = msg.replace("\n", "\n" + " " * (len(prefix) + 3))
        return f"{prefix} - [{record.levelname}] {msg}"


def create_logger(filepath: Optional[str] = None) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    fmt = _ElapsedFormatter()
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fh = logging.FileHandler(filepath, "w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class EarlyStopping:
    """Higher-is-better early stopping. ``checkpoint_path`` is overwritten
    on improvement; auxiliary metrics are stored in ``best_extra_metrics``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        patience: int = 5,
        delta: float = 0.0,
        label: str = "",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.delta = delta
        self.label = label + " " if label else ""
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_extra_metrics: Optional[dict] = None
        self.early_stop = False

    def __call__(self, score: float, model, extra_metrics: Optional[dict] = None) -> bool:
        """Return True iff a new best was just saved."""
        improved = self.best_score is None or score > self.best_score + self.delta
        if improved:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self._save(model)
            self.counter = 0
            return True
        self.counter += 1
        logging.info(f"{self.label}earlyStopping {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def _save(self, model) -> None:
        import torch
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
