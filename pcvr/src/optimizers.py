"""Optimizer factory + LR scheduler.

Whitelist enforces the single-model rule: no SWA, SAM, Lookahead, or other
weight-averaging wrappers. Anything that maintains a moving-average shadow
of weights is banned.
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple


_ALLOWED_DENSE = {"adamw", "sgd"}
_ALLOWED_SPARSE = {"adagrad"}


def cosine_warmup_lambda(
    warmup_steps: int, total_steps: int, min_lr_factor: float = 0.1,
) -> Callable[[int], float]:
    """Returns a multiplier function compatible with torch's LambdaLR.

    Linear warmup over ``warmup_steps``, then cosine decay from 1.0 to
    ``min_lr_factor`` over the remainder.
    """
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)
    min_lr_factor = float(min_lr_factor)

    def _f(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + math.cos(math.pi * progress))

    return _f


def build_optimizers(
    dense_params: List,
    sparse_params: List,
    dense_optimizer: str,
    sparse_optimizer: str,
    dense_lr: float,
    sparse_lr: float,
    weight_decay: float = 0.0,
    sparse_weight_decay: float = 0.0,
):
    """Build (dense_opt, sparse_opt) using the whitelisted optimizer families.

    Raises ValueError for any unknown optimizer name. This is the gatekeeper
    for the no-ensembles single-model rule on the optimizer side.
    """
    if dense_optimizer not in _ALLOWED_DENSE:
        raise ValueError(
            f"dense_optimizer must be in {sorted(_ALLOWED_DENSE)}, "
            f"got {dense_optimizer!r}")
    if sparse_optimizer not in _ALLOWED_SPARSE:
        raise ValueError(
            f"sparse_optimizer must be in {sorted(_ALLOWED_SPARSE)}, "
            f"got {sparse_optimizer!r}")
    import torch
    if dense_optimizer == "adamw":
        dense_opt = torch.optim.AdamW(dense_params, lr=dense_lr,
                                      betas=(0.9, 0.98), weight_decay=weight_decay)
    else:  # sgd
        dense_opt = torch.optim.SGD(dense_params, lr=dense_lr,
                                    momentum=0.9, weight_decay=weight_decay)
    sparse_opt = torch.optim.Adagrad(
        sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
    return dense_opt, sparse_opt


def build_scheduler(optimizer, warmup_steps: int, total_steps: int,
                    min_lr_factor: float = 0.1):
    """Wrap optimizer in LambdaLR with cosine+warmup schedule."""
    import torch
    f = cosine_warmup_lambda(warmup_steps, total_steps, min_lr_factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
