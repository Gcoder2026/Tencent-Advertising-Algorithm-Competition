"""First-submission config — short budget so we land a leaderboard anchor today.

Inherits all defaults from `configs.baseline.Config` and overrides:
- ``num_epochs=3`` (matches the starter kit run.sh's first-real-submission budget).
- ``patience=2`` (early-stop tighter so we don't spin if the loss plateaus).
- ``eval_every_n_steps=2000`` (lets early stopping fire mid-epoch on long epochs).

After we have a leaderboard score, switch to the `baseline` config (or a stream-
specific one) for full-budget runs.
"""
from dataclasses import dataclass

from configs.baseline import Config as _Baseline


@dataclass
class Config(_Baseline):
    num_epochs: int = 3
    patience: int = 4
    eval_every_n_steps: int = 2000
