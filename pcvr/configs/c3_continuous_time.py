"""C3 (per roadmap-v3): continuous log(1+dt) time encoding.

Inherits first_submission (the config that produced the 0.81144 v0 anchor)
and flips exactly ONE lever: ``use_continuous_time = True``.

Per the roadmap-v3 C3 card, this REPLACES the discrete 65-bucket time
embedding with a continuous ``log(1 + (label_ts - event_ts))`` linear
projection — it is NOT additive. The bucket path is preserved behind the
flag for back-compat with v0 checkpoints; with this config the bucket
path is unused at training time.

Single-lever discipline (roadmap discipline checklist):
- Only field changed: ``use_continuous_time``.
- ``num_epochs``, ``patience``, ``valid_ratio``, ``seq_encoder_type``,
  ``loss_type``, etc. all inherit from first_submission unchanged.
- Test contract: ``pcvr/tests/test_c3_continuous_time.py`` ships with
  the 5 mandatory assertions from the roadmap card.

Rollback: set ``use_continuous_time = False``. State-dict keys differ
(``time_embedding.weight`` vs ``time_encoder.proj.weight``) so a c3
checkpoint cannot be loaded into a baseline model and vice versa; the
flag determines which path the model constructs at __init__ time.
"""
from dataclasses import dataclass

from configs.first_submission import Config as _FirstSubmission


@dataclass
class Config(_FirstSubmission):
    use_continuous_time: bool = True
