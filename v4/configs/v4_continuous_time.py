"""v4: continuous log-delta time encoding for sequence events.

Inherits the *first_submission* config (which produced the 0.81144 leaderboard
score) and adds exactly one change: ``use_continuous_time = True``.

Falls back to the pcvr 0.81144 base on purpose — v3's (transformer + RoPE +
focal) bundle regressed to 0.806681, so we don't carry those knobs forward.
v4 isolates the time-feature change so its uplift is cleanly attributable.

What changes mechanically:
  * ``src/data.py`` emits ``{domain}_time_log_dt`` (float32 [B, L]) alongside
    the existing ``{domain}_time_bucket``. log_dt = log(1 + (label_ts - event_ts)).
  * ``src/model.py`` PCVRHyFormer constructs a small ``continuous_time_proj``
    (Linear(1, d_model) -> SiLU -> LayerNorm) and adds its output to each
    sequence token, in addition to the existing discrete bucket embedding.
  * Everything else (encoder, loss, LR schedule, bf16, valid_ratio, etc.)
    stays identical to first_submission.

Why log(1 + dt) and not raw dt:
  Time-deltas span ~12 orders of magnitude in this dataset (seconds to a
  year). A linear projection of raw seconds would be dominated by old events;
  log(1+dt) compresses the dynamic range so 1s vs 5s and 1h vs 5h get
  comparable gradient signal.

Why ADDITIVE rather than replacing the bucket embedding:
  The bucket embedding has been trained alongside everything else to give
  the 0.81144 base; ablating it would re-train more parameters than needed.
  The continuous head is purely additive — disabling ``use_continuous_time``
  at inference time would degrade to the exact pcvr base behaviour.
"""
from dataclasses import dataclass

from configs.first_submission import Config as _FirstSubmission


@dataclass
class Config(_FirstSubmission):
    use_continuous_time: bool = True
