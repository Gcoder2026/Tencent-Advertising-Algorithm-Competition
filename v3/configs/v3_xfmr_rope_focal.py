"""v3: transformer encoder + RoPE + focal loss, on the pcvr 0.81144 base.

Inherits everything from first_submission (which is what produced the 0.81144
leaderboard score) and flips three knobs that are already implemented in the
codebase but were left disabled there:

  1. seq_encoder_type "swiglu" -> "transformer".
     v1's SwiGLU encoder is FFN-only (no attention). The transformer variant
     adds standard Pre-LN self-attention per behaviour-domain sequence so the
     model can actually learn ordering inside seq_a/b/c/d. v2 already proved
     this doesn't blow the inference budget (260s vs v1's 381s on platform).

  2. use_rope False -> True.
     RoPE only does anything when there is an attention layer to apply it in,
     which is why we flip it together with seq_encoder_type. With both off
     (pcvr base) RoPE is a no-op; with transformer on but RoPE off, the
     attention sees no positional info — strictly worse than RoPE on.

  3. loss_type "bce" -> "focal" with alpha=0.25, gamma=2.0.
     The default focal_alpha=0.1 in baseline.py is too aggressive a
     positive-class downweight for our ~12% conversion rate (it would
     suppress positives further). 0.25 is a more conservative
     downweight on negatives that gradient-balances mild imbalance.
     gamma=2.0 is the standard focusing value.

Bundling three changes into one submission breaks the usual "one variable per
submission" rule. The trade-off is intentional given the leaderboard pressure
(team best 0.81144 vs leaderboard top-12 cutoff ~0.83) — we can't afford to
spend three submission slots flipping single config flags. If v3 lands a
clean uplift, future submissions can ablate which of the three did the work.
"""
from dataclasses import dataclass

from configs.first_submission import Config as _FirstSubmission


@dataclass
class Config(_FirstSubmission):
    seq_encoder_type: str = "transformer"
    use_rope: bool = True
    loss_type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
