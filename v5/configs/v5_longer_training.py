"""v5: longer training only, on the pcvr 0.81144 base.

Falls back to the pcvr 0.81144 base (NOT v3's transformer+RoPE+focal, NOT
v4's continuous-time projection). Adds *zero* new trainable parameters
and changes *zero* source code — only three config overrides:

  * num_epochs   3   -> 6      (give the model double the wall-clock budget)
  * patience     4   -> 7      (don't early-stop on a single noisy val dip)
  * valid_ratio  0.05 -> 0.10  (2x larger val set => less noisy early-stop)

Rationale: every single one of my prior v2/v3/v4 submissions regressed
relative to the pcvr base. The common factor was that each one *added*
new trainable parameters (transformer attention weights in v3, the
continuous_time_proj head in v4, etc.) and gave them only 3 epochs to
converge. The 0.81144 base is tightly tuned at the existing training
budget; any new params we add risk perturbing trained signal without
having time to learn good values themselves.

v5 tests the opposite hypothesis: "is the model itself just under-trained?"
If yes, longer training alone should bump us above 0.81144 with no
architectural risk. If no, we'll learn that future improvements *must*
come from feature/architectural changes — and that those changes will
also need a longer training budget to converge.

Either outcome is informative. The cost is small (one config file, no
source edits, near-zero regression risk).
"""
from dataclasses import dataclass

from configs.first_submission import Config as _FirstSubmission


@dataclass
class Config(_FirstSubmission):
    num_epochs: int = 6
    patience: int = 7
    valid_ratio: float = 0.10
