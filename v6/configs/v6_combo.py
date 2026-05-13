"""v6: longer training + continuous time (bug-fixed) + deeper user_dense MLP.

Bundles three complementary improvements on the pcvr 0.81144 base. The
explicit goal is to give the model both new capacity AND time to use it,
since every prior single change (v2..v5) regressed.

What changes vs first_submission (the pcvr 0.81144 config):

  * num_epochs            3   -> 5   (one extra epoch over base; v5
                                       went to 6 and regressed, so we
                                       step more modestly here)
  * patience              4   -> 6   (slack to ride out noisy val dips)
  * valid_ratio                 0.05 (UNCHANGED from base; v5 raised
                                       this to 0.10 and regressed,
                                       suggesting the 5% training data
                                       loss hurt more than the cleaner
                                       early-stop signal helped)
  * use_continuous_time   False -> True

Code-side changes that ship in v6/src/ (not in pcvr/src/):

  * data.py emits {domain}_time_log_dt per sequence event
    (log(1 + (label_ts - event_ts)), 0 at padding).
  * model.py's PCVRHyFormer accepts use_continuous_time, builds a
    bias-free Linear(1, d_model) initialized with std=0.01. The
    bias-free init means padding (log_dt=0) -> exactly zero output,
    fixing v4's LayerNorm-bias leak. Small std keeps the contribution
    from dominating the trained bucket signal early in training.
  * model.py's user_dense_proj is upgraded from a single Linear to a
    2-layer MLP (Linear -> SiLU -> Dropout -> Linear -> LN). Rationale:
    the ~755-dim user dense embeddings were being squeezed through one
    Linear to one 64-dim token, losing information. This is the
    "user侧" (user-side) improvement.
  * trainer.py and infer.py populate seq_time_log_dt in ModelInput;
    train.py and infer.py pass use_continuous_time when building
    PCVRHyFormer.

Why bundle three changes when single ones regressed:
  Each prior single change failed in part because new trainable params
  didn't have time to converge in 3 epochs. v6 pairs the new params
  (continuous_time_proj + deeper user_dense MLP) with one extra epoch
  + more patience so they get a chance to learn. Attribution will be
  imperfect, but the alternative (single isolated changes) has lost
  us 4 submissions in a row.
"""
from dataclasses import dataclass

from configs.first_submission import Config as _FirstSubmission


@dataclass
class Config(_FirstSubmission):
    num_epochs: int = 5
    patience: int = 6
    use_continuous_time: bool = True
