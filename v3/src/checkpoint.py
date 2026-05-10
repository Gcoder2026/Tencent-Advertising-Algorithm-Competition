"""Checkpoint save/load + sidecars + single-model invariants.

Each saved checkpoint dir is self-contained:
  global_stepN[.layer=L.head=H.hidden=D][.best_model]/
    ├── model.pt              # state_dict ONLY, single tensor dict
    ├── schema.json           # copy of the training schema
    ├── train_config.json     # asdict(Config) — full hyperparams
    ├── ns_groups.json        # iff ns groups were used
    ├── user_id_sample.json   # 100 sample user_id reprs from training data
    └── train_state.pt        # iff resume saving — opt + sched + RNG state

infer.py rebuilds the model from these files alone. No shared state with
the training run.

Single-model rule guards (called by ``save_checkpoint`` and ``load_state_dict``):
  - state_dict must be Mapping[str, Tensor].
  - No keys starting with ``ema_``, ``swa_``, ``shadow_``, ``averaged_``.
  - ``load_state_dict(path)`` refuses paths containing glob characters.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional


_FORBIDDEN_KEY_PREFIXES = ("ema_", "swa_", "shadow_", "averaged_", "polyak_")
_GLOB_CHARS = set("*?[]")


def build_step_dir_name(
    global_step: int,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    hidden: Optional[int] = None,
    is_best: bool = False,
) -> str:
    """Constructs a checkpoint directory name with the platform-mandated
    ``global_step`` prefix.
    """
    parts = [f"global_step{global_step}"]
    if layer is not None:
        parts.append(f"layer={layer}")
    if head is not None:
        parts.append(f"head={head}")
    if hidden is not None:
        parts.append(f"hidden={hidden}")
    name = ".".join(parts)
    if is_best:
        name += ".best_model"
    return name


def _assert_single_state_dict(sd: Dict[str, Any]) -> None:
    """Hard-fail if ``sd`` contains any key suggestive of an ensemble/EMA/SWA setup."""
    if not isinstance(sd, dict):
        raise ValueError(f"state_dict must be a dict, got {type(sd)}")
    for k in sd.keys():
        for pref in _FORBIDDEN_KEY_PREFIXES:
            if k.startswith(pref):
                raise ValueError(
                    f"state_dict contains forbidden key {k!r}; this would be "
                    f"flagged as ensemble/EMA/SWA in code review and disqualify "
                    f"the submission. Remove the shadow accumulator before save.")


def save_checkpoint(
    out_dir: str,
    global_step: int,
    model,
    cfg,
    schema_path: str,
    ns_groups_path: Optional[str],
    user_id_sample: List[str],
    is_best: bool = False,
    train_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model + sidecars under ``out_dir``. Returns the checkpoint dir.

    ``train_state`` (optimizer + scheduler + rng state) is written to
    ``train_state.pt`` IFF provided; the published-for-inference checkpoint
    must not include it (drop the file when exporting).
    """
    import torch
    name = build_step_dir_name(
        global_step,
        layer=getattr(cfg, "num_hyformer_blocks", None),
        head=getattr(cfg, "num_heads", None),
        hidden=getattr(cfg, "d_model", None),
        is_best=is_best,
    )
    ckpt_dir = os.path.join(out_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    sd = model.state_dict()
    _assert_single_state_dict(sd)
    torch.save(sd, os.path.join(ckpt_dir, "model.pt"))

    if os.path.exists(schema_path):
        shutil.copy2(schema_path, ckpt_dir)

    ns_basename = ""
    if ns_groups_path and os.path.exists(ns_groups_path):
        shutil.copy2(ns_groups_path, ckpt_dir)
        ns_basename = os.path.basename(ns_groups_path)

    cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    if ns_basename:
        cfg_dict["ns_groups_json"] = ns_basename
    with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    with open(os.path.join(ckpt_dir, "user_id_sample.json"), "w", encoding="utf-8") as f:
        json.dump({"sample": list(user_id_sample[:100])}, f)

    if train_state is not None:
        torch.save(train_state, os.path.join(ckpt_dir, "train_state.pt"))

    return ckpt_dir


def load_state_dict(ckpt_dir: str):
    """Load ``model.pt`` from ``ckpt_dir`` after asserting single-model invariants.

    Refuses any path containing glob metacharacters (`*`, `?`, `[`, `]`) — to
    prevent accidental "average all matching checkpoints" patterns in user code.
    """
    if any(c in ckpt_dir for c in _GLOB_CHARS):
        raise ValueError(
            f"load_state_dict accepts a single ckpt directory; refusing glob: {ckpt_dir!r}")
    import torch
    pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(pt):
        raise FileNotFoundError(f"No model.pt under {ckpt_dir}")
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    # Defensive: strip 'module.' prefix from DDP-wrapped state_dicts.
    if any(k.startswith("module.") for k in sd):
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    _assert_single_state_dict(sd)
    return sd


def find_ckpt_dir(model_output_path: str) -> str:
    """Resolve $MODEL_OUTPUT_PATH to a single checkpoint dir (flat or
    one-global_step-subdir). Refuses ambiguity beyond the *.best_model preference.
    """
    if os.path.isfile(os.path.join(model_output_path, "model.pt")):
        return model_output_path
    cands = sorted(glob.glob(os.path.join(model_output_path, "global_step*")))
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "model.pt"))]
    if not cands:
        raise FileNotFoundError(f"No checkpoint with model.pt under {model_output_path}")
    best = [c for c in cands if c.endswith(".best_model")]
    if best:
        return best[-1]
    return cands[-1]


def resolve_ns_groups_path(ns_field: Optional[str], ckpt_dir: str) -> Optional[str]:
    """Resolve train_config['ns_groups_json'] against ckpt_dir.
    See checkpoint sidecar conventions in module docstring.
    """
    if not ns_field:
        return None
    if os.path.isabs(ns_field) and os.path.exists(ns_field):
        return ns_field
    cand = os.path.join(ckpt_dir, os.path.basename(ns_field))
    return cand if os.path.exists(cand) else None
