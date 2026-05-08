"""Single-model rule audit — run before submission, included in CI."""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict


def audit_single_model(ckpt_dir: str) -> Dict[str, Any]:
    """Verify a checkpoint dir conforms to the single-model rule and return a
    parameter-fingerprint summary fit for code review.

    Raises ValueError on any violation. Returns ``{n_tensors, n_params, sha256, keys_sample}``.
    """
    import torch
    from src.checkpoint import _assert_single_state_dict, load_state_dict

    sd = load_state_dict(ckpt_dir)  # also asserts via _assert_single_state_dict
    n_tensors = len(sd)
    n_params = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode("utf-8"))
        v = sd[k].detach().to("cpu").contiguous()
        h.update(v.numpy().tobytes())
    return {
        "ckpt_dir": ckpt_dir,
        "n_tensors": n_tensors,
        "n_params": n_params,
        "sha256": h.hexdigest(),
        "keys_sample": list(sorted(sd.keys()))[:10],
    }


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir")
    args = p.parse_args()
    print(json.dumps(audit_single_model(args.ckpt_dir), indent=2))
