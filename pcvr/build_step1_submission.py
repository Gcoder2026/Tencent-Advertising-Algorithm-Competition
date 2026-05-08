"""Build the Step-1 (training) submission zip for AngelML.

Step-1 zip contains everything the platform needs to run training:
  run.sh, prepare.sh, train.py, src/*, configs/*, requirements.txt.
It does NOT contain: infer.py, tests/, docs, README, __pycache__, .git, .pyc.

Usage:
    python build_step1_submission.py [output.zip]

Default output: ``step1_train.zip`` next to this script.
"""
from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent

# Files we ship into the Step-1 zip.
ROOT_FILES = [
    "run.sh",
    "prepare.sh",
    "train.py",
    "requirements.txt",
]
CONFIGS_FILES = [
    "__init__.py",
    "baseline.py",
    "first_submission.py",
]
SRC_FILES = [
    "__init__.py",
    "utils.py",
    "data.py",
    "model.py",
    "optimizers.py",
    "checkpoint.py",
    "audit.py",
    "trainer.py",
]


def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> int:
    """Add ``src`` to the zip under ``arcname``. Returns size in bytes."""
    if not src.is_file():
        raise FileNotFoundError(f"missing required file: {src}")
    zf.write(src, arcname=arcname)
    return src.stat().st_size


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "step1_train.zip"
    out = HERE / target
    if out.exists():
        out.unlink()

    entries: list[tuple[str, int]] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in ROOT_FILES:
            sz = _add(zf, HERE / fn, fn)
            entries.append((fn, sz))
        for fn in CONFIGS_FILES:
            arc = f"configs/{fn}"
            sz = _add(zf, HERE / "configs" / fn, arc)
            entries.append((arc, sz))
        for fn in SRC_FILES:
            arc = f"src/{fn}"
            sz = _add(zf, HERE / "src" / fn, arc)
            entries.append((arc, sz))

    total = out.stat().st_size
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    entries.sort()

    print(f"Wrote {out}")
    print(f"Size: {total / 1024:.1f} KiB ({total} bytes)")
    print(f"SHA256[:16]: {sha}")
    print(f"Entries ({len(entries)}):")
    for name, sz in entries:
        print(f"  {sz:>10} bytes  {name}")

    if total > 100 * 1024 * 1024:
        print("ERROR: zip exceeds 100 MB platform cap", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
