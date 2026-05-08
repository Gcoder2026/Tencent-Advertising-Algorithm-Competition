"""Build the Step-3 (model evaluation) submission zip for AngelML.

Step-3 zip contains ONLY the files infer.py needs at evaluation time:
  infer.py, prepare.sh, requirements.txt,
  configs/{__init__, baseline}.py,
  src/{__init__, utils, data, model, checkpoint}.py.

Excluded (training-only): train.py, src/trainer.py, src/optimizers.py,
src/audit.py, configs/first_submission.py, tests/.

The trained checkpoint itself is NOT in this zip — it lives on the platform
under $MODEL_OUTPUT_PATH after Step 2 (Export Model) publishes it.

Usage:
    python build_step3_submission.py [output.zip]

Default output: ``step3_infer.zip`` next to this script.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent

ROOT_FILES = [
    "infer.py",
    "prepare.sh",
    "requirements.txt",
]
CONFIGS_FILES = [
    "__init__.py",
    "baseline.py",
]
SRC_FILES = [
    "__init__.py",
    "utils.py",
    "data.py",
    "model.py",
    "checkpoint.py",
]


def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> int:
    if not src.is_file():
        raise FileNotFoundError(f"missing required file: {src}")
    zf.write(src, arcname=arcname)
    return src.stat().st_size


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "step3_infer.zip"
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
