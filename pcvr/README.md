# PCVR — TAAC 2026 / KDD Cup submission

Solution scaffold for the Tencent KDD Cup 2026 PCVR (post-click conversion-rate) challenge.

## Layout
- `configs/baseline.py` — typed Config dataclass; one experiment = one config file.
- `src/` — library code (data, model, trainer, optimizers, checkpoint, audit, utils).
- `train.py`, `infer.py` — platform Step-1 / Step-3 entries.
- `run.sh`, `prepare.sh`, `build_submission.sh` — platform glue.
- `tests/` — pytest unit + smoke tests.

## Local dev
Most tests skip without torch installed. For full end-to-end, create a torch-enabled conda env (Python 3.10 to match the platform):
```
conda create -n taac python=3.10
conda activate taac
pip install -r requirements.txt
python -m pytest tests/    # 37/37 should pass once torch is in
```

## Platform submission
- Step 1 (training): zip the project, upload, AngelML runs `run.sh`.
- Step 2 (export): publish a `global_step*` checkpoint in the UI.
- Step 3 (inference): run `bash build_submission.sh`, upload the resulting zip.

See `ARCHITECTURE.md` for design decisions.
