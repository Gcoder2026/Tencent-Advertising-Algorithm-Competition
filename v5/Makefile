# pcvr/Makefile — local development entry points.
# Use POSIX make (works under git-bash/WSL). Windows users: invoke via bash.

PYTHON ?= /c/Users/84447/anaconda3/envs/taac/python.exe

.PHONY: help test smoke build-step1 build-step3 prepare-sample repro clean

help:
	@echo "Targets:"
	@echo "  test           Run full pytest suite"
	@echo "  smoke          Run smoke test only"
	@echo "  prepare-sample Download HF data sample to data_sample_1000/"
	@echo "  build-step1    Build step1_train.zip"
	@echo "  build-step3    Build step3_infer.zip"
	@echo "  repro          End-to-end repro: prepare-sample -> build-step1 (no train; assumes torch+data ready on platform)"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v

prepare-sample:
	$(PYTHON) tools/prepare_hf_sample.py

build-step1:
	$(PYTHON) build_step1_submission.py

build-step3:
	$(PYTHON) build_step3_submission.py

# `repro` is a quick end-to-end check we can rebuild a Step-1 zip from a clean
# tree. It does NOT actually run training (that requires the platform GPU)
# but verifies all the moving pieces stay in sync.
repro: test build-step1
	@echo "OK — pytest green and step1_train.zip rebuilt"

clean:
	rm -f step1_train.zip step3_infer.zip
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
