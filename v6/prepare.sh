#!/bin/bash
# Platform pre-inference hook. The AngelML image already provides:
#   - torch==2.7.1+cu126
#   - pyarrow==23.0.1
#   - numpy
# Add only what the inference path actually needs and is missing.
exit 0
