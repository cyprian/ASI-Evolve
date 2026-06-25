#!/bin/bash
set -euo pipefail

STEP_DIR="$(pwd)"
CODE_PATH="${STEP_DIR}/code"
RESULTS_PATH="${STEP_DIR}/results.json"
EXPERIMENT_DIR="$(dirname "$(dirname "$STEP_DIR")")"

source /home/cyprian/miniconda3/etc/profile.d/conda.sh
conda activate p13c13

python "${EXPERIMENT_DIR}/evaluator.py" "${CODE_PATH}" "${RESULTS_PATH}"
