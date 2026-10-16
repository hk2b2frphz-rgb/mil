#!/usr/bin/env bash
set -euo pipefail
source scripts/setup_proxy.sh
export MODEL_ID="$RUN_ID"
export CUDA_VISIBLE_DEVICES=0
export MODEL_WEIGHT="$PWD/experiments/_fullft_sweeps/${RUN_ID}_f01/exported/best/model.safetensors"
test -s "$MODEL_WEIGHT"
export MODEL_CONFIG="$(dirname "$MODEL_WEIGHT")/moshi_lm_kwargs.json"
export FDB_OUT_DIR="$PWD/eval_runs/full_duplex/$RUN_ID"
unset LORA_PATH
export FDB_BATCH_PARALLELISM=3 REFRESH_FDB_DATA=0 FDB_TTS_BUILD_GPUS=0
bash scripts/run_full_duplex_eval.sh
# A report is complete only after all three summaries exist. Missing results
# are reported as pending, never filled with zero or compared as successes.
python3 scripts/2026-10-07/report_comparison.py --root "$COMPARE_ROOT" --id "$COMPARE_ID" --allow-pending
