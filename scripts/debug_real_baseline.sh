#!/usr/bin/env bash
set -euo pipefail

# One-case, end-to-end smoke test for the HPC-local comparison baselines.
# It is intentionally a normal Bash program (not PBS): run it inside an
# already allocated interactive V100 session before submitting the batch.
#
#   bash scripts/debug_real_baseline.sh cascade
#   bash scripts/debug_real_baseline.sh speechllm
#
# Useful overrides:
#   DEBUG_CASES_PER_TASK=1  (default)   DEBUG_KEEP_OUTPUT=1
#   CASCADE_ASR_MODEL=turbo               (default for cascade smoke test)
#   CASCADE_LLM_MODEL=google/gemma-4-E2B-it (default for cascade)
#   CASCADE_LLM_MAX_NEW_TOKENS=80         (default for cascade smoke test)
#   CASCADE_* / SPEECHLLM_*             passed through unchanged

SYSTEM="${1:-}"
case "$SYSTEM" in cascade|speechllm) ;; *)
    echo "Usage: bash scripts/debug_real_baseline.sh {cascade|speechllm}" >&2
    exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not on PATH." >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: run this in a GPU allocation; nvidia-smi is unavailable." >&2; exit 2; }

DEBUG_CASES_PER_TASK="${DEBUG_CASES_PER_TASK:-1}"
if ! [[ "$DEBUG_CASES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: DEBUG_CASES_PER_TASK must be a positive integer." >&2
    exit 2
fi
DEBUG_ID="debug_${SYSTEM}_$(date +%Y%m%d_%H%M%S)"
DEBUG_OUT="${DEBUG_OUT:-$REPO_ROOT/eval_runs/debug/$DEBUG_ID}"
mkdir -p "$DEBUG_OUT"
exec > >(tee "$DEBUG_OUT/run.log") 2>&1

echo "===== $SYSTEM smoke test ====="
echo "output: $DEBUG_OUT"
echo "gpu:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo "main runtime:"
uv run python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
echo "isolated worker runtime:"
if [[ "$SYSTEM" == "speechllm" ]]; then
    uv run --project gemma_runtime python -c 'from transformers import Qwen2AudioForConditionalGeneration; print("Qwen2-Audio worker import: OK")'
else
    uv run --project gemma_runtime python -c 'from transformers import AutoModelForCausalLM; print("Gemma worker import: OK")'
fi

# Avoid optional UTMOS and second-pass output ASR while proving the important
# path (input audio -> model response -> audible output -> deterministic score).
# Full batch evaluation restores both defaults.
if [[ "$SYSTEM" == "cascade" ]]; then
    # This smoke path intentionally uses the same public Gemma checkpoint and
    # worker task as synthetic dialogue generation.  Keep the values explicit
    # here: a debug run must never accidentally fall back to a gated Gemma 2
    # model or another caller's shell setting.
    MODEL_ID="${MODEL_ID:-debug_cascade_gemma4_e2b}"
    CASCADE_LLM_MODEL="${CASCADE_LLM_MODEL:-google/gemma-4-E2B-it}"
    CASCADE_LLM_DTYPE="${CASCADE_LLM_DTYPE:-float16}"
    CASCADE_LLM_TASK="${CASCADE_LLM_TASK:-any-to-any}"
    CASCADE_ASR_MODEL="${CASCADE_ASR_MODEL:-turbo}"
    CASCADE_LLM_MAX_NEW_TOKENS="${CASCADE_LLM_MAX_NEW_TOKENS:-80}"
    echo "cascade asr:      $CASCADE_ASR_MODEL"
    echo "cascade llm:      $CASCADE_LLM_MODEL"
    echo "cascade llm dtype: $CASCADE_LLM_DTYPE"
    echo "cascade llm task:  $CASCADE_LLM_TASK"
    echo "cascade llm max:   $CASCADE_LLM_MAX_NEW_TOKENS"
    env MODEL_ID="$MODEL_ID" RUN_ID="$DEBUG_ID" REAL_OUT_DIR="$DEBUG_OUT" \
        REAL_CASES_PER_TASK="$DEBUG_CASES_PER_TASK" REAL_MOS_BACKEND=none \
        REAL_BACKCHANNEL=0 CASCADE_SKIP_OUTPUT_ALIGNMENT=1 \
        CASCADE_LLM_MODEL="$CASCADE_LLM_MODEL" \
        CASCADE_LLM_DTYPE="$CASCADE_LLM_DTYPE" \
        CASCADE_LLM_TASK="$CASCADE_LLM_TASK" \
        CASCADE_ASR_MODEL="$CASCADE_ASR_MODEL" \
        CASCADE_LLM_MAX_NEW_TOKENS="$CASCADE_LLM_MAX_NEW_TOKENS" \
        bash scripts/run_real_cascade_eval.sh
else
    MODEL_ID="${MODEL_ID:-debug_speechllm}"
    env MODEL_ID="$MODEL_ID" RUN_ID="$DEBUG_ID" REAL_OUT_DIR="$DEBUG_OUT" \
        REAL_CASES_PER_TASK="$DEBUG_CASES_PER_TASK" REAL_MOS_BACKEND=none \
        REAL_BACKCHANNEL=0 \
        bash scripts/run_real_speechllm_eval.sh
fi

meta_count=$(find "$DEBUG_OUT/inference" -path '*/seed_*/output.meta.json' -type f | wc -l)
wav_count=$(find "$DEBUG_OUT/inference" -path '*/seed_*/output.wav' -type f -size +44c | wc -l)
if [[ "$meta_count" -lt 1 || "$wav_count" -lt 1 ]]; then
    echo "ERROR: smoke test completed without required output.meta.json/output.wav artifacts." >&2
    exit 1
fi
echo "SUCCESS: $SYSTEM completed $meta_count case(s), with $wav_count audible-output artifact(s)."
echo "summary: $DEBUG_OUT/benchmark_results/summary.json"
if [[ "$SYSTEM" == "cascade" ]]; then
    uv run python eval/profile_cascade_run.py --run-dir "$DEBUG_OUT/inference"
fi
