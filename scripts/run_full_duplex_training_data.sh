#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-full_duplex_$(date +%Y-%m-%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/runs/$RUN_ID}"
NUM_CASES="${NUM_CASES:-140}"
FULL_DUPLEX_TASKS="${FULL_DUPLEX_TASKS:-all}"
SEED="${SEED:-0}"
STEPS="${STEPS:-use_cases,dialogues,enrich,audio}"
LISTENING_RATIO="${LISTENING_RATIO:-0.7}"

# Natural Japanese turn-taking timing.
# Short transition gap (Stivers et al. 2009: Japanese gaps are near-zero).
GAP_SEC="${GAP_SEC:-0.2}"
LEAD_IN_SEC="${LEAD_IN_SEC:-0.3}"

USE_CASES_PATH="$OUT_ROOT/use_cases.jsonl"
DIALOGUES_DIR="$OUT_ROOT/gemma_dialogues"
DIALOGUES_RAW="$DIALOGUES_DIR/dialogues.jsonl"
DIALOGUES_ENRICHED="$DIALOGUES_DIR/dialogues_enriched.jsonl"
TRAINING_DIR="$OUT_ROOT/training_set"
mkdir -p "$OUT_ROOT"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

has_step() {
    [[ ",$STEPS," == *",$1,"* ]]
}

echo "===== Full-duplex Japanese training data ====="
echo "run_id:     $RUN_ID"
echo "out_root:   $OUT_ROOT"
echo "num_cases:  $NUM_CASES"
echo "tasks:      $FULL_DUPLEX_TASKS"
echo "steps:      $STEPS"

if has_step use_cases; then
    uv run python scripts/build_full_duplex_training_use_cases.py \
        --out-path "$USE_CASES_PATH" \
        --num "$NUM_CASES" \
        --seed "$SEED" \
        --tasks "$FULL_DUPLEX_TASKS" \
        --listening-ratio "$LISTENING_RATIO"
fi

if has_step dialogues; then
    gemma_args=(
        uv run python scripts/generate_synthetic_moshi_training_data.py
        --out-dir "$DIALOGUES_DIR"
        --use-cases-jsonl "$USE_CASES_PATH"
        --num-dialogues "$NUM_CASES"
        --seed "$SEED"
        --mode dialogues-only
        --gemma-backend "${GEMMA_BACKEND:-transformers-subprocess}"
        --gemma-model "${GEMMA_MODEL:-google/gemma-4-E2B-it}"
        --gemma-dtype "${GEMMA_DTYPE:-bfloat16}"
        --gemma-max-new-tokens "${GEMMA_MAX_NEW_TOKENS:-1100}"
        --gemma-timeout-sec "${GEMMA_TIMEOUT_SEC:-7200}"
    )
    if [[ "${ALLOW_TEMPLATE_FALLBACK:-0}" == "1" ]]; then
        gemma_args+=(--allow-template-fallback)
    fi
    "${gemma_args[@]}"
fi

if has_step enrich; then
    uv run python scripts/enrich_dialogue_timing.py \
        --in "$DIALOGUES_RAW" \
        --out "$DIALOGUES_ENRICHED" \
        --seed "$SEED"
fi

# Use the enriched dialogues for audio if they exist, otherwise the raw ones.
AUDIO_DIALOGUES="$DIALOGUES_RAW"
if [[ -s "$DIALOGUES_ENRICHED" ]]; then
    AUDIO_DIALOGUES="$DIALOGUES_ENRICHED"
fi

if has_step audio; then
    uv run python scripts/generate_qwen3_tts_data.py \
        --out-dir "$TRAINING_DIR" \
        --dialogues-jsonl "$AUDIO_DIALOGUES" \
        --model "${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" \
        --device "${TTS_DEVICE:-cuda}" \
        --dtype "${TTS_DTYPE:-bfloat16}" \
        --gap-sec "$GAP_SEC" \
        --lead-in-sec "$LEAD_IN_SEC" \
        --speaker-other "${TTS_SPEAKER_OTHER:-Dylan}" \
        --speaker-background "${TTS_SPEAKER_BACKGROUND:-Ryan}"
fi

echo "manifest: $TRAINING_DIR/synthetic_moshi_train.jsonl"
echo "reuse with: SRC_RUN_DIR=$OUT_ROOT"
