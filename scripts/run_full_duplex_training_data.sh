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
TTS_BACKEND="${TTS_BACKEND:-qwen3}"

# Natural Japanese turn-taking timing.
# Short transition gap (Stivers et al. 2009: Japanese gaps are near-zero).
GAP_SEC="${GAP_SEC:-0.2}"
LEAD_IN_SEC="${LEAD_IN_SEC:-0.3}"

USE_CASES_PATH="$OUT_ROOT/use_cases.jsonl"
DIALOGUES_DIR="$OUT_ROOT/llm_dialogues"
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
    uc_args=(
        uv run python scripts/build_full_duplex_training_use_cases.py
        --out-path "$USE_CASES_PATH"
        --num "$NUM_CASES"
        --seed "$SEED"
        --tasks "$FULL_DUPLEX_TASKS"
        --listening-ratio "$LISTENING_RATIO"
    )
    # Optional concrete talking-point bank for content diversity.
    if [[ -n "${CONTENT_SEEDS:-}" && -s "${CONTENT_SEEDS}" ]]; then
        uc_args+=(--content-seeds "$CONTENT_SEEDS")
        echo "[fdb] using content seeds: $CONTENT_SEEDS"
    fi
    "${uc_args[@]}"
fi

if has_step dialogues; then
    llm_args=(
        uv run python scripts/generate_synthetic_moshi_training_data.py
        --out-dir "$DIALOGUES_DIR"
        --use-cases-jsonl "$USE_CASES_PATH"
        --num-dialogues "$NUM_CASES"
        --seed "$SEED"
        --mode dialogues-only
        --llm-backend "${LLM_BACKEND:-transformers-subprocess}"
        --llm-model "${LLM_MODEL:-google/gemma-2-2b-it}"
        --llm-dtype "${LLM_DTYPE:-bfloat16}"
        --llm-temperature "${LLM_TEMPERATURE:-0.8}"
        --llm-max-new-tokens "${LLM_MAX_NEW_TOKENS:-4096}"
        --llm-timeout-sec "${LLM_TIMEOUT_SEC:-7200}"
        --dialogue-generation-mode "${DIALOGUE_GENERATION_MODE:-single}"
        --multi-agent-min-pairs "${MULTI_AGENT_MIN_PAIRS:-3}"
        --multi-agent-max-pairs "${MULTI_AGENT_MAX_PAIRS:-5}"
        --multi-agent-user-temperature "${MULTI_AGENT_USER_TEMPERATURE:-0.85}"
        --multi-agent-system-temperature "${MULTI_AGENT_SYSTEM_TEMPERATURE:-0.2}"
        --multi-agent-judge-temperature "${MULTI_AGENT_JUDGE_TEMPERATURE:-0.0}"
        --multi-agent-aizuchi-temperature "${MULTI_AGENT_AIZUCHI_TEMPERATURE:-0.2}"
        --multi-agent-aizuchi-min-chars "${MULTI_AGENT_AIZUCHI_MIN_CHARS:-32}"
        --multi-agent-max-aizuchi-per-user "${MULTI_AGENT_MAX_AIZUCHI_PER_USER:-1}"
    )
    if [[ "${NO_MULTI_AGENT_AIZUCHI:-0}" == "1" ]]; then
        llm_args+=(--no-multi-agent-aizuchi)
    fi
    if [[ "${DUMP_AGENT_PROMPTS:-0}" == "1" ]]; then
        llm_args+=(--dump-agent-prompts)
    fi
    if [[ "${LLM_BACKEND:-transformers-subprocess}" == "openai-compatible" ]]; then
        llm_args+=(
            --llm-api-base "${LLM_API_BASE:-http://127.0.0.1:8080/v1}"
            --llm-api-key "${LLM_API_KEY:-}"
            --llm-frequency-penalty "${LLM_FREQUENCY_PENALTY:-0.3}"
            --llm-presence-penalty "${LLM_PRESENCE_PENALTY:-0.1}"
            --llm-repetition-penalty "${LLM_REPETITION_PENALTY:-1.1}"
        )
        # Reasoning models (gpt-oss): keep effort low so the JSON channel is
        # emitted, and force JSON output. Both are no-ops if left empty.
        if [[ -n "${LLM_REASONING_EFFORT:-}" ]]; then
            llm_args+=(--llm-reasoning-effort "$LLM_REASONING_EFFORT")
        fi
        if [[ -n "${LLM_RESPONSE_FORMAT:-}" ]]; then
            llm_args+=(--llm-response-format "$LLM_RESPONSE_FORMAT")
        fi
    fi
    if [[ "${ALLOW_TEMPLATE_FALLBACK:-0}" == "1" ]]; then
        llm_args+=(--allow-template-fallback)
    fi
    "${llm_args[@]}"
    # Drop any stale enriched file so a regenerated dialogues.jsonl is never
    # rendered from previous-run enrichment.
    rm -f "$DIALOGUES_ENRICHED"
fi

if has_step enrich; then
    enrich_args=(
        uv run python scripts/enrich_dialogue_timing.py
        --in "$DIALOGUES_RAW"
        --out "$DIALOGUES_ENRICHED"
        --seed "$SEED"
    )
    # Backchannel density knobs (defaults match the script). Lower SEC_PER_AIZUCHI
    # = more frequent あいづち; higher AIZUCHI_PROB = more user turns get them.
    [[ -n "${SEC_PER_AIZUCHI:-}" ]] && enrich_args+=(--sec-per-aizuchi "$SEC_PER_AIZUCHI")
    [[ -n "${AIZUCHI_PROB:-}" ]] && enrich_args+=(--aizuchi-prob "$AIZUCHI_PROB")
    [[ -n "${MAX_AIZUCHI_PER_TURN:-}" ]] && enrich_args+=(--max-aizuchi-per-turn "$MAX_AIZUCHI_PER_TURN")
    # Apply backchannels to labeled-task dialogues too (not just free-form chat).
    if [[ "${ENRICH_LABELED_TASKS:-0}" == "1" ]]; then
        enrich_args+=(--enrich-labeled-tasks)
    fi
    "${enrich_args[@]}"
fi

# Use the enriched dialogues only if they exist and are at least as new as the
# raw dialogues; otherwise fall back to the raw ones.
AUDIO_DIALOGUES="$DIALOGUES_RAW"
if [[ -n "${DIALOGUES_JSONL:-}" ]]; then
    AUDIO_DIALOGUES="$DIALOGUES_JSONL"
elif [[ -s "$DIALOGUES_ENRICHED" && ! "$DIALOGUES_RAW" -nt "$DIALOGUES_ENRICHED" ]]; then
    AUDIO_DIALOGUES="$DIALOGUES_ENRICHED"
fi

if has_step audio; then
    audio_args=(
        uv run python scripts/generate_qwen3_tts_data.py
        --out-dir "$TRAINING_DIR"
        --dialogues-jsonl "$AUDIO_DIALOGUES"
        --tts-backend "$TTS_BACKEND"
        --model "${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
        --device "${TTS_DEVICE:-cuda}"
        --dtype "${TTS_DTYPE:-bfloat16}"
        --gap-sec "$GAP_SEC"
        --lead-in-sec "$LEAD_IN_SEC"
        --speaker-other "${TTS_SPEAKER_OTHER:-Dylan}"
        --speaker-background "${TTS_SPEAKER_BACKGROUND:-Ryan}"
    )
    if [[ "$TTS_BACKEND" == "moss-ttsd" ]]; then
        audio_args+=(
            --moss-model "${MOSS_TTS_MODEL:-OpenMOSS-Team/MOSS-TTSD-v1.0}"
            --moss-codec-model "${MOSS_CODEC_MODEL:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"
        )
        if [[ -n "${MOSS_REF_USER:-}" ]]; then
            audio_args+=(--moss-ref-user "$MOSS_REF_USER")
            [[ -n "${MOSS_REF_TEXT_USER:-}" ]] && audio_args+=(--moss-ref-text-user "$MOSS_REF_TEXT_USER")
        fi
        if [[ -n "${MOSS_REF_MOSHI:-}" ]]; then
            audio_args+=(--moss-ref-moshi "$MOSS_REF_MOSHI")
            [[ -n "${MOSS_REF_TEXT_MOSHI:-}" ]] && audio_args+=(--moss-ref-text-moshi "$MOSS_REF_TEXT_MOSHI")
        fi
        if [[ -n "${MOSS_REF_OTHER:-}" ]]; then
            audio_args+=(--moss-ref-other "$MOSS_REF_OTHER")
            [[ -n "${MOSS_REF_TEXT_OTHER:-}" ]] && audio_args+=(--moss-ref-text-other "$MOSS_REF_TEXT_OTHER")
        fi
        if [[ -n "${MOSS_REF_BACKGROUND:-}" ]]; then
            audio_args+=(--moss-ref-background "$MOSS_REF_BACKGROUND")
            [[ -n "${MOSS_REF_TEXT_BACKGROUND:-}" ]] && audio_args+=(--moss-ref-text-background "$MOSS_REF_TEXT_BACKGROUND")
        fi
    fi
    "${audio_args[@]}"
fi

echo "manifest: $TRAINING_DIR/synthetic_moshi_train.jsonl"
echo "reuse with: SRC_RUN_DIR=$OUT_ROOT"
