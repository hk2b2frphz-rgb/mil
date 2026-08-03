#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$POC_ROOT/.." && pwd)"

INPUT="${1:-$POC_ROOT/artifacts/train_dialogues.jsonl}"
OUTPUT="${2:-$POC_ROOT/artifacts/tts}"
NUM_DIALOGUES="${NUM_DIALOGUES:-}"
TTS_DEVICE="${TTS_DEVICE:-cuda}"
TTS_DTYPE="${TTS_DTYPE:-float16}"
LEAD_IN_SEC="${LEAD_IN_SEC:-0.3}"
GAP_SEC="${GAP_SEC:-0.2}"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: dialogue JSONL not found: $INPUT" >&2
    echo "Run: python screw_poc/scripts/generate_dialogues.py" >&2
    exit 1
fi

ARGS=(
    --out-dir "$OUTPUT/training_set"
    --dialogues-jsonl "$INPUT"
    --model "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    --speaker-moshi "Ono_Anna"
    --speaker-user "Serena"
    --user-speaker-pool "Serena,Sohee,Vivian,Dylan,Eric,Aiden,Uncle_Fu,Ryan"
    --speaker-other "Dylan"
    --speaker-background "Ryan"
    --device "$TTS_DEVICE"
    --dtype "$TTS_DTYPE"
    --lead-in-sec "$LEAD_IN_SEC"
    --gap-sec "$GAP_SEC"
    --no-opening-greeting
    --no-emotion
    --whole-utterance
)
if [[ -n "$NUM_DIALOGUES" ]]; then
    ARGS+=(--num-dialogues "$NUM_DIALOGUES")
fi

cd "$REPO_ROOT"
uv run python scripts/generate_qwen3_tts_data.py "${ARGS[@]}"
