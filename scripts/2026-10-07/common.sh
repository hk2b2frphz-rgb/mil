#!/usr/bin/env bash
# Shared, dated three-arm experiment settings. Source from the repo root.
set -euo pipefail
export COMPARE_N="${COMPARE_N:-10000}"
[[ "$COMPARE_N" =~ ^[0-9]+$ && "$COMPARE_N" -ge 2 ]] || { echo 'COMPARE_N must be >=2' >&2; exit 1; }
# New default ID keeps the revised AI direct-TTS arm separate from old KABURI
# data, prepared datasets, checkpoints and benchmark results.
export COMPARE_ID="${COMPARE_ID:-aizuchi_compare_2026-10-08_${COMPARE_N}}"
[[ "$COMPARE_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid COMPARE_ID' >&2; exit 1; }
export COMPARE_ROOT="$PWD/data/runs/$COMPARE_ID"
export COMPARE_SOURCE="$COMPARE_ROOT/shared/dialogues.jsonl"
export COMPARE_VOCAB="$PWD/scripts/2026-09-15/aizuchi_vocab_no_bare_un.tsv"
export COMPARE_BANK="${COMPARE_BANK:-$PWD/data/runs/diversity/real_aizuchi_bank_v2}"
export SEED="${COMPARE_SEED:-0}"
export AIZUCHI_VOCAB_FILE="$COMPARE_VOCAB"
export NUM_CASES="$COMPARE_N" NUM_DIALOGUES="$COMPARE_N"
export NUM_SHARDS="${COMPARE_SHARDS:-4}"
[[ "$NUM_SHARDS" =~ ^[0-9]+$ && "$NUM_SHARDS" -ge 1 && "$NUM_SHARDS" -le "$COMPARE_N" ]] || {
    echo 'COMPARE_SHARDS must be positive and at most COMPARE_N' >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_SHARDS - 1)))"
# Zero spares: replacing a failed dialogue would break the paired comparison.
export SPARE_RATIO=0 RESUME=auto DROP_GREETING=1 QWEN_NO_OPENING_GREETING=1
export CLONE_OUT_DIR_MOSHI="${COMPARE_CLONE_DIR:-$PWD/data/clone_examples/99999}"
export QWEN_VOICE_MODE=mixed REF_RANK=1 CLONE_MODE=in-context
export USER_SPEAKER_POOL="${COMPARE_USER_VOICES:-Ono_Anna,Sohee,Vivian,Dylan,Eric,Aiden}"
export KEEP_TEXT=1 MATCH_TOP_K=5 RASTER_MODE=pred
export FDB_OPENING_GREETING=0 FDB_SEEDS="${COMPARE_EVAL_SEEDS:-0}"
export FDB_DATA_DIR="$PWD/data/full_duplex_ja_${COMPARE_ID}"
# Clear data routing inherited through PBS -V; each stage sets its own paths.
unset DIALOGUES_JSONL SOURCE_BATCH_ID BATCH_ID OUT_ROOT SRC_RUN_DIR
unset USER_ONLY_OUT DIALOGUE_OUT FULL_RENDER AIZUCHI_REPOSITION_SOURCE
unset BACKCHANNEL_TEXT BACKCHANNEL_DIST
unset DIALOGUE_EXTRA_ARGS DIALOGUE_EXTRA_ARGS_FILE
unset USER_REF_WAV USER_REF_TEXT CLONE_OUT_DIR_USER MOSHI_REF_WAV MOSHI_REF_TEXT

if [[ -n "${ARM:-}" ]]; then
    case "$ARM" in real_v2|traditional_overlap|ai_placement) ;; *) echo "Unknown ARM=$ARM" >&2; exit 1 ;; esac
    export RUN_ID="${COMPARE_ID}_${ARM}"
    export ARM_ROOT="$COMPARE_ROOT/$ARM"
fi
