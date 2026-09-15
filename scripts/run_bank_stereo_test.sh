#!/usr/bin/env bash
set -euo pipefail

# Build stereo dialogues whose listener side comes entirely from the bank.
#
#   bash scripts/run_bank_stereo_test.sh 3
#   BACKCHANNEL_TEXT=un bash scripts/run_bank_stereo_test.sh 3   # one word only
#
# Four steps, and the first is the one that makes this test possible at all:
#
#   1. rewrite   the listener's backchannels, drawn from the frequencies
#                measured in the real recording. The corpus vocabulary has no
#                "un" in it -- the four words that were 76% of the 116 real
#                backchannels are none of them in it -- so a corpus to test a
#                bank of them against has to be made, not found. CPU, instant.
#   2. Qwen3     synthesize the whole dialogue as usual. The user side is what
#                is kept; the listener side is thrown away by step 4 and only
#                serves as the A/B baseline. V100.
#   3. KABURI    where the listener should come in. CPU, no acoustic model.
#   4. splice    replace the listener channel: each backchannel moves to
#                KABURI's gap and is voiced by a bank clip of the SAME word.
#
# So the output has a user side that was synthesized, a listener side that was
# retrieved, and timing that was predicted -- which is the whole proposal, end
# to end, for the first time.
#
# The bank must hold the words the rewrite puts in. The 20-spelling diversity
# run covers all seven of the default distribution; the temperature sweep holds
# only "un", so pair that one with BACKCHANNEL_TEXT.
#
# Overrides: BANK_DIR, BACKCHANNEL_TEXT, BACKCHANNEL_DIST, NUM_DIALOGUES,
# AIZUCHI_PRESET, DIALOGUES_JSONL, MATCH_TOP_K, RASTER_MODE, SEED, OUT_ROOT,
# CLONE_OUT_DIR_MOSHI, CUDA_VISIBLE_DEVICES.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PBS_O_WORKDIR="$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/run_id_utils.sh"

NUM_DIALOGUES="${1:-${NUM_DIALOGUES:-3}}"
if ! [[ "$NUM_DIALOGUES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_DIALOGUES must be a positive integer: $NUM_DIALOGUES" >&2
    exit 1
fi

AIZUCHI_PRESET="${AIZUCHI_PRESET:-normal}"
case "$AIZUCHI_PRESET" in
    normal) PRESET_VERSION="v6" ;;
    eager)  PRESET_VERSION="v7" ;;
    flood)  PRESET_VERSION="v6" ;;
    *) echo "ERROR: AIZUCHI_PRESET must be normal, eager or flood" >&2; exit 1 ;;
esac
AIZUCHI_VERSION="${AIZUCHI_VERSION:-$PRESET_VERSION}"
CORPUS_ROOT="aizuchi_${AIZUCHI_PRESET}_3000_${AIZUCHI_VERSION}"
SOURCE_DIALOGUES="${DIALOGUES_JSONL:-$REPO_ROOT/data/runs/$CORPUS_ROOT/dialogue/llm_dialogues/dialogues.jsonl}"
RASTER_MODE="${RASTER_MODE:-pred}"
MATCH_TOP_K="${MATCH_TOP_K:-5}"
SEED="${SEED:-0}"
CLONE_OUT_DIR_MOSHI="${CLONE_OUT_DIR_MOSHI:-$REPO_ROOT/data/clone_examples/99999}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(run_id_stamp)"

if [[ ! -s "$SOURCE_DIALOGUES" ]]; then
    echo "ERROR: dialogues JSONL not found: $SOURCE_DIALOGUES" >&2
    exit 1
fi
if [[ -z "${BANK_DIR:-}" ]]; then
    BANK_DIR="$(ls -1dt "$REPO_ROOT"/data/runs/diversity/*/ 2>/dev/null | head -n 1 || true)"
    BANK_DIR="${BANK_DIR%/}"
fi
if [[ -z "$BANK_DIR" || ! -s "$BANK_DIR/samples.jsonl" ]]; then
    echo "ERROR: bank not found. Pass BANK_DIR=<a diversity run directory>." >&2
    exit 1
fi

BATCH_ID="${CORPUS_ROOT}_bank_stereo_${STAMP}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/runs/smoke/$BATCH_ID}"
DIALOGUE_OUT="$OUT_ROOT/dialogues_bank.jsonl"
QWEN_ROOT="$OUT_ROOT/qwen"
QWEN_DIR="$QWEN_ROOT/shard_000/training_set"
PLACEMENT_OUT="$OUT_ROOT/kaburi_placement"
BANK_OUT="$OUT_ROOT/shard_000/training_set"

echo "===== bank stereo test ====="
echo "repo:        $REPO_ROOT"
echo "dialogues:   $SOURCE_DIALOGUES"
echo "bank:        $BANK_DIR"
echo "n:           $NUM_DIALOGUES"
echo "out:         $OUT_ROOT"
echo "started_at:  $(date -Iseconds)"
echo "============================"

echo
echo ">>> 1/4 rewrite the backchannels -> $DIALOGUE_OUT"
REWRITE_ARGS=(
    --dialogues-jsonl "$SOURCE_DIALOGUES"
    --out-jsonl "$DIALOGUE_OUT"
    --num-dialogues "$NUM_DIALOGUES"
    --seed "$SEED"
)
if [[ -n "${BACKCHANNEL_TEXT:-}" ]]; then
    REWRITE_ARGS+=(--text "$BACKCHANNEL_TEXT")
elif [[ -n "${BACKCHANNEL_DIST:-}" ]]; then
    REWRITE_ARGS+=(--dist "$BACKCHANNEL_DIST")
fi
uv run python scripts/rewrite_aizuchi_vocab.py "${REWRITE_ARGS[@]}"
VOCAB_FILE="${DIALOGUE_OUT}.vocab.txt"

echo
echo ">>> 2/4 Qwen3-TTS over the rewritten dialogues -> $QWEN_DIR"
(
    export CLONE_OUT_DIR_MOSHI CUDA_VISIBLE_DEVICES
    export SOURCE_BATCH_ID="${BATCH_ID}_src"
    export DIALOGUES_JSONL="$DIALOGUE_OUT"
    export BATCH_ID="${BATCH_ID}_qwen"
    export OUT_ROOT="$QWEN_ROOT"
    export NUM_DIALOGUES NUM_SHARDS=1 SPARE_RATIO=0 RESUME=0 LOG_EVERY=1
    bash scripts/run_qwen_tts_vllm_3000_4gpu.pbs
)

echo
echo ">>> 3/4 KABURI placement (no acoustic model) -> $PLACEMENT_OUT"
(
    export KABURI_REPO="${KABURI_REPO:-$REPO_ROOT/../kaburi-tts}"
    export DIALOGUES_JSONL="$DIALOGUE_OUT"
    export SOURCE_DIR="$QWEN_DIR"
    export BANK_DIR
    export OUT_ROOT="$OUT_ROOT/placement_bank"
    export AIZUCHI_VOCAB_FILE="$VOCAB_FILE"
    export NUM_DIALOGUES RASTER_MODE MATCH_TOP_K SEED
    bash scripts/run_kaburi_placement_bank.sh "$NUM_DIALOGUES"
)

echo
echo ">>> 4/4 done"
BANK_OUT="$OUT_ROOT/placement_bank/shard_000/training_set"
echo "listen: $QWEN_DIR/data_stereo  (baseline, listener synthesized)"
echo "    vs: $BANK_OUT/data_stereo  (listener from the bank at KABURI's gaps)"
echo "swaps:  $BANK_OUT/bank_swaps.jsonl"
echo "vocab:  $VOCAB_FILE"
echo "finished_at: $(date -Iseconds)"
