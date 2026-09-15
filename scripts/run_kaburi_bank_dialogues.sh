#!/usr/bin/env bash
set -euo pipefail

# Render a few dialogues with KABURI, then swap the listener's backchannels for
# banked ones, and print the turn-taking numbers for both.
#
#   bash scripts/run_kaburi_bank_dialogues.sh 3
#   BANK_DIR=data/runs/diversity/<run> bash scripts/run_kaburi_bank_dialogues.sh 5
#
# The point of the split: KABURI decides WHEN the listener reacts and the bank
# decides WHAT it sounds like. The swap edits the finished stereo file in place
# of re-synthesizing, so placement and the other speaker are sample-identical
# between the two outputs and the pair can be A/B'd directly.
#
# The bank is one word. In the 116 measured backchannels of the real recording
# "un" alone was 28, and the top four were 76% -- none of them in the
# vocabulary these corpora are generated with. So the spliced dialogue has a
# listener that only ever says "un", which is closer to the real frequencies
# than what it replaces, and is the cheapest way to hear whether banked audio
# at KABURI's timings holds up at all.
#
# Needs an A100 for the KABURI render (its codec decodes in bfloat16); the
# splice itself is CPU and can be re-run on its own afterwards with different
# matching settings:
#
#   uv run python scripts/splice_aizuchi_bank.py --training-dir <dir> \
#       --bank-dir <bank> --out-dir <dir>_bank --match-top-k 1
#
# Overrides: NUM_DIALOGUES, BANK_DIR, BANK_TEXT, BANK_TEMPERATURE, MATCH_TOP_K,
# AIZUCHI_PRESET, AIZUCHI_VERSION, DIALOGUES_JSONL, CLONE_OUT_DIR_MOSHI,
# CUDA_VISIBLE_DEVICES, RASTER_MODE, SEED.

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
    *)
        echo "ERROR: AIZUCHI_PRESET must be normal, eager or flood: $AIZUCHI_PRESET" >&2
        exit 1
        ;;
esac
AIZUCHI_VERSION="${AIZUCHI_VERSION:-$PRESET_VERSION}"
CORPUS_ROOT="aizuchi_${AIZUCHI_PRESET}_3000_${AIZUCHI_VERSION}"
SOURCE_BATCH_ID="qwen_aizuchi_3000_${AIZUCHI_PRESET}_${AIZUCHI_VERSION}"
DIALOGUES_JSONL="${DIALOGUES_JSONL:-$REPO_ROOT/data/runs/$CORPUS_ROOT/dialogue/llm_dialogues/dialogues.jsonl}"
CLONE_OUT_DIR_MOSHI="${CLONE_OUT_DIR_MOSHI:-$REPO_ROOT/data/clone_examples/99999}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RASTER_MODE="${RASTER_MODE:-pred}"
MATCH_TOP_K="${MATCH_TOP_K:-5}"
# Which backchannels a banked "un" may stand in for. Without this the rule is a
# character count, and a count cannot tell a continuer from an assessment:
# replacing "sou-nan-desu-ne" with "un" deletes the listener's reaction to what
# was said and the dialogue stops making sense.
AIZUCHI_VOCAB_FILE="${AIZUCHI_VOCAB_FILE:-$REPO_ROOT/scripts/2026-09-15/continuer_vocab.txt}"
SEED="${SEED:-0}"
STAMP="$(run_id_stamp)"

# Newest diversity run unless told otherwise. Printed either way, because
# "which bank was this" is the first thing to ask of a result.
if [[ -z "${BANK_DIR:-}" ]]; then
    BANK_DIR="$(ls -1dt "$REPO_ROOT"/data/runs/diversity/*/ 2>/dev/null | head -n 1 || true)"
    BANK_DIR="${BANK_DIR%/}"
fi
if [[ -z "$BANK_DIR" || ! -s "$BANK_DIR/samples.jsonl" ]]; then
    echo "ERROR: bank not found. Pass BANK_DIR=<a diversity run directory>." >&2
    echo "Build one with scripts/2026-09-14/qwen3_tts_diversity_probe.pbs" >&2
    exit 1
fi
if [[ ! -s "$DIALOGUES_JSONL" ]]; then
    echo "ERROR: dialogues JSONL not found: $DIALOGUES_JSONL" >&2
    exit 1
fi

BATCH_ID="${CORPUS_ROOT}_kaburi_${RASTER_MODE}_bank_${STAMP}"
OUT_ROOT="$REPO_ROOT/data/runs/smoke/$BATCH_ID"
PLAIN_DIR="$OUT_ROOT/shard_000/training_set"
BANK_OUT="$OUT_ROOT/bank/shard_000/training_set"

echo "===== KABURI + aizuchi bank ====="
echo "repo:          $REPO_ROOT"
echo "dialogues:     $DIALOGUES_JSONL"
echo "num_dialogues: $NUM_DIALOGUES"
echo "timing:        $RASTER_MODE"
echo "bank:          $BANK_DIR"
echo "bank_text:     ${BANK_TEXT:-un (default)}"
echo "started_at:    $(date -Iseconds)"
echo "================================="

echo
echo ">>> 1/2 KABURI render -> $PLAIN_DIR"
(
    export CLONE_OUT_DIR_MOSHI CUDA_VISIBLE_DEVICES SOURCE_BATCH_ID DIALOGUES_JSONL
    export BATCH_ID OUT_ROOT NUM_DIALOGUES
    export NUM_SHARDS=1 SPARE_RATIO=0 RESUME=0 LOG_EVERY=1
    export RASTER_MODE
    export SMOKE_REPORT_LABEL="$AIZUCHI_PRESET/kaburi-$RASTER_MODE"
    bash scripts/run_kaburi_tts.pbs
)

echo
echo ">>> 2/2 splice the bank -> $BANK_OUT"
SPLICE_ARGS=(
    --training-dir "$PLAIN_DIR"
    --bank-dir "$BANK_DIR"
    --out-dir "$BANK_OUT"
    --match-top-k "$MATCH_TOP_K"
    --seed "$SEED"
)
[[ -n "$AIZUCHI_VOCAB_FILE" ]] && SPLICE_ARGS+=(--aizuchi-vocab-file "$AIZUCHI_VOCAB_FILE")
[[ -n "${BANK_TEXT:-}" ]] && SPLICE_ARGS+=(--bank-text "$BANK_TEXT")
[[ -n "${BANK_TEMPERATURE:-}" ]] && SPLICE_ARGS+=(--bank-temperature "$BANK_TEMPERATURE")
uv run python scripts/splice_aizuchi_bank.py "${SPLICE_ARGS[@]}"

echo
echo "===== report ====="
# Overlap and switches should barely move: the bank changes the sound, not the
# placement. A large move means the swapped clips are running far past the
# slots KABURI left for them.
uv run python scripts/report_tts_smoke.py \
    --projection "${REPORT_PROJECTION:-3000}" \
    --json-out "$OUT_ROOT/report_bank_${STAMP}.json" \
    --training-dir "$PLAIN_DIR" --label "kaburi-$RASTER_MODE" \
    --training-dir "$BANK_OUT" --label "kaburi-$RASTER_MODE+bank"

echo
echo "listen:  $PLAIN_DIR/data_stereo  vs  $BANK_OUT/data_stereo"
echo "swaps:   $BANK_OUT/bank_swaps.jsonl"
echo "finished_at: $(date -Iseconds)"
