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
#                It also writes a user-only copy for the next step.
#   2. Qwen3     synthesize the USER SIDE ONLY. The listener is never
#                synthesized: its audio comes from the bank, and rendering it
#                to throw it away would also push the user's turns later on
#                the timeline, since a listener turn that is not overlapped
#                occupies time of its own. V100.
#   3. KABURI    where the listener should come in, from the full dialogue
#                text. CPU, no acoustic model.
#   4. insert    write the bank clips into the silent listener channel at
#                KABURI's positions, matched on word and length.
#
# FULL_RENDER=1 goes back to synthesizing the whole dialogue and replacing the
# listener channel afterwards. That costs the wasted synthesis and the shifted
# timeline, and buys one thing: the discarded listener audio is an A/B
# baseline -- the same words in the same places, synthesized rather than
# retrieved.
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
# CORPUS_ROOT can be set directly for a corpus that is not one of the aizuchi
# presets -- the listening-mode 500 uses that.
CORPUS_ROOT="${CORPUS_ROOT:-aizuchi_${AIZUCHI_PRESET}_3000_${AIZUCHI_VERSION}}"
SOURCE_DIALOGUES="${DIALOGUES_JSONL:-$REPO_ROOT/data/runs/$CORPUS_ROOT/dialogue/llm_dialogues/dialogues.jsonl}"
# The dialogues may already carry the real vocabulary (generated that way
# rather than rewritten into it). Then the rewrite must not draw new words --
# it only writes the user-only copy and the vocabulary list.
KEEP_TEXT="${KEEP_TEXT:-0}"
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
USER_ONLY_OUT="$OUT_ROOT/dialogues_user_only.jsonl"
FULL_RENDER="${FULL_RENDER:-0}"
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
    --user-only-jsonl "$USER_ONLY_OUT"
    --num-dialogues "$NUM_DIALOGUES"
    --seed "$SEED"
)
# 語彙を渡さないと文字数で相槌を判定することになり、冒頭の名乗りと
# 「聞いていますか?」への応答まで相槌扱いになる。前者は user のみの写しから
# 落ちて音声に残らず、後者はバンクに無いので別の語の音が当たる。
if [[ -n "${AIZUCHI_VOCAB_FILE:-}" ]]; then
    REWRITE_ARGS+=(--aizuchi-vocab-file "$AIZUCHI_VOCAB_FILE")
fi
if [[ "$KEEP_TEXT" == "1" ]]; then
    REWRITE_ARGS+=(--keep-text)
elif [[ -n "${BACKCHANNEL_TEXT:-}" ]]; then
    REWRITE_ARGS+=(--text "$BACKCHANNEL_TEXT")
elif [[ -n "${BACKCHANNEL_DIST:-}" ]]; then
    REWRITE_ARGS+=(--dist "$BACKCHANNEL_DIST")
fi
uv run python scripts/rewrite_aizuchi_vocab.py "${REWRITE_ARGS[@]}"
VOCAB_FILE="${DIALOGUE_OUT}.vocab.txt"

echo
if [[ "$FULL_RENDER" == "1" ]]; then
    TTS_DIALOGUES="$DIALOGUE_OUT"
    echo ">>> 2/4 Qwen3-TTS over the WHOLE dialogue (FULL_RENDER=1) -> $QWEN_DIR"
else
    TTS_DIALOGUES="$USER_ONLY_OUT"
    echo ">>> 2/4 Qwen3-TTS over the user side only -> $QWEN_DIR"
fi
(
    export CLONE_OUT_DIR_MOSHI CUDA_VISIBLE_DEVICES
    export SOURCE_BATCH_ID="${BATCH_ID}_src"
    export DIALOGUES_JSONL="$TTS_DIALOGUES"
    export BATCH_ID="${BATCH_ID}_qwen"
    export OUT_ROOT="$QWEN_ROOT"
    # 3 本の動作確認では 1 GPU・再開なしでよいが、本番の規模ではどちらも
    # 要る。10000 本を 1 GPU で回すと walltime に収まらず、RESUME=0 だと
    # 途中で切れたぶんが全部消える。既定は従来どおりなので smoke は不変。
    export NUM_DIALOGUES SPARE_RATIO=0 LOG_EVERY=1
    export NUM_SHARDS="${NUM_SHARDS:-1}"
    export RESUME="${RESUME:-0}"
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
