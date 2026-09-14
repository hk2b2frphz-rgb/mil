#!/usr/bin/env bash
set -euo pipefail

# Borrow KABURI's timing only, and put banked backchannels at it. No GPU.
#
#   bash scripts/run_kaburi_placement_bank.sh 3
#   SOURCE_DIR=data/runs/<other_run>/training_set \
#     bash scripts/run_kaburi_placement_bank.sh 3
#
# SOURCE_DIR defaults to whatever the preset was already synthesized into --
# data/runs/<corpus>/tts/{merged,shard_*}/training_set, else the newest Qwen
# smoke run for it. It is printed either way.
#
# This is the "use only the turn-taking positions" route, as opposed to
# scripts/run_kaburi_bank_dialogues.sh, which renders the whole dialogue with
# KABURI and needs an A100 for the codec. Nothing of KABURI's audio is used
# here, so nothing of KABURI's audio has to be produced:
#
#   KABURI   when the listener reacts -- realizer + gap model, CPU, seconds
#   SOURCE   the user's speech -- an existing rendered corpus, already on disk
#   bank     what the backchannel sounds like -- a diversity run
#
# What transfers is the gap: for each backchannel, how long after (or before)
# the user's utterance ends the listener comes in. Negative gaps are the
# overlap the Qwen path never produces. The gap is measured on KABURI's
# timeline and applied to the source's real one, so the user's audio is never
# stretched or moved -- only the backchannel is, and then replaced.
#
# Backchannels are matched by order of appearance. If the counts disagree the
# dialogue is skipped rather than silently misaligned, so a mismatch shows up
# as a skip line and not as backchannels landing in the wrong places.
#
# Needs the KABURI checkout (scripts/setup_kaburi_env.sh) for the timing model
# and a reference pack, both of which build on CPU. It does NOT need the
# acoustic checkpoint to be usable, and never loads it.
#
# Overrides: SOURCE_DIR, BANK_DIR, BANK_TEXT, BANK_TEMPERATURE, MATCH_TOP_K,
# DIALOGUES_JSONL, AIZUCHI_PRESET, NUM_DIALOGUES, RASTER_MODE, OUT_ROOT, SEED.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PBS_O_WORKDIR="$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/run_id_utils.sh"
# The timing model still pulls its text tokenizer from huggingface.co, and the
# reference pack decodes with the codec, so a compute node needs the proxy just
# as the render job does -- it is only the acoustic checkpoint that is skipped.
if [[ -f "$REPO_ROOT/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/scripts/setup_proxy.sh"
fi

export KABURI_REPO="${KABURI_REPO:-$REPO_ROOT/../kaburi-tts}"
if [[ ! -d "$KABURI_REPO/kaburi_tts" ]]; then
    echo "ERROR: KABURI-TTS checkout not found: $KABURI_REPO" >&2
    echo "Run scripts/setup_kaburi_env.sh first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaburi_uv_env.sh"
kaburi_uv_run KABURI_UV

AIZUCHI_PRESET="${AIZUCHI_PRESET:-normal}"
case "$AIZUCHI_PRESET" in
    normal) PRESET_VERSION="v6" ;;
    eager)  PRESET_VERSION="v7" ;;
    flood)  PRESET_VERSION="v6" ;;
    *) echo "ERROR: AIZUCHI_PRESET must be normal, eager or flood" >&2; exit 1 ;;
esac
AIZUCHI_VERSION="${AIZUCHI_VERSION:-$PRESET_VERSION}"
CORPUS_ROOT="aizuchi_${AIZUCHI_PRESET}_3000_${AIZUCHI_VERSION}"
DIALOGUES_JSONL="${DIALOGUES_JSONL:-$REPO_ROOT/data/runs/$CORPUS_ROOT/dialogue/llm_dialogues/dialogues.jsonl}"
RASTER_MODE="${RASTER_MODE:-pred}"
NUM_DIALOGUES="${NUM_DIALOGUES:-${1:-3}}"
MATCH_TOP_K="${MATCH_TOP_K:-5}"
SEED="${SEED:-0}"
STAMP="$(run_id_stamp)"

# Find where this preset was already synthesized. The TTS job's output root
# depends on which wrapper submitted it -- data/runs/<corpus>/tts for the
# aizuchi wrappers, data/runs/<BATCH_ID> when OUT_ROOT was left to default --
# so search any run directory whose name carries the corpus, rather than the
# one path this repo's own wrapper happens to use.
#
# Ranking, best first:
#   merged over a shard          the whole corpus rather than a slice
#   a non-KABURI render          a KABURI render already has KABURI's timing,
#                                so borrowing it back proves nothing
#   newer over older
#
# A slice is perfectly usable: dialogues are paired by id, not by filename.
resolve_source_dir() {
    local candidate rank best_rank=-1 best=""
    while IFS= read -r candidate; do
        # wav と JSON は <training_set>/data_stereo/ の下。training_set 直下を
        # 見ても何も無い。
        compgen -G "$candidate/data_stereo/*.json" >/dev/null 2>&1 || continue
        rank=0
        [[ "$candidate" == *"/merged/"* ]] && rank=$((rank + 2))
        [[ "$candidate" != *kaburi* ]] && rank=$((rank + 1))
        if (( rank > best_rank )); then
            best_rank=$rank
            best="$candidate"
        fi
    done < <(
        {
            ls -1dt "$REPO_ROOT"/data/runs/"$CORPUS_ROOT"/tts/merged/training_set 2>/dev/null
            ls -1dt "$REPO_ROOT"/data/runs/"$CORPUS_ROOT"/tts/shard_*/training_set 2>/dev/null
            ls -1dt "$REPO_ROOT"/data/runs/*"$CORPUS_ROOT"*/merged/training_set 2>/dev/null
            ls -1dt "$REPO_ROOT"/data/runs/*"$CORPUS_ROOT"*/shard_*/training_set 2>/dev/null
            ls -1dt "$REPO_ROOT"/data/runs/smoke/*"$CORPUS_ROOT"*/shard_*/training_set 2>/dev/null
        } || true
    )
    [[ -n "$best" ]] || return 1
    printf '%s' "$best"
}

if [[ -z "${SOURCE_DIR:-}" ]]; then
    SOURCE_DIR="$(resolve_source_dir || true)"
    if [[ -n "$SOURCE_DIR" ]]; then
        echo "[source] using $SOURCE_DIR"
        echo "[source] set SOURCE_DIR to override"
    fi
fi
if [[ -z "$SOURCE_DIR" ]]; then
    echo "ERROR: nothing rendered found for $CORPUS_ROOT." >&2
    echo >&2
    echo "SOURCE_DIR must be an already rendered training_set whose user-side" >&2
    echo "audio is kept, for THESE dialogues:" >&2
    echo "  $DIALOGUES_JSONL" >&2
    echo >&2
    echo "Every training_set on this machine (pick one over the same corpus," >&2
    echo "or set AIZUCHI_PRESET to the one that does have a render):" >&2
    find "$REPO_ROOT/data/runs" -maxdepth 6 -type d -name training_set 2>/dev/null \
        | head -n 40 >&2 || true
    echo >&2
    echo "If there is no render of this corpus at all, make one first:" >&2
    echo "  bash scripts/run_tts_smoke.sh qwen 3     # V100, needs .venv-vllm-omni" >&2
    echo "or render and splice in one go on an A100 instead, which needs no" >&2
    echo "existing corpus because KABURI supplies the whole dialogue:" >&2
    echo "  bash scripts/run_kaburi_bank_dialogues.sh 3" >&2
    exit 1
fi
if ! compgen -G "$SOURCE_DIR/data_stereo/*.json" >/dev/null \
   && ! compgen -G "$SOURCE_DIR/*.json" >/dev/null; then
    echo "ERROR: no dialogue JSON under SOURCE_DIR: $SOURCE_DIR" >&2
    echo "Expected $SOURCE_DIR/data_stereo/<stem>.json" >&2
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
if [[ ! -s "$DIALOGUES_JSONL" ]]; then
    echo "ERROR: dialogues JSONL not found: $DIALOGUES_JSONL" >&2
    exit 1
fi

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/runs/smoke/${CORPUS_ROOT}_placement_bank_${STAMP}}"
PLACEMENT_OUT="$OUT_ROOT/kaburi_placement"
BANK_OUT="$OUT_ROOT/shard_000/training_set"

# The reference pack only supplies the speaker names and the template chunk id
# that the timing model conditions on; building it decodes on CPU.
KABURI_REF_PACK="${KABURI_REF_PACK:-}"
if [[ -z "$KABURI_REF_PACK" ]]; then
    KABURI_REF_PACK="$REPO_ROOT/data/kaburi_ref_packs/placement_only"
    CLONE_OUT_DIR_MOSHI="${CLONE_OUT_DIR_MOSHI:-$REPO_ROOT/data/clone_examples/99999}" \
    REF_PACK_DEVICE=cpu \
        bash scripts/make_kaburi_ref_pack.sh "$KABURI_REF_PACK"
fi

echo "===== KABURI placement + bank (CPU) ====="
echo "repo:        $REPO_ROOT"
echo "dialogues:   $DIALOGUES_JSONL"
echo "source:      $SOURCE_DIR"
echo "bank:        $BANK_DIR"
echo "timing:      $RASTER_MODE"
echo "n:           $NUM_DIALOGUES"
echo "out:         $OUT_ROOT"
echo "started_at:  $(date -Iseconds)"
echo "========================================="

echo
echo ">>> 1/2 KABURI placement (no acoustic model) -> $PLACEMENT_OUT"
mkdir -p "$PLACEMENT_OUT"
"${KABURI_UV[@]}" scripts/generate_kaburi_tts_data.py \
    --dialogues-jsonl "$DIALOGUES_JSONL" \
    --out-dir "$PLACEMENT_OUT" \
    --kaburi-repo "$KABURI_REPO" \
    --ref-pack "$KABURI_REF_PACK" \
    --mode "$RASTER_MODE" \
    --device cpu \
    --placement-only \
    --num-dialogues "$NUM_DIALOGUES" \
    --log-every 1

echo
echo ">>> 2/2 retime and splice -> $BANK_OUT"
SPLICE_ARGS=(
    --training-dir "$SOURCE_DIR"
    --bank-dir "$BANK_DIR"
    --placement-dir "$PLACEMENT_OUT/placement"
    --out-dir "$BANK_OUT"
    --match-top-k "$MATCH_TOP_K"
    --seed "$SEED"
    --limit "$NUM_DIALOGUES"
)
[[ -n "${BANK_TEXT:-}" ]] && SPLICE_ARGS+=(--bank-text "$BANK_TEXT")
[[ -n "${BANK_TEMPERATURE:-}" ]] && SPLICE_ARGS+=(--bank-temperature "$BANK_TEMPERATURE")
uv run python scripts/splice_aizuchi_bank.py "${SPLICE_ARGS[@]}"

echo
echo "===== report ====="
# The source row is the Qwen-style baseline (backchannels appended, little
# overlap); the bank row should show KABURI's gaps pulling them in.
uv run python scripts/report_tts_smoke.py \
    --projection "${REPORT_PROJECTION:-3000}" \
    --json-out "$OUT_ROOT/report_placement_bank_${STAMP}.json" \
    --training-dir "$SOURCE_DIR" --label "source" \
    --training-dir "$BANK_OUT" --label "kaburi-$RASTER_MODE placement + bank"

echo
echo "listen: $SOURCE_DIR/data_stereo  vs  $BANK_OUT/data_stereo"
echo "swaps:  $BANK_OUT/bank_swaps.jsonl"
echo "finished_at: $(date -Iseconds)"
