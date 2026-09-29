#!/usr/bin/env bash
set -euo pipefail

# Qwen3-TTS diversity probe: one clone reference, many short backchannels,
# then a report on how much the output actually moves.
#
#   bash scripts/run_qwen3_tts_diversity.sh                # 20 texts x 10 = 200
#   REPEATS=20 bash scripts/run_qwen3_tts_diversity.sh     # 20 texts x 20 = 400
#   TEXTS_FILE=my.txt bash scripts/run_qwen3_tts_diversity.sh
#
# One string, many draws, across temperatures -- how far the SAME input can be
# pushed, with the text held still so nothing is confounded by content:
#
#   TEXTS=<word> REPEATS=50 TEMPERATURE_SWEEP=0.7,1.0,1.3 \
#     bash scripts/run_qwen3_tts_diversity.sh
#
# Re-measuring needs no GPU and no vLLM, and must not re-synthesize: rebuilding
# the audio would give a different bank to compare against.
#
#   uv run python scripts/qwen3_tts_diversity_probe.py --out-dir <dir> --reanalyze
#
# The question behind it: a bank of pre-synthesized backchannels is only worth
# building if the bank has variety in it. This measures the two places variety
# could come from -- resampling the same string, and changing how the string is
# spelled (uun / uuun / uun with a small u) -- and prints the ratio between
# them. See scripts/qwen3_tts_diversity_probe.py for the metric definitions.
#
# Same environment as the Qwen3 arm of the TTS smokes: the vLLM-Omni venv
# (scripts/setup_vllm_omni_v100_env.sh) on a V100. The moshi clone reference is
# resolved from data/clone_examples/99999, the voice every Qwen3 corpus in this
# repo already uses, so the numbers describe the voice actually in production.
#
# Overrides: REPEATS, LIMIT, TEXTS, TEXTS_FILE, CLONE_OUT_DIR_MOSHI, REF_RANK,
# CLONE_MODE, OUT_DIR, BATCH_ID, TTS_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE,
# TEMPERATURE_SWEEP, TOP_P, TOP_K, SEED, CUDA_VISIBLE_DEVICES, VLLM_PYTHON.
#
# TEMPERATURE_SWEEP runs the whole plan once per temperature on one engine
# load, so REPEATS is per temperature: one word at REPEATS=50 with three
# temperatures is 150 samples, grouped as three.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# An interactive PBS session already has PBS_O_WORKDIR pointing at wherever the
# session started (home, typically); everything below wants the repo.
export PBS_O_WORKDIR="$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/run_id_utils.sh"
# huggingface.co has to be reachable from the compute node to pull the model.
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/setup_proxy.sh"

REPEATS="${REPEATS:-10}"
LIMIT="${LIMIT:-0}"
CLONE_OUT_DIR_MOSHI="${CLONE_OUT_DIR_MOSHI:-$REPO_ROOT/data/clone_examples/99999}"
REF_RANK="${REF_RANK:-1}"
CLONE_MODE="${CLONE_MODE:-in-context}"
QWEN_CLONE_MODEL="${QWEN_CLONE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
VLLM_STAGE_CONFIG="${VLLM_STAGE_CONFIG:-$REPO_ROOT/configs/qwen3_tts_v100_batch16.yaml}"
VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv-vllm-omni/bin/python}"
TTS_BATCH_SIZE="${TTS_BATCH_SIZE:-16}"
# A backchannel is under a second; production's 2048 only buys runaway room.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-cuda12.6_cudnn9.7.1_nccl2.24.3}"

BATCH_ID="${BATCH_ID:-qwen3_diversity_$(run_id_stamp)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/runs/diversity/$BATCH_ID}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: vLLM Python not found: $VLLM_PYTHON" >&2
    echo "Run scripts/setup_vllm_omni_v100_env.sh first." >&2
    exit 1
fi
if [[ ! -s "$VLLM_STAGE_CONFIG" ]]; then
    echo "ERROR: vLLM stage config not found: $VLLM_STAGE_CONFIG" >&2
    exit 1
fi
if [[ ! -d "$CLONE_OUT_DIR_MOSHI" ]]; then
    echo "ERROR: moshi clone reference folder not found: $CLONE_OUT_DIR_MOSHI" >&2
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$module_init" ]]; then
            # shellcheck source=/dev/null
            source "$module_init"
            break
        fi
    done
fi
if command -v module >/dev/null 2>&1; then
    module load "$VLLM_CUDA_MODULE"
else
    echo "WARNING: environment modules unavailable; not loading $VLLM_CUDA_MODULE" >&2
fi

if ! "$VLLM_PYTHON" scripts/patch_vllm_omni_qwen3_tts_v100.py --check; then
    echo "ERROR: the vLLM-Omni Qwen3-TTS V100 patch is missing or stale." >&2
    echo "Re-run scripts/setup_vllm_omni_v100_env.sh." >&2
    exit 1
fi

# Japanese transcripts must never travel through qsub -v or a shell argument
# list; resolve_clone_refs.py is the one place that reads them.
REF_WAV="$(uv run python scripts/resolve_clone_refs.py \
    --clone-out-dir "$CLONE_OUT_DIR_MOSHI" --rank "$REF_RANK" \
    --mode "$CLONE_MODE" --field wav)"
REF_TEXT=""
PROBE_CLONE_FLAGS=()
if [[ "$CLONE_MODE" == "in-context" ]]; then
    REF_TEXT="$(uv run python scripts/resolve_clone_refs.py \
        --clone-out-dir "$CLONE_OUT_DIR_MOSHI" --rank "$REF_RANK" \
        --mode "$CLONE_MODE" --field text)"
else
    PROBE_CLONE_FLAGS+=(--x-vector-only)
fi
if [[ ! -s "$REF_WAV" ]]; then
    echo "ERROR: clone reference wav not found: $REF_WAV" >&2
    exit 1
fi

PROBE_ARGS=(
    --out-dir "$OUT_DIR"
    --ref-audio "$REF_WAV"
    --repeats "$REPEATS"
    --limit "$LIMIT"
    --model "$QWEN_CLONE_MODEL"
    --stage-config "$VLLM_STAGE_CONFIG"
    --batch-size "$TTS_BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
)
[[ -n "$REF_TEXT" ]] && PROBE_ARGS+=(--ref-text "$REF_TEXT")
[[ -n "${TEXTS_FILE:-}" ]] && PROBE_ARGS+=(--texts-file "$TEXTS_FILE")
[[ -n "${TEXTS:-}" ]] && PROBE_ARGS+=(--texts "$TEXTS")
[[ -n "${TEMPERATURE:-}" ]] && PROBE_ARGS+=(--temperature "$TEMPERATURE")
[[ -n "${TEMPERATURE_SWEEP:-}" ]] && PROBE_ARGS+=(--temperature-sweep "$TEMPERATURE_SWEEP")
[[ -n "${TOP_P:-}" ]] && PROBE_ARGS+=(--top-p "$TOP_P")
[[ -n "${TOP_K:-}" ]] && PROBE_ARGS+=(--top-k "$TOP_K")
[[ -n "${SEED:-}" ]] && PROBE_ARGS+=(--seed "$SEED")

echo "===== Qwen3-TTS diversity probe ====="
echo "repo:       $REPO_ROOT"
echo "model:      $QWEN_CLONE_MODEL"
echo "reference:  $REF_WAV ($CLONE_MODE, rank $REF_RANK)"
echo "repeats:    $REPEATS  limit: $LIMIT"
echo "texts:      ${TEXTS:-${TEXTS_FILE:-(default 20)}}"
echo "temps:      ${TEMPERATURE_SWEEP:-${TEMPERATURE:-(engine default)}}"
echo "out_dir:    $OUT_DIR"
echo "gpu:        $CUDA_VISIBLE_DEVICES"
echo "started_at: $(date -Iseconds)"
echo "===================================="

mkdir -p "$OUT_DIR"
"$VLLM_PYTHON" scripts/qwen3_tts_diversity_probe.py \
    "${PROBE_ARGS[@]}" ${PROBE_CLONE_FLAGS[@]+"${PROBE_CLONE_FLAGS[@]}"}

echo
echo "finished_at: $(date -Iseconds)"
