#!/usr/bin/env bash
set -euo pipefail

# Full-Duplex-Bench-JA eval for the local cascade (ASR->LLM->TTS) baseline,
# written as a drop-in sibling of scripts/run_full_duplex_eval.sh: it takes the
# same MODEL_ID / RUN_ID / FDB_OUT_DIR interface and produces the identical
# $FDB_OUT_DIR/benchmark_results/summary.json layout, so a cascade row can be
# dropped straight into scripts/run_full_duplex_eval_batch.sh and land in the
# same combined_summary.json side by side with Moshi models. Unlike
# scripts/run_local_baseline_cascade_eval.pbs this runs ONLY the Full-Duplex
# pipeline (no domain-C support) -- the batch is a full-duplex comparison.
#
# This job never calls OpenAI/Azure APIs (same rule as the Moshi eval script) --
# it only produces the deterministic summary and the azure_judge_input.jsonl to
# be judged later on the local PC. See docs/full_duplex_evaluation.md,
# docs/local_baselines.md and eval/run_local_baseline_full_duplex.py.
#
# Cascade-specific overrides (all optional; defaults match
# scripts/run_local_baseline_cascade_eval.pbs):
#   CASCADE_ASR_MODEL=large-v3            faster-whisper model size.
#   CASCADE_ASR_DEVICE=cuda
#   CASCADE_ASR_COMPUTE_TYPE=float16
#   CASCADE_LLM_MODEL=google/gemma-2-2b-it
#   CASCADE_LLM_MAX_NEW_TOKENS=200
#   CASCADE_TTS_BACKEND=qwen3
#   CASCADE_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
#   CASCADE_TTS_SPEAKER=Serena
#   CASCADE_DEVICE=cuda
#   CASCADE_DTYPE=float16

# See scripts/run_full_duplex_eval.sh for why this ERR trap exists (a signal
# kill -- OOM/walltime/CUDA crash -- otherwise stops the log dead with no
# explanation).
trap '
    status=$?
    echo "[cascade-fdb] ERROR: command failed (exit $status): $BASH_COMMAND" >&2
    case "$status" in
        137) echo "[cascade-fdb]   exit 137 = SIGKILL -- likely GPU/host OOM kill or PBS walltime/mem limit." >&2 ;;
        139) echo "[cascade-fdb]   exit 139 = SIGSEGV -- likely a CUDA driver/kernel crash, not a Python exception." >&2 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv >&2
' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# This script is also run directly (not only through its PBS wrapper).  Make
# sure a login-node-only proxy inherited from that shell never reaches ASR,
# TTS, or the isolated Gemma worker.
source "$SCRIPT_DIR/setup_proxy.sh"

MODEL_ID="${MODEL_ID:?Set MODEL_ID to a stable label such as cascade_gemma2b}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"

# Dataset build knobs -- kept identical to scripts/run_full_duplex_eval.sh so a
# cascade row reuses the exact same cached dataset dir a Moshi row builds
# (byte-identical input.wav timing), which is the whole point of comparing them.
FDB_SCENARIOS="${FDB_SCENARIOS:-$REPO_ROOT/eval_sets/full_duplex_ja/scenarios_expanded.jsonl}"
FDB_TASKS="${FDB_TASKS:-all}"
FDB_CASES_PER_TASK="${FDB_CASES_PER_TASK:-}"
FDB_SEEDS="${FDB_SEEDS:-0}"
FDB_TTS_BACKEND="${FDB_TTS_BACKEND:-qwen3}"
FDB_TTS_MODEL="${FDB_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
FDB_TTS_SPEAKER="${FDB_TTS_SPEAKER:-Ono_Anna}"
FDB_TTS_BACKGROUND_SPEAKER="${FDB_TTS_BACKGROUND_SPEAKER:-Uncle_Fu}"
FDB_DEVICE="${FDB_DEVICE:-cuda}"
FDB_DTYPE="${FDB_DTYPE:-float16}"
FDB_ASR_MODEL="${FDB_ASR_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
FDB_TTS_SPEED="${FDB_TTS_SPEED:-1.0}"
FDB_WHOLE_UTTERANCE="${FDB_WHOLE_UTTERANCE:-1}"
FDB_WHOLE_UTTERANCE_MAX_CHARS="${FDB_WHOLE_UTTERANCE_MAX_CHARS:-150}"
# A cascade baseline was never trained to say Moshi's fixed opening greeting and
# has no streaming behavior to reserve lead-in time for, so default to the
# no-greeting dataset. FDB_OPENING_GREETING can still force the greeting dataset
# if you deliberately want the timing to line up with a greeting-trained Moshi
# run (the batch script sets this to 0 automatically for weightless rows).
FDB_OPENING_GREETING="${FDB_OPENING_GREETING:-0}"
FDB_OPENING_GREETING_GAP_SEC="${FDB_OPENING_GREETING_GAP_SEC:-0.4}"
if [[ "$FDB_OPENING_GREETING" == "1" ]]; then
    FDB_DATA_DIR_DEFAULT="$REPO_ROOT/data/full_duplex_ja_v2_greeting"
else
    FDB_DATA_DIR_DEFAULT="$REPO_ROOT/data/full_duplex_ja_v2_nogreeting"
fi
FDB_DATA_DIR="${FDB_DATA_DIR:-$FDB_DATA_DIR_DEFAULT}"
FDB_OUT_DIR="${FDB_OUT_DIR:-$REPO_ROOT/eval_runs/full_duplex/$RUN_ID}"
REFRESH_FDB_DATA="${REFRESH_FDB_DATA:-0}"

# Cascade system knobs (defaults match run_local_baseline_cascade_eval.pbs).
CASCADE_ASR_MODEL="${CASCADE_ASR_MODEL:-large-v3}"
CASCADE_ASR_DEVICE="${CASCADE_ASR_DEVICE:-cuda}"
CASCADE_ASR_COMPUTE_TYPE="${CASCADE_ASR_COMPUTE_TYPE:-float16}"
CASCADE_LLM_MODEL="${CASCADE_LLM_MODEL:-google/gemma-2-2b-it}"
CASCADE_LLM_MAX_NEW_TOKENS="${CASCADE_LLM_MAX_NEW_TOKENS:-200}"
CASCADE_TTS_BACKEND="${CASCADE_TTS_BACKEND:-qwen3}"
CASCADE_TTS_MODEL="${CASCADE_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
CASCADE_TTS_SPEAKER="${CASCADE_TTS_SPEAKER:-Serena}"
CASCADE_DEVICE="${CASCADE_DEVICE:-cuda}"
CASCADE_DTYPE="${CASCADE_DTYPE:-float16}"

if [[ -n "$FDB_CASES_PER_TASK" ]] && ! [[ "$FDB_CASES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FDB_CASES_PER_TASK must be a positive integer: $FDB_CASES_PER_TASK" >&2
    exit 1
fi

# Disable Triton / torch.compile -- V100 lacks support for many Triton kernels.
export TORCHINDUCTOR_DISABLE=1
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export TORCH_COMPILE_DISABLE=1

# Unbuffer Python stdout so per-trial progress prints appear in real time
# through the `tee` pipe below (see the same note in run_full_duplex_eval.sh).
export PYTHONUNBUFFERED=1
export FDB_GIT_COMMIT="${FDB_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

mkdir -p "$FDB_OUT_DIR"
rm -f "$FDB_OUT_DIR/azure_judge_input.jsonl"
exec > >(tee "$FDB_OUT_DIR/run.log") 2>&1

echo "===== Full-Duplex-Bench-JA (cascade) ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "asr:          $CASCADE_ASR_MODEL ($CASCADE_ASR_COMPUTE_TYPE, $CASCADE_ASR_DEVICE)"
echo "llm:          $CASCADE_LLM_MODEL"
echo "tts:          $CASCADE_TTS_BACKEND / $CASCADE_TTS_SPEAKER"
echo "tasks:        $FDB_TASKS"
echo "cases/task:   ${FDB_CASES_PER_TASK:-all}"
echo "seeds:        $FDB_SEEDS"
echo "data:         $FDB_DATA_DIR"
echo "output:       $FDB_OUT_DIR"
echo "greeting:     $FDB_OPENING_GREETING"
echo "upstream:     Full-Duplex-Bench@3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
echo "git_commit:   $FDB_GIT_COMMIT"
echo "started_at:   $(date -Iseconds)"
echo "=========================================="

# Match the Moshi runner's shared-dataset lock. Parallel model workers must
# never build or overwrite the same input set concurrently.
FDB_DATA_LOCK_FD=""
if [[ "$REFRESH_FDB_DATA" == "1" || ! -f "$FDB_DATA_DIR/manifest.json" ]]; then
    if [[ "${FDB_BATCH_PARALLELISM:-1}" -gt 1 ]]; then
        command -v flock >/dev/null 2>&1 || {
            echo "ERROR: flock is required for parallel dataset construction." >&2
            exit 1
        }
        mkdir -p "$(dirname "$FDB_DATA_DIR")"
        exec {FDB_DATA_LOCK_FD}>"${FDB_DATA_DIR}.build.lock"
        echo "[cascade-fdb] waiting for dataset build lock: ${FDB_DATA_DIR}.build.lock"
        flock "$FDB_DATA_LOCK_FD"
    fi
fi

if [[ "$REFRESH_FDB_DATA" == "1" || ! -f "$FDB_DATA_DIR/manifest.json" ]]; then
    build_args=(
        uv run python eval/build_full_duplex_ja_dataset.py
        --scenarios "$FDB_SCENARIOS"
        --out-dir "$FDB_DATA_DIR"
        --tts-backend "$FDB_TTS_BACKEND"
        --tts-model "$FDB_TTS_MODEL"
        --tts-speaker "$FDB_TTS_SPEAKER"
        --tts-background-speaker "$FDB_TTS_BACKGROUND_SPEAKER"
        --device "$FDB_DEVICE"
        --dtype "$FDB_DTYPE"
        --tts-speed "$FDB_TTS_SPEED"
        --whole-utterance-max-chars "$FDB_WHOLE_UTTERANCE_MAX_CHARS"
        --opening-greeting-gap-sec "$FDB_OPENING_GREETING_GAP_SEC"
        --overwrite
    )
    if [[ "$FDB_WHOLE_UTTERANCE" == "1" ]]; then build_args+=(--whole-utterance); fi
    if [[ "$FDB_OPENING_GREETING" != "1" ]]; then build_args+=(--no-opening-greeting); fi
    if [[ -n "${FDB_OPENING_GREETING_CACHE_DIR:-}" ]]; then
        build_args+=(--opening-greeting-cache-dir "$FDB_OPENING_GREETING_CACHE_DIR")
    fi
    "${build_args[@]}"
else
    echo "[cascade-fdb] reusing Japanese dataset: $FDB_DATA_DIR"
fi
if [[ -n "$FDB_DATA_LOCK_FD" ]]; then
    flock -u "$FDB_DATA_LOCK_FD"
    exec {FDB_DATA_LOCK_FD}>&-
fi

baseline_args=(
    uv run python eval/run_local_baseline_full_duplex.py
    --system cascade
    --dataset-dir "$FDB_DATA_DIR"
    --out-dir "$FDB_OUT_DIR/inference"
    --model-id "$MODEL_ID"
    --tasks "$FDB_TASKS"
    --seeds "$FDB_SEEDS"
    --require-v15-profile
    --asr-model "$CASCADE_ASR_MODEL"
    --asr-device "$CASCADE_ASR_DEVICE"
    --asr-compute-type "$CASCADE_ASR_COMPUTE_TYPE"
    --llm-model "$CASCADE_LLM_MODEL"
    --llm-max-new-tokens "$CASCADE_LLM_MAX_NEW_TOKENS"
    --tts-backend "$CASCADE_TTS_BACKEND"
    --tts-model "$CASCADE_TTS_MODEL"
    --tts-speaker "$CASCADE_TTS_SPEAKER"
    --device "$CASCADE_DEVICE"
    --dtype "$CASCADE_DTYPE"
    --overwrite
)
if [[ -n "$FDB_CASES_PER_TASK" ]]; then baseline_args+=(--cases-per-task "$FDB_CASES_PER_TASK"); fi
"${baseline_args[@]}"

# The local baseline runner re-recognizes synthesized Japanese response audio
# with faster-whisper and writes audio-derived chunks itself. Preserve those
# artifacts rather than replacing them with English Parakeet-TDT alignment.
uv run python eval/full_duplex_official_timing.py --run-dir "$FDB_OUT_DIR/inference"

FDB_BACKCHANNEL_GT="${FDB_BACKCHANNEL_GT:-$REPO_ROOT/eval_sets/full_duplex_ja/backchannel_gt.json}"
evaluate_args=(
    uv run python eval/evaluate_full_duplex_ja.py
    --run-dir "$FDB_OUT_DIR/inference"
    --out-dir "$FDB_OUT_DIR/benchmark_results"
)
if [[ -f "$FDB_BACKCHANNEL_GT" ]]; then
    evaluate_args+=(--backchannel-gt "$FDB_BACKCHANNEL_GT")
fi
evaluation_status=0
if "${evaluate_args[@]}"; then
    :
else
    evaluation_status=$?
    echo "[cascade-fdb] WARNING: deterministic evaluation was partial (exit $evaluation_status); packing successful trials for Azure." >&2
fi

uv run python eval/pack_full_duplex_azure.py \
    --per-case "$FDB_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$FDB_OUT_DIR/azure_judge_input.jsonl"

echo "[cascade-fdb] benchmark summary: $FDB_OUT_DIR/benchmark_results/summary.json"
echo "[cascade-fdb] Azure input only:  $FDB_OUT_DIR/azure_judge_input.jsonl"
echo "[cascade-fdb] No OpenAI/Azure API was called by this server run."
echo "finished_at: $(date -Iseconds)"
exit "$evaluation_status"
