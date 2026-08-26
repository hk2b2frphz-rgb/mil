#!/usr/bin/env bash
set -euo pipefail

# set -e aborts silently on the first failing command with no explanation --
# a GPU OOM kill, a PBS walltime kill, or a CUDA driver crash all terminate
# the python subprocess by signal with no Python traceback, so without this
# trap the log just stops dead after the subprocess's last printed line
# (e.g. "Acoustic delay: N frames" from load_models(), right before the
# first real generation step) with zero indication of why. Report the exit
# code so 137 (SIGKILL: OOM or walltime) and 139 (SIGSEGV: driver/CUDA
# crash) can be told apart from an ordinary Python exception (which would
# already have printed its own traceback above this line).
trap '
    status=$?
    echo "[fdb] ERROR: command failed (exit $status): $BASH_COMMAND" >&2
    case "$status" in
        137) echo "[fdb]   exit 137 = SIGKILL -- likely GPU/host OOM kill or PBS walltime/mem limit." >&2 ;;
        139) echo "[fdb]   exit 139 = SIGSEGV -- likely a CUDA driver/kernel crash, not a Python exception." >&2 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv >&2
' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Keep direct launches consistent with the PBS wrapper: stale inherited
# proxies are cleared before model/TTS downloads initialize HTTP clients.
source "$SCRIPT_DIR/setup_proxy.sh"

MODEL_ID="${MODEL_ID:?Set MODEL_ID to a stable label such as base, lora_h01, or full_f01}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
HF_REPO="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"
MODEL_WEIGHT="${MODEL_WEIGHT:-${MOSHI_WEIGHT:-}}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
MIMI_WEIGHT="${MIMI_WEIGHT:-}"
TOKENIZER="${TOKENIZER:-}"
FDB_SCENARIOS="${FDB_SCENARIOS:-$REPO_ROOT/eval_sets/full_duplex_ja/scenarios_expanded.jsonl}"
FDB_TASKS="${FDB_TASKS:-all}"
FDB_CASES_PER_TASK="${FDB_CASES_PER_TASK:-}"
FDB_SEEDS="${FDB_SEEDS:-0}"
FDB_TAIL_SEC="${FDB_TAIL_SEC:-8}"
FDB_TTS_BACKEND="${FDB_TTS_BACKEND:-qwen3}"
FDB_TTS_MODEL="${FDB_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
FDB_TTS_SPEAKER="${FDB_TTS_SPEAKER:-Ono_Anna}"
FDB_TTS_BACKGROUND_SPEAKER="${FDB_TTS_BACKGROUND_SPEAKER:-Uncle_Fu}"
FDB_DEVICE="${FDB_DEVICE:-cuda}"
FDB_DTYPE="${FDB_DTYPE:-float16}"
FDB_TTS_SPEED="${FDB_TTS_SPEED:-1.0}"
FDB_WHOLE_UTTERANCE="${FDB_WHOLE_UTTERANCE:-1}"
FDB_WHOLE_UTTERANCE_MAX_CHARS="${FDB_WHOLE_UTTERANCE_MAX_CHARS:-150}"
# Baseline Moshi / llm-jp were never trained to open with the fixed greeting,
# so they don't need to wait for it: set FDB_OPENING_GREETING=0 for those
# runs. The dataset dir defaults to a name that encodes this setting, so
# flipping the flag can never silently reuse a cached dataset built with the
# other setting (which would otherwise desync scenario timing from what the
# model in front of it actually expects).
FDB_OPENING_GREETING="${FDB_OPENING_GREETING:-1}"
FDB_OPENING_GREETING_GAP_SEC="${FDB_OPENING_GREETING_GAP_SEC:-0.4}"
if [[ "$FDB_OPENING_GREETING" == "1" ]]; then
    FDB_DATA_DIR_DEFAULT="$REPO_ROOT/data/full_duplex_ja_v2_greeting"
else
    FDB_DATA_DIR_DEFAULT="$REPO_ROOT/data/full_duplex_ja_v2_nogreeting"
fi
FDB_DATA_DIR="${FDB_DATA_DIR:-$FDB_DATA_DIR_DEFAULT}"
FDB_OUT_DIR="${FDB_OUT_DIR:-$REPO_ROOT/eval_runs/full_duplex/$RUN_ID}"
REFRESH_FDB_DATA="${REFRESH_FDB_DATA:-0}"
FDB_REQUIRE_OVERLAP="${FDB_REQUIRE_OVERLAP:-1}"
FDB_ASR_MODEL="${FDB_ASR_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
FDB_WITH_UTMOS="${FDB_WITH_UTMOS:-1}"
# In the 4-GPU batch PBS job this is set to all allocated GPUs. It is used
# only for the one-time, cache-miss input-TTS build; each model inference
# process remains pinned to its own single GPU.
FDB_TTS_BUILD_GPUS="${FDB_TTS_BUILD_GPUS:-}"
HALF="${HALF:-1}"

if [[ -n "$FDB_CASES_PER_TASK" ]] && ! [[ "$FDB_CASES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FDB_CASES_PER_TASK must be a positive integer: $FDB_CASES_PER_TASK" >&2
    exit 1
fi

# Disable Triton / torch.compile -- V100 lacks support for many Triton kernels.
export TORCHINDUCTOR_DISABLE=1
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export TORCH_COMPILE_DISABLE=1

# Unbuffer Python stdout. This script pipes stdout through `tee` (see the
# exec redirect below), so Python block-buffers stdout (~4-8 KB) and the
# per-trial "[fdb] N/total ..." progress prints don't appear until the
# buffer fills -- often many trials later, or not until the process exits.
# The logging module writes to stderr and flushes per line, which is why
# "Loading ..." / "Acoustic delay: N frames" show immediately but progress
# then appears to stall. Unbuffering makes progress visible in real time.
export PYTHONUNBUFFERED=1
export FDB_GIT_COMMIT="${FDB_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

# If MODEL_CONFIG wasn't given, auto-detect a moshi_lm_kwargs.json sitting next
# to MODEL_WEIGHT. scripts/export_fullft_checkpoint.py (via clean_moshi.py)
# always writes model.safetensors and moshi_lm_kwargs.json into the same export
# dir, so full-FT models can be evaluated by passing only MODEL_WEIGHT. Merged
# LoRA models have no such sibling and correctly fall back to the HF default
# config (same architecture as the base).
if [[ -z "$MODEL_CONFIG" && -n "$MODEL_WEIGHT" ]]; then
    AUTO_MODEL_CONFIG="$(dirname "$MODEL_WEIGHT")/moshi_lm_kwargs.json"
    if [[ -f "$AUTO_MODEL_CONFIG" ]]; then
        MODEL_CONFIG="$AUTO_MODEL_CONFIG"
        echo "[fdb] auto-detected MODEL_CONFIG next to MODEL_WEIGHT: $MODEL_CONFIG"
    fi
fi

if [[ -n "$MODEL_WEIGHT" && ! -f "$MODEL_WEIGHT" ]]; then
    echo "ERROR: MODEL_WEIGHT does not exist: $MODEL_WEIGHT" >&2
    exit 1
fi
if [[ -n "$MODEL_CONFIG" && ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: MODEL_CONFIG does not exist: $MODEL_CONFIG" >&2
    exit 1
fi

mkdir -p "$FDB_OUT_DIR"
rm -f "$FDB_OUT_DIR/azure_judge_input.jsonl"
exec > >(tee "$FDB_OUT_DIR/run.log") 2>&1

echo "===== Full-Duplex-Bench-JA ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "hf_repo:      $HF_REPO"
echo "model_weight: ${MODEL_WEIGHT:-<hf default>}"
echo "model_config: ${MODEL_CONFIG:-<hf default>}"
echo "tasks:        $FDB_TASKS"
echo "cases/task:   ${FDB_CASES_PER_TASK:-all}"
echo "seeds:        $FDB_SEEDS"
echo "data:         $FDB_DATA_DIR"
echo "output:       $FDB_OUT_DIR"
echo "dtype:        $([[ "$HALF" == "1" ]] && echo float16 || echo bfloat16)"
echo "upstream:     Full-Duplex-Bench@3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
echo "git_commit:   $FDB_GIT_COMMIT"
echo "started_at:   $(date -Iseconds)"
echo "================================"

# Parallel batch workers can share this dataset. Validate every rendering input
# before reuse; an old manifest or a different TTS/scenario setting must not
# silently become the comparison input for another checkpoint.
dataset_matches() {
    [[ -f "$FDB_DATA_DIR/manifest.json" ]] || return 1
    uv run python eval/verify_full_duplex_ja_dataset.py \
        --data-dir "$FDB_DATA_DIR" \
        --scenarios "$FDB_SCENARIOS" \
        --tts-backend "$FDB_TTS_BACKEND" \
        --tts-model "$FDB_TTS_MODEL" \
        --tts-speaker "$FDB_TTS_SPEAKER" \
        --tts-background-speaker "$FDB_TTS_BACKGROUND_SPEAKER" \
        --tts-speed "$FDB_TTS_SPEED" \
        --whole-utterance "$FDB_WHOLE_UTTERANCE" \
        --whole-utterance-max-chars "$FDB_WHOLE_UTTERANCE_MAX_CHARS" \
        --opening-greeting "$FDB_OPENING_GREETING" \
        --opening-greeting-gap-sec "$FDB_OPENING_GREETING_GAP_SEC"
}

NEED_FDB_DATA_BUILD=0
if [[ "$REFRESH_FDB_DATA" == "1" ]] || ! dataset_matches; then
    NEED_FDB_DATA_BUILD=1
fi

# Serialize a dataset build and also serialize *different* greeting variants.
# Their paths differ, so a per-directory lock alone would let both variants
# start a multi-GPU TTS build at the same time in a mixed baseline batch.
FDB_DATA_LOCK_FD=""
FDB_GLOBAL_BUILD_LOCK_FD=""
if [[ "$NEED_FDB_DATA_BUILD" == "1" ]]; then
    if [[ "${FDB_BATCH_PARALLELISM:-1}" -gt 1 ]]; then
        command -v flock >/dev/null 2>&1 || {
            echo "ERROR: flock is required for parallel dataset construction." >&2
            exit 1
        }
        mkdir -p "$(dirname "$FDB_DATA_DIR")"
        exec {FDB_DATA_LOCK_FD}>"${FDB_DATA_DIR}.build.lock"
        exec {FDB_GLOBAL_BUILD_LOCK_FD}>"$(dirname "$FDB_DATA_DIR")/.full_duplex_ja.build.global.lock"
        echo "[fdb] waiting for dataset build lock: ${FDB_DATA_DIR}.build.lock"
        flock "$FDB_DATA_LOCK_FD"
        echo "[fdb] waiting for global Full-Duplex-Bench dataset build lock"
        flock "$FDB_GLOBAL_BUILD_LOCK_FD"
    fi
fi

if [[ "$REFRESH_FDB_DATA" != "1" ]] && dataset_matches; then
    NEED_FDB_DATA_BUILD=0
fi

if [[ "$NEED_FDB_DATA_BUILD" == "1" ]]; then
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
    IFS=',' read -r -a tts_build_gpus <<< "$FDB_TTS_BUILD_GPUS"
    if [[ -n "$FDB_TTS_BUILD_GPUS" && "${#tts_build_gpus[@]}" -gt 1 ]]; then
        shard_count="${#tts_build_gpus[@]}"
        shard_root="${FDB_DATA_DIR}.tts_shards_${RUN_ID}_$$"
        rm -rf "$shard_root"
        mkdir -p "$shard_root"
        echo "[fdb] building input TTS across $shard_count GPUs: ${tts_build_gpus[*]}"

        # Populate the shared greeting cache before shard workers start. This
        # makes every shard reserve precisely the same lead-in duration.
        if [[ "$FDB_OPENING_GREETING" == "1" ]]; then
            CUDA_VISIBLE_DEVICES="${tts_build_gpus[0]}" "${build_args[@]}" \
                --out-dir "$shard_root/greeting_warmup" --num-shards 999999 --shard-index 0
            rm -rf "$shard_root/greeting_warmup"
        fi

        shard_pids=()
        for shard_index in "${!tts_build_gpus[@]}"; do
            CUDA_VISIBLE_DEVICES="${tts_build_gpus[$shard_index]}" "${build_args[@]}" \
                --out-dir "$shard_root/shard_$shard_index" \
                --num-shards "$shard_count" --shard-index "$shard_index" &
            shard_pids+=("$!")
        done
        shard_failed=0
        for pid in "${shard_pids[@]}"; do
            wait "$pid" || shard_failed=1
        done
        if [[ "$shard_failed" == "1" ]]; then
            rm -rf "$shard_root"
            echo "ERROR: one or more parallel input-TTS shards failed." >&2
            exit 1
        fi
        merge_args=(
            uv run python eval/merge_full_duplex_ja_dataset_shards.py
            --scenarios "$FDB_SCENARIOS" --out-dir "$FDB_DATA_DIR" --overwrite
        )
        for shard_index in "${!tts_build_gpus[@]}"; do
            merge_args+=(--shard-dir "$shard_root/shard_$shard_index")
        done
        "${merge_args[@]}"
        rm -rf "$shard_root"
    else
        "${build_args[@]}"
    fi
else
    echo "[fdb] reusing Japanese dataset: $FDB_DATA_DIR"
fi
if [[ -n "$FDB_DATA_LOCK_FD" ]]; then
    flock -u "$FDB_DATA_LOCK_FD"
    exec {FDB_DATA_LOCK_FD}>&-
fi
if [[ -n "$FDB_GLOBAL_BUILD_LOCK_FD" ]]; then
    flock -u "$FDB_GLOBAL_BUILD_LOCK_FD"
    exec {FDB_GLOBAL_BUILD_LOCK_FD}>&-
fi

inference_args=(
    uv run python eval/run_full_duplex_bench.py
    --dataset-dir "$FDB_DATA_DIR"
    --out-dir "$FDB_OUT_DIR/inference"
    --model-id "$MODEL_ID"
    --hf-repo "$HF_REPO"
    --tasks "$FDB_TASKS"
    --seeds "$FDB_SEEDS"
    --tail-sec "$FDB_TAIL_SEC"
    --require-v15-profile
    --overwrite
)
if [[ -n "$FDB_CASES_PER_TASK" ]]; then inference_args+=(--cases-per-task "$FDB_CASES_PER_TASK"); fi
if [[ "$HALF" == "1" ]]; then inference_args+=(--half); fi
if [[ -n "$MODEL_WEIGHT" ]]; then inference_args+=(--moshi-weight "$MODEL_WEIGHT"); fi
if [[ -n "$MODEL_CONFIG" ]]; then inference_args+=(--config "$MODEL_CONFIG"); fi
if [[ -n "$MIMI_WEIGHT" ]]; then inference_args+=(--mimi-weight "$MIMI_WEIGHT"); fi
if [[ -n "$TOKENIZER" ]]; then inference_args+=(--tokenizer "$TOKENIZER"); fi
if [[ -n "${TEMP:-}" ]]; then inference_args+=(--temp "$TEMP"); fi
if [[ -n "${TEMP_TEXT:-}" ]]; then inference_args+=(--temp-text "$TEMP_TEXT"); fi
if [[ -n "${CFG_COEF:-}" ]]; then inference_args+=(--cfg-coef "$CFG_COEF"); fi

"${inference_args[@]}"

# The Japanese adaptation evaluates the time-aligned Japanese text pieces
# emitted by Moshi itself.  Do not overwrite output.json/clean_output.json
# with the English Parakeet-TDT ASR: a silent (valid) model response has no
# ASR tokens, which makes TDT timestamp extraction fail before metrics can be
# written.  The inference runner has already written the required JSON files.
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
if [[ "$FDB_REQUIRE_OVERLAP" == "1" ]]; then
    evaluate_args+=(--require-overlap)
fi
evaluation_status=0
if "${evaluate_args[@]}"; then
    :
else
    evaluation_status=$?
    echo "[fdb] WARNING: deterministic evaluation was partial (exit $evaluation_status); packing successful trials for Azure." >&2
fi

# per_case.jsonl contains only trials accepted into summary.json. Pack it even
# when the evaluator reports a partial result so Azure judging can proceed for
# the successful subset.
uv run python eval/pack_full_duplex_azure.py \
    --per-case "$FDB_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$FDB_OUT_DIR/azure_judge_input.jsonl"

adaptation_args=(
    uv run python eval/evaluate_full_duplex_adaptation.py
    --run-dir "$FDB_OUT_DIR/inference"
    --out "$FDB_OUT_DIR/benchmark_results/acoustic_adaptation.json"
)
if [[ "$FDB_WITH_UTMOS" == "1" ]]; then
    adaptation_args+=(--with-utmos)
fi
"${adaptation_args[@]}"

echo "[fdb] benchmark summary: $FDB_OUT_DIR/benchmark_results/summary.json"
echo "[fdb] Azure input only:  $FDB_OUT_DIR/azure_judge_input.jsonl"
echo "[fdb] No OpenAI/Azure API was called by this server run."
echo "finished_at: $(date -Iseconds)"
exit "$evaluation_status"
