#!/usr/bin/env bash
# Launch one full fine-tuning experiment with nu-dialogue/moshi-finetune.
#
# Usage:
#   bash scripts/run_nu_fullft_experiment.sh <EXP_NAME> <SRC_RUN_DIR>
#
# LoRA experiments continue to use scripts/run_experiment.sh and
# kyutai-labs/moshi-finetune. This script is only for full fine-tuning.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <EXP_NAME> <SRC_RUN_DIR>" >&2
    exit 1
fi

EXP_NAME="$1"
SRC_RUN_DIR_RAW="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="$REPO_ROOT/experiments/$EXP_NAME"
SRC_RUN_DIR="$(realpath -m "$SRC_RUN_DIR_RAW")"
SRC_TRAIN_SET="$SRC_RUN_DIR/training_set"
SRC_MANIFEST="$SRC_TRAIN_SET/synthetic_moshi_train.jsonl"

if [[ ! -f "$SRC_MANIFEST" ]]; then
    echo "ERROR: source manifest not found: $SRC_MANIFEST" >&2
    echo "  SRC_RUN_DIR must point to a run directory containing training_set/synthetic_moshi_train.jsonl" >&2
    exit 1
fi

mkdir -p "$EXP_DIR"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is not available on PATH." >&2
    exit 1
}

NU_MOSHI_FT_REPO="${NU_MOSHI_FT_REPO:-$REPO_ROOT/../moshi-finetune-nu-dialogue}"
NU_MOSHI_FT_REPO="$(realpath -m "$NU_MOSHI_FT_REPO")"
if [[ ! -d "$NU_MOSHI_FT_REPO/.git" ]]; then
    echo "[nu-fullft] cloning nu-dialogue/moshi-finetune -> $NU_MOSHI_FT_REPO"
    git clone https://github.com/nu-dialogue/moshi-finetune "$NU_MOSHI_FT_REPO"
fi

if [[ "${NU_SKIP_UV_SYNC:-0}" != "1" && ! -d "$NU_MOSHI_FT_REPO/.venv" ]]; then
    echo "[nu-fullft] installing nu-dialogue dependencies with uv"
    (cd "$NU_MOSHI_FT_REPO" && uv sync --python "${NU_PYTHON:-3.12}")
fi

python3 scripts/patch_nu_dialogue_repo.py --nu-repo "$NU_MOSHI_FT_REPO"

NPROC="${NPROC:-2}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    devs=""
    for ((i=0; i<NPROC; i++)); do
        devs+="${devs:+,}$i"
    done
    export CUDA_VISIBLE_DEVICES="$devs"
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"

RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
EXP_LOG="$EXP_DIR/run_nu_${RUN_TS}.log"
NU_OUTPUT_DIR="$EXP_DIR/checkpoints/nu_${RUN_TS}"
NU_DATA_DIR="${NU_DATA_DIR:-$REPO_ROOT/data/nu_fullft/${RUN_ID:-$(basename "$SRC_RUN_DIR")}}"
NU_DATA_DIR="$(realpath -m "$NU_DATA_DIR")"
NU_LAUNCH_CONFIG="$EXP_DIR/nu_launch_${RUN_TS}.json"

HP_LR="${HP_LR:-3e-5}"
HP_WEIGHT_DECAY="${HP_WEIGHT_DECAY:-0.1}"
HP_PCT_START="${HP_PCT_START:-0.05}"
HP_BATCH_SIZE="${HP_BATCH_SIZE:-1}"
HP_NUM_MICROBATCHES="${HP_NUM_MICROBATCHES:-8}"
HP_MAX_STEPS="${HP_MAX_STEPS:-1200}"
HP_CKPT_FREQ="${HP_CKPT_FREQ:-120}"
HP_EVAL_FREQ="${HP_EVAL_FREQ:-60}"
HP_LOG_FREQ="${HP_LOG_FREQ:-5}"
HP_MAX_NORM="${HP_MAX_NORM:-1.0}"
HP_DURATION_SEC="${HP_DURATION_SEC:-100}"
HP_SEED="${HP_SEED:-0}"

NU_TEXT_TOKENIZER_REPO="${NU_TEXT_TOKENIZER_REPO:-llm-jp/llm-jp-moshi-v1}"
NU_TEXT_TOKENIZER_NAME="${NU_TEXT_TOKENIZER_NAME:-tokenizer_spm_32k_3.model}"
NU_AUDIO_TOKENIZER_REPO="${NU_AUDIO_TOKENIZER_REPO:-llm-jp/llm-jp-moshi-v1}"
NU_AUDIO_TOKENIZER_NAME="${NU_AUDIO_TOKENIZER_NAME:-tokenizer-e351c8d8-checkpoint125.safetensors}"
NU_MOSHI_LM_REPO="${NU_MOSHI_LM_REPO:-llm-jp/llm-jp-moshi-v1}"
NU_MOSHI_LM_NAME="${NU_MOSHI_LM_NAME:-model.safetensors}"
NU_MOSHI_LM_CONFIG_NAME="${NU_MOSHI_LM_CONFIG_NAME:-moshi_lm_kwargs.json}"
NU_MODEL_DTYPE="${NU_MODEL_DTYPE:-float32}"
NU_MODEL_DIR="${NU_MODEL_DIR:-$NU_MOSHI_FT_REPO/init_models/llm-jp-moshi-v1-both_streams-${NU_MODEL_DTYPE}}"
NU_MODEL_DIR="$(realpath -m "$NU_MODEL_DIR")"
NU_DEEPSPEED_CONFIG="${NU_DEEPSPEED_CONFIG:-$REPO_ROOT/configs/deepspeed_zero3_fp16_act_ckpt.json}"
NU_DEEPSPEED_CONFIG="$(realpath -m "$NU_DEEPSPEED_CONFIG")"
NU_TOKENIZE_AUDIO_WORKERS="${NU_TOKENIZE_AUDIO_WORKERS:-$NPROC}"
NU_TOKENIZE_TEXT_WORKERS="${NU_TOKENIZE_TEXT_WORKERS:-4}"
NU_DATASET_WORKERS="${NU_DATASET_WORKERS:-16}"
NU_AUDIO_CHUNK_SIZE="${NU_AUDIO_CHUNK_SIZE:-1200}"
NU_EVAL_FRACTION="${NU_EVAL_FRACTION:-0.1}"
NU_MIN_LENGTH="${NU_MIN_LENGTH:-128}"
NU_MAX_LENGTH="${NU_MAX_LENGTH:-$(python3 - "$HP_DURATION_SEC" <<'PY'
import sys
print(int(float(sys.argv[1]) * 12.5))
PY
)}"
NU_WARMUP_STEPS="${NU_WARMUP_STEPS:-0}"

mkdir -p "$NU_DATA_DIR" "$NU_OUTPUT_DIR"

if [[ "${REFRESH_NU_DATA:-0}" == "1" ]]; then
    echo "[nu-fullft] REFRESH_NU_DATA=1: clearing $NU_DATA_DIR"
    rm -rf "$NU_DATA_DIR"
    mkdir -p "$NU_DATA_DIR"
fi

if [[ ! -f "$NU_DATA_DIR/manifest.json" ]]; then
    echo "[nu-fullft] preparing nu-dialogue raw dataset"
    python3 scripts/prepare_nu_fullft_dataset.py \
        --manifest "$SRC_MANIFEST" \
        --output-dir "$NU_DATA_DIR" \
        --eval-fraction "$NU_EVAL_FRACTION" \
        --seed "$HP_SEED"
else
    echo "[nu-fullft] reusing prepared raw dataset: $NU_DATA_DIR"
fi

tokenize_text_args=(
    --text_tokenizer_repo "$NU_TEXT_TOKENIZER_REPO"
    --text_tokenizer_name "$NU_TEXT_TOKENIZER_NAME"
    --text_padding_id "${NU_TEXT_PADDING_ID:-3}"
    --end_of_text_padding_id "${NU_END_OF_TEXT_PADDING_ID:-0}"
    --num_workers "$NU_TOKENIZE_TEXT_WORKERS"
    --resume
)
if [[ "${NU_NO_WHITESPACE_BEFORE_WORD:-1}" == "1" ]]; then
    tokenize_text_args+=(--no_whitespace_before_word)
fi

tokenize_split() {
    local split="$1"
    local audio_dir="$NU_DATA_DIR/raw/$split/audio"
    local text_dir="$NU_DATA_DIR/raw/$split/text"
    local token_audio_dir="$NU_DATA_DIR/tokenized/$split/audio"
    local token_text_dir="$NU_DATA_DIR/tokenized/$split/text"
    local parquet_prefix="$NU_DATA_DIR/processed/$split"

    if [[ ! -d "$audio_dir" || ! -d "$text_dir" ]]; then
        echo "[nu-fullft] split '$split' is absent; skipping tokenization"
        return 0
    fi

    mkdir -p "$token_audio_dir" "$token_text_dir" "$NU_DATA_DIR/processed"

    echo "[nu-fullft] tokenizing audio ($split)"
    (cd "$NU_MOSHI_FT_REPO" && \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        uv run python -m tools.tokenize_audio \
            --audio_dir "$audio_dir" \
            --output_dir "$token_audio_dir" \
            --audio_tokenizer_repo "$NU_AUDIO_TOKENIZER_REPO" \
            --audio_tokenizer_name "$NU_AUDIO_TOKENIZER_NAME" \
            --audio_chunk_size "$NU_AUDIO_CHUNK_SIZE" \
            --num_workers "$NU_TOKENIZE_AUDIO_WORKERS" \
            --resume)

    echo "[nu-fullft] tokenizing text ($split)"
    (cd "$NU_MOSHI_FT_REPO" && \
        uv run python -m tools.tokenize_text \
            --word_transcript_dir "$text_dir" \
            --output_dir "$token_text_dir" \
            "${tokenize_text_args[@]}")

    echo "[nu-fullft] writing parquet ($split)"
    rm -f "${parquet_prefix}"-*.parquet
    (cd "$NU_MOSHI_FT_REPO" && \
        uv run python -m tools.prepare_dataset \
            --tokenized_text_dir "$token_text_dir" \
            --tokenized_audio_dir "$token_audio_dir" \
            --output_prefix "$parquet_prefix")
}

if ! compgen -G "$NU_DATA_DIR/processed/train-*.parquet" >/dev/null; then
    tokenize_split train
else
    echo "[nu-fullft] reusing train parquet: $NU_DATA_DIR/processed/train-*.parquet"
fi
if [[ -d "$NU_DATA_DIR/raw/eval/audio" ]]; then
    if ! compgen -G "$NU_DATA_DIR/processed/eval-*.parquet" >/dev/null; then
        tokenize_split eval
    else
        echo "[nu-fullft] reusing eval parquet: $NU_DATA_DIR/processed/eval-*.parquet"
    fi
fi

if [[ ! -f "$NU_MODEL_DIR/model.safetensors" || ! -f "$NU_MODEL_DIR/moshi_lm_kwargs.json" ]]; then
    echo "[nu-fullft] initializing full-FT model: $NU_MODEL_DIR"
    init_args=(
        --nu-repo "$NU_MOSHI_FT_REPO"
        --save-dir "$NU_MODEL_DIR"
        --moshi-lm-repo "$NU_MOSHI_LM_REPO"
        --moshi-lm-name "$NU_MOSHI_LM_NAME"
        --model-dtype "$NU_MODEL_DTYPE"
        --extend-modules-for-user-stream
    )
    if [[ -n "$NU_MOSHI_LM_CONFIG_NAME" ]]; then
        init_args+=(--moshi-lm-config-name "$NU_MOSHI_LM_CONFIG_NAME")
    else
        init_args+=(--moshi-lm-config-name "")
    fi
    if [[ "${NU_INIT_TEXT_EMBEDDINGS:-0}" == "1" ]]; then
        init_args+=(--init-text-embeddings)
    fi
    (cd "$NU_MOSHI_FT_REPO" && uv run python "$REPO_ROOT/scripts/init_nu_moshi_for_ft.py" "${init_args[@]}")
else
    echo "[nu-fullft] reusing initialized model: $NU_MODEL_DIR"
fi

TRAIN_COUNT="$(python3 - "$NU_DATA_DIR/manifest.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["splits"]["train"]["count"])
PY
)"
NU_NUM_TRAIN_EPOCHS="${NU_NUM_TRAIN_EPOCHS:-$(python3 - "$TRAIN_COUNT" "$HP_BATCH_SIZE" "$NPROC" "$HP_NUM_MICROBATCHES" "$HP_MAX_STEPS" <<'PY'
import math, sys
train_count, per_device, nproc, accum, max_steps = map(int, sys.argv[1:])
global_batch = max(1, per_device * nproc * accum)
steps_per_epoch = max(1, math.ceil(train_count / global_batch))
print(max(1, math.ceil(max_steps / steps_per_epoch)))
PY
)}"

cat > "$NU_LAUNCH_CONFIG" <<EOF
{
  "runner": "nu-dialogue/moshi-finetune",
  "nu_repo": "$NU_MOSHI_FT_REPO",
  "exp_name": "$EXP_NAME",
  "src_run_dir": "$SRC_RUN_DIR",
  "nu_data_dir": "$NU_DATA_DIR",
  "nu_output_dir": "$NU_OUTPUT_DIR",
  "model_dir": "$NU_MODEL_DIR",
  "model_repo": "$NU_MOSHI_LM_REPO",
  "model_name": "$NU_MOSHI_LM_NAME",
  "model_config_name": "$NU_MOSHI_LM_CONFIG_NAME",
  "deepspeed_config": "$NU_DEEPSPEED_CONFIG",
  "nproc": $NPROC,
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "master_addr": "$MASTER_ADDR",
  "master_port": "$MASTER_PORT",
  "per_device_train_batch_size": $HP_BATCH_SIZE,
  "gradient_accumulation_steps": $HP_NUM_MICROBATCHES,
  "max_train_steps": $HP_MAX_STEPS,
  "num_train_epochs": $NU_NUM_TRAIN_EPOCHS,
  "learning_rate": "$HP_LR",
  "weight_decay": "$HP_WEIGHT_DECAY",
  "warmup_steps": $NU_WARMUP_STEPS,
  "max_grad_norm": "$HP_MAX_NORM",
  "max_length": $NU_MAX_LENGTH,
  "min_length": $NU_MIN_LENGTH,
  "save_steps": $HP_CKPT_FREQ,
  "eval_steps": $HP_EVAL_FREQ,
  "logging_steps": $HP_LOG_FREQ,
  "seed": $HP_SEED
}
EOF

LAUNCH_CMD=(
    uv run accelerate launch
    --num_processes "$NPROC"
    --num_machines 1
    --main_process_port "$MASTER_PORT"
    --use_deepspeed
    --deepspeed_config_file "$NU_DEEPSPEED_CONFIG"
    finetune.py
    --launcher accelerate
    --output_dir "$NU_OUTPUT_DIR"
    --train_data_files "$NU_DATA_DIR/processed/train-*.parquet"
    --model_dir "$NU_MODEL_DIR"
    --model_dtype "$NU_MODEL_DTYPE"
    --model_user_stream
    --moshi_speakers A
    --max_length "$NU_MAX_LENGTH"
    --min_length "$NU_MIN_LENGTH"
    --dataset_processing_workers "$NU_DATASET_WORKERS"
    --dataset_cache_dir "$NU_DATA_DIR/hf_cache"
    --process_group_timeout "${NU_PROCESS_GROUP_TIMEOUT:-3600}"
    --per_device_train_batch_size "$HP_BATCH_SIZE"
    --gradient_accumulation_steps "$HP_NUM_MICROBATCHES"
    --per_device_eval_batch_size "${NU_EVAL_BATCH_SIZE:-$HP_BATCH_SIZE}"
    --tempformer_learning_rate "$HP_LR"
    --depformer_learning_rate "$HP_LR"
    --weight_decay "$HP_WEIGHT_DECAY"
    --num_train_epochs "$NU_NUM_TRAIN_EPOCHS"
    --max_train_steps "$HP_MAX_STEPS"
    --num_warmup_steps "$NU_WARMUP_STEPS"
    --logging_steps "$HP_LOG_FREQ"
    --save_steps "$HP_CKPT_FREQ"
    --activation_checkpointing
    --parameters_to_finetune "${NU_PARAMETERS_TO_FINETUNE:-all}"
    --seed "$HP_SEED"
)
if compgen -G "$NU_DATA_DIR/processed/eval-*.parquet" >/dev/null; then
    LAUNCH_CMD+=(--eval_data_files "$NU_DATA_DIR/processed/eval-*.parquet")
    LAUNCH_CMD+=(--eval_steps "$HP_EVAL_FREQ")
fi
if [[ "${NU_REPORT_TO_WANDB:-0}" == "1" ]]; then
    LAUNCH_CMD+=(--report_to wandb --project_name "${NU_WANDB_PROJECT:-moshi-finetuning}")
fi

echo "[nu-fullft] launching nu-dialogue full fine-tuning"
echo "  exp:       $EXP_NAME"
echo "  repo:      $NU_MOSHI_FT_REPO"
echo "  data:      $NU_DATA_DIR"
echo "  output:    $NU_OUTPUT_DIR"
echo "  model:     $NU_MODEL_DIR"
echo "  ds:        $NU_DEEPSPEED_CONFIG"
echo "  nproc:     $NPROC"
echo "  cuda:      $CUDA_VISIBLE_DEVICES"
echo "  port:      $MASTER_PORT"
echo "  hp:        lr=$HP_LR batch=$HP_BATCH_SIZE grad_accum=$HP_NUM_MICROBATCHES steps=$HP_MAX_STEPS wd=$HP_WEIGHT_DECAY warmup=$NU_WARMUP_STEPS max_len=$NU_MAX_LENGTH"
echo "  log:       $EXP_LOG"

MLFLOW_ENABLED=0
MLFLOW_SYNC_PID=""
MLFLOW_SYNC_CMD=()
run_mlflow_sync_once() {
    local sync_timeout="${MLFLOW_SYNC_TIMEOUT:-60}"
    if [[ ${#MLFLOW_SYNC_CMD[@]} -eq 0 ]]; then
        return 0
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout "$sync_timeout" "${MLFLOW_SYNC_CMD[@]}"
    else
        "${MLFLOW_SYNC_CMD[@]}"
    fi
}

if [[ -n "${MLFLOW_EXPERIMENT_NAME:-}" || -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    MLFLOW_ENABLED=1
    echo "[nu-fullft] MLflow stdout sync enabled"
    if command -v timeout >/dev/null 2>&1; then
        MLFLOW_CHECK_CMD=(timeout "${MLFLOW_IMPORT_TIMEOUT:-120}" uv run python -c "import mlflow")
    else
        MLFLOW_CHECK_CMD=(uv run python -c "import mlflow")
    fi
    if ! "${MLFLOW_CHECK_CMD[@]}" >/dev/null 2>&1; then
        echo "[nu-fullft] installing mlflow into project uv environment"
        uv pip install mlflow
    fi
    MLFLOW_RUN_ID_FILE="$EXP_DIR/mlflow_run_${RUN_TS}.id"
    MLFLOW_SYNC_CMD=(uv run python scripts/sync_nu_mlflow_metrics.py
        --output-dir "$NU_OUTPUT_DIR"
        --log-file "$EXP_LOG"
        --config "$NU_LAUNCH_CONFIG"
        --experiment "${MLFLOW_EXPERIMENT_NAME:-job_fullft_3h}"
        --run-name "${MLFLOW_RUN_NAME:-$(basename "$EXP_NAME")}"
        --run-id-file "$MLFLOW_RUN_ID_FILE")
    if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
        MLFLOW_SYNC_CMD+=(--tracking-uri "$MLFLOW_TRACKING_URI")
    fi
    if [[ -n "${MLFLOW_ARTIFACT_ROOT:-}" ]]; then
        MLFLOW_SYNC_CMD+=(--artifact-root "$MLFLOW_ARTIFACT_ROOT")
    fi

    MLFLOW_LIVE_SYNC_INTERVAL="${MLFLOW_LIVE_SYNC_INTERVAL:-300}"
    if [[ "$MLFLOW_LIVE_SYNC_INTERVAL" != "0" ]]; then
        (
            while true; do
                run_mlflow_sync_once || echo "[nu-fullft] WARN: periodic MLflow sync failed" >&2
                sleep "$MLFLOW_LIVE_SYNC_INTERVAL"
            done
        ) &
        MLFLOW_SYNC_PID="$!"
        echo "[nu-fullft] MLflow live sync interval: ${MLFLOW_LIVE_SYNC_INTERVAL}s"
    fi
fi

set +e
(
    cd "$NU_MOSHI_FT_REPO" && \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    NO_TORCH_COMPILE="$NO_TORCH_COMPILE" \
    "${LAUNCH_CMD[@]}"
) 2>&1 | tee "$EXP_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ -n "$MLFLOW_SYNC_PID" ]]; then
    kill "$MLFLOW_SYNC_PID" >/dev/null 2>&1 || true
    wait "$MLFLOW_SYNC_PID" 2>/dev/null || true
fi

if [[ "$MLFLOW_ENABLED" == "1" ]]; then
    echo "[nu-fullft] final MLflow sync"
    run_mlflow_sync_once || echo "[nu-fullft] WARN: final MLflow sync failed" >&2
fi

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "[nu-fullft] ERROR: training failed with status $TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi
