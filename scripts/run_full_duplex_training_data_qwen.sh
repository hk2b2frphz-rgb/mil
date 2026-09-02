#!/usr/bin/env bash
# Drop-in replacement for run_full_duplex_training_data.sh's "dialogues" and
# "audio" steps: dialogues come from a local vLLM-served Qwen3.6-27B instead
# of gemma (matching scripts/run_dialogues_qwen_10000.pbs), and audio uses
# Qwen3-TTS whole-utterance synthesis (matching
# scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs) instead of one TTS
# call per turn. use_cases/enrich are unchanged.
#
# Accepts the same environment contract as run_full_duplex_training_data.sh
# (RUN_ID, OUT_ROOT, NUM_CASES, STEPS, DIALOGUE_RESUME, AUDIO_RESUME, ...) so
# it can be swapped in wherever that script is called, including
# run_full_duplex_lora_chain.pbs's data stage (DATA_RUNNER=).
#
# vLLM is only started when STEPS includes "dialogues". A resumed job that
# only has "audio" left to do (whole-utterance TTS) skips the vLLM download
# and server startup entirely.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STEPS="${STEPS:-use_cases,dialogues,enrich,audio}"
has_step() {
    [[ ",$STEPS," == *",$1,"* ]]
}

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

export WHOLE_UTTERANCE=1
export WHOLE_UTTERANCE_MAX_CHARS="${WHOLE_UTTERANCE_MAX_CHARS:-150}"
export TTS_BACKEND="${TTS_BACKEND:-qwen3}"

if ! has_step dialogues; then
    echo "[fdq] STEPS does not include dialogues; skipping vLLM, running run_full_duplex_training_data.sh directly."
    exec bash scripts/run_full_duplex_training_data.sh
fi

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is not available on PATH." >&2
    exit 1
}

# --------------------------------------------------------------------------
# vLLM server: same model/settings as run_dialogues_qwen_10000.pbs.
# --------------------------------------------------------------------------
if [[ -z "${PORT:-}" ]]; then
    JOB_NUM="${PBS_JOBID%%.*}"
    JOB_NUM="${JOB_NUM//[^0-9]/}"
    if [[ -n "$JOB_NUM" ]]; then
        export PORT="$((18000 + (10#$JOB_NUM % 20000)))"
    else
        export PORT="8000"
    fi
fi
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu129}"
export VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
export VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-}"
export VLLM_ENV_DIR="${VLLM_ENV_DIR:-.venv_vllm_qwen}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
if [[ ! -v VLLM_LIMIT_MM_PER_PROMPT ]]; then
    export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'
else
    export VLLM_LIMIT_MM_PER_PROMPT
fi

VLLM_PYTHON="$VLLM_ENV_DIR/bin/python"
VLLM_BIN="$VLLM_ENV_DIR/bin/vllm"
CPU_ARCH="$(uname -m)"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_34_${CPU_ARCH}.whl}"
VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-$VLLM_WHEEL_URL}"

echo "===== Qwen dialogue vLLM server ====="
echo "model:         $VLLM_MODEL"
echo "port:          $PORT"
echo "max_model_len: $MAX_MODEL_LEN"
echo "venv:          $VLLM_ENV_DIR"
echo "======================================"

if [[ -n "$VLLM_CUDA_MODULE" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
            if [[ -f "$module_init" ]]; then
                # shellcheck source=/dev/null
                source "$module_init"
                break
            fi
        done
    fi
    command -v module >/dev/null 2>&1 || {
        echo "ERROR: environment modules are not available; cannot load $VLLM_CUDA_MODULE." >&2
        exit 1
    }
    module load "$VLLM_CUDA_MODULE"
fi

[[ -d "$VLLM_ENV_DIR" ]] || uv venv "$VLLM_ENV_DIR"
export PATH="$(cd "$VLLM_ENV_DIR" && pwd)/bin:$PATH"
# shellcheck disable=SC2086
uv pip install --python "$VLLM_PYTHON" ${VLLM_BUILD_TOOLS:-ninja cmake}
uv pip install --python "$VLLM_PYTHON" "$VLLM_INSTALL_SPEC" --torch-backend="$UV_TORCH_BACKEND"

vllm_args=(
    "$VLLM_BIN" serve "$VLLM_MODEL"
    --tensor-parallel-size 1
    --host 127.0.0.1
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --reasoning-parser qwen3
)
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    vllm_args+=(--enforce-eager)
fi
if [[ -z "${VLLM_QWEN_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    VLLM_QWEN_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
fi
if [[ -n "${VLLM_QWEN_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    vllm_args+=(--default-chat-template-kwargs "$VLLM_QWEN_CHAT_TEMPLATE_KWARGS")
fi
if [[ -n "$VLLM_LIMIT_MM_PER_PROMPT" ]]; then
    vllm_args+=(--limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT")
fi

"${vllm_args[@]}" &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "waiting for vLLM ($VLLM_MODEL) to become ready (no health-check timeout)..."
while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited before serving $VLLM_MODEL (port ${PORT} may be held by another server)." >&2
        exit 1
    fi
    if curl --fail --silent --show-error \
        "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -qF "$VLLM_MODEL"; then
        break
    fi
    sleep 5
done
echo "vLLM is ready and serving $VLLM_MODEL; starting dialogue generation."

export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.9}"
export LLM_NUM_FEWSHOT="${LLM_NUM_FEWSHOT:-2}"
export LLM_BACKEND="openai-compatible"
export LLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export LLM_API_KEY="${LLM_API_KEY:-}"
if [[ -z "${LLM_OPENAI_ENDPOINT:-}" ]]; then
    CHAT_PROBE_CODE="$(curl -s -o /dev/null -w '%{http_code}' \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${VLLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping /no_think\"}],\"max_tokens\":1}" \
        2>/dev/null || true)"
    if [[ "$CHAT_PROBE_CODE" == "200" ]]; then
        export LLM_OPENAI_ENDPOINT="chat"
    else
        export LLM_OPENAI_ENDPOINT="completions"
    fi
    echo "endpoint probe: /v1/chat/completions -> HTTP ${CHAT_PROBE_CODE:-none}; using LLM_OPENAI_ENDPOINT=$LLM_OPENAI_ENDPOINT"
else
    export LLM_OPENAI_ENDPOINT
fi
export LLM_COMPLETIONS_NO_THINK="${LLM_COMPLETIONS_NO_THINK:-1}"
export LLM_MODEL="$VLLM_MODEL"
export LLM_FREQUENCY_PENALTY="${LLM_FREQUENCY_PENALTY:-0.3}"
export LLM_PRESENCE_PENALTY="${LLM_PRESENCE_PENALTY:-0.1}"
export DIALOGUE_GENERATION_MODE="${DIALOGUE_GENERATION_MODE:-multi-agent}"
export MULTI_AGENT_MIN_PAIRS="${MULTI_AGENT_MIN_PAIRS:-3}"
export MULTI_AGENT_MAX_PAIRS="${MULTI_AGENT_MAX_PAIRS:-5}"
export MULTI_AGENT_CONCURRENCY="${MULTI_AGENT_CONCURRENCY:-64}"
export MULTI_AGENT_USER_TEMPERATURE="${MULTI_AGENT_USER_TEMPERATURE:-0.85}"
export MULTI_AGENT_SYSTEM_TEMPERATURE="${MULTI_AGENT_SYSTEM_TEMPERATURE:-0.2}"
export MULTI_AGENT_JUDGE_TEMPERATURE="${MULTI_AGENT_JUDGE_TEMPERATURE:-0.0}"
export MULTI_AGENT_AIZUCHI_TEMPERATURE="${MULTI_AGENT_AIZUCHI_TEMPERATURE:-0.2}"
export MULTI_AGENT_AIZUCHI_MIN_CHARS="${MULTI_AGENT_AIZUCHI_MIN_CHARS:-16}"
export MULTI_AGENT_MAX_AIZUCHI_PER_USER="${MULTI_AGENT_MAX_AIZUCHI_PER_USER:-2}"
export MULTI_AGENT_EMPTY_POLICY="${MULTI_AGENT_EMPTY_POLICY:-fail}"
export MULTI_AGENT_AIZUCHI_MODE="${MULTI_AGENT_AIZUCHI_MODE:-separate}"

bash scripts/run_full_duplex_training_data.sh
