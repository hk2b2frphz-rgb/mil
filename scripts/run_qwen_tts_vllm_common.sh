#!/usr/bin/env bash
# Shared body of the Qwen3-TTS vLLM-Omni production wrappers.
#
# The per-scale wrappers (run_qwen_tts_vllm_{1000,3000,10000}_4gpu.pbs) carry
# only their PBS directives and a SOURCE_BATCH_ID default, then exec this file.
# Everything that decides what gets rendered lives here, so a fix to the voice
# mode rules or the engine settings lands on every scale at once.
#
# V100 32GB x4 throughput preset for Qwen3-TTS 1.7B.  Each shard sees one GPU
# and hosts one complete vLLM-Omni pipeline.  Re-submit the same wrapper after
# walltime; the underlying production script resumes.
#
# This file is not hashed into the run's generation_config.json -- only the
# values it exports are -- so it can be refactored without breaking the resume
# of a run that is already in flight.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [[ -z "${SOURCE_BATCH_ID:-}" ]]; then
    echo "ERROR: SOURCE_BATCH_ID must be set before sourcing this script." >&2
    echo "Submit one of scripts/run_qwen_tts_vllm_{1000,3000,10000}_4gpu.pbs." >&2
    exit 1
fi
export SOURCE_BATCH_ID

# shellcheck source=/dev/null
source "$PWD/scripts/run_id_utils.sh"

export VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-cuda12.6_cudnn9.7.1_nccl2.24.3}"
if ! command -v module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$module_init" ]]; then
            # shellcheck source=/dev/null
            source "$module_init"
            break
        fi
    done
fi
command -v module >/dev/null 2>&1 || {
    echo "ERROR: environment modules are unavailable; cannot load $VLLM_CUDA_MODULE" >&2
    exit 1
}
module load "$VLLM_CUDA_MODULE"

# Resolve auto from the references visible to this PBS process.  Doing this
# before choosing BATCH_ID prevents clone/customvoice runs from sharing an
# output directory, and makes a folder-only submission do what it says.
resolve_qwen_voice_mode() {
    local requested="${QWEN_VOICE_MODE:-auto}"
    local moshi_audio=0 user_audio=0 moshi_any=0 user_any=0
    [[ -n "${CLONE_OUT_DIR_MOSHI:-}${MOSHI_REF_WAV:-}" ]] && moshi_audio=1
    [[ -n "${CLONE_OUT_DIR_USER:-}${USER_REF_WAV:-}" ]] && user_audio=1
    [[ "$moshi_audio" == "1" || -n "${MOSHI_REF_TEXT:-}" ]] && moshi_any=1
    [[ "$user_audio" == "1" || -n "${USER_REF_TEXT:-}" ]] && user_any=1

    if [[ "$moshi_any" == "1" && "$moshi_audio" == "0" ]]; then
        echo "ERROR: MOSHI_REF_TEXT was set without CLONE_OUT_DIR_MOSHI or MOSHI_REF_WAV" >&2
        return 1
    fi
    if [[ "$user_any" == "1" && "$user_audio" == "0" ]]; then
        echo "ERROR: USER_REF_TEXT was set without CLONE_OUT_DIR_USER or USER_REF_WAV" >&2
        return 1
    fi

    case "$requested" in
        auto)
            if [[ "$moshi_audio" == "1" && "$user_audio" == "1" ]]; then
                printf 'clone'
            elif [[ "$moshi_audio" == "1" ]]; then
                printf 'mixed'
            elif [[ "$user_audio" == "1" ]]; then
                echo "ERROR: a user clone source without a moshi clone source is unsupported" >&2
                return 1
            else
                printf 'customvoice'
            fi
            ;;
        customvoice)
            if [[ "$moshi_any" == "1" || "$user_any" == "1" ]]; then
                echo "ERROR: QWEN_VOICE_MODE=customvoice conflicts with clone reference variables" >&2
                return 1
            fi
            printf 'customvoice'
            ;;
        mixed)
            if [[ "$moshi_audio" == "0" || "$user_any" == "1" ]]; then
                echo "ERROR: QWEN_VOICE_MODE=mixed needs only a moshi clone source" >&2
                return 1
            fi
            printf 'mixed'
            ;;
        clone)
            if [[ "$moshi_audio" == "0" || "$user_audio" == "0" ]]; then
                echo "ERROR: QWEN_VOICE_MODE=clone needs both moshi and user clone sources" >&2
                return 1
            fi
            printf 'clone'
            ;;
        *)
            echo "ERROR: QWEN_VOICE_MODE must be auto, customvoice, clone, or mixed: $requested" >&2
            return 1
            ;;
    esac
}
export QWEN_VOICE_MODE="$(resolve_qwen_voice_mode)"
case "$QWEN_VOICE_MODE" in
    customvoice) default_batch_id="${SOURCE_BATCH_ID}_tts_vllm_4gpu" ;;
    clone)       default_batch_id="${SOURCE_BATCH_ID}_tts_vllm_clone_4gpu" ;;
    mixed)       default_batch_id="${SOURCE_BATCH_ID}_tts_vllm_mixed_4gpu" ;;
esac
if [[ -z "${BATCH_ID:-}" ]]; then
    if [[ "${FRESH:-0}" == "1" ]]; then
        export BATCH_ID="$(default_run_id "$default_batch_id")"
    else
        export BATCH_ID="$default_batch_id"
    fi
fi
export TTS_BACKEND=qwen3
export QWEN_ENGINE=vllm-omni
export TTS_DTYPE=float16
export TTS_BATCH_SIZE="${TTS_BATCH_SIZE:-16}"
export DIALOGUE_BATCH_SIZE="${DIALOGUE_BATCH_SIZE:-16}"
export VLLM_STAGE_CONFIG="${VLLM_STAGE_CONFIG:-$PWD/configs/qwen3_tts_v100_batch16.yaml}"
export VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"

# One production script drives every scale; SOURCE_BATCH_ID selects the corpus
# and NUM_DIALOGUES trims it.  Its filename still says 10000 for historical
# reasons.
exec bash scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs
