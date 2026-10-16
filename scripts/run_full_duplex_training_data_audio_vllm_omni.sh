#!/usr/bin/env bash
# Audio-stage runner for run_full_duplex_lora_chain.pbs's "audio" pipeline
# stage: Qwen3-TTS whole-utterance synthesis via the vLLM-Omni engine
# (matching scripts/run_qwen_tts_vllm_common.sh / run_qwen_tts_vllm_10000_4gpu.pbs)
# instead of the slower transformers engine.
#
# Prerequisite (run once, separately, before using this):
#   qsub -V scripts/setup_vllm_omni_v100_env.pbs
# This engine is V100-specific (sm70-patched vLLM-Omni build), which is why
# the chain's AUDIO_QUEUE defaults to xvn_s while DATA_QUEUE (dialogue
# generation, Qwen3.6-27B) defaults to xan_s.
#
# Voice mode (QWEN_VOICE_MODE=auto|customvoice|clone|mixed, default auto):
#   customvoice        no clone refs given -> built-in voices
#   mixed              CLONE_OUT_DIR_MOSHI (or MOSHI_REF_WAV) given -> moshi's
#                       voice is cloned, user stays built-in (USER_SPEAKER_POOL)
#   clone              both CLONE_OUT_DIR_MOSHI and CLONE_OUT_DIR_USER (or
#                       MOSHI_REF_WAV/USER_REF_WAV) given -> both cloned
# CLONE_MODE=in-context (default) also needs *_REF_TEXT (or is read from the
# clone_voice_examples.py output dir); x-vector mode does not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

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
export CLONE_MODE="${CLONE_MODE:-in-context}"
export REF_RANK="${REF_RANK:-1}"

if [[ "$TTS_BACKEND" == "qwen3" || -z "${TTS_BACKEND:-}" ]] \
    && [[ "$QWEN_VOICE_MODE" == "clone" || "$QWEN_VOICE_MODE" == "mixed" ]]; then
    resolve_clone_ref() {  # resolve_clone_ref <field> <role>
        local field="$1" role="$2" clone_out
        if [[ "$role" == "moshi" ]]; then
            clone_out="${CLONE_OUT_DIR_MOSHI:-}"
        else
            clone_out="${CLONE_OUT_DIR_USER:-}"
        fi
        if [[ -z "$clone_out" ]]; then
            echo "ERROR: CLONE_OUT_DIR_${role^^} is required when an explicit ${role^^}_REF_* is absent" >&2
            return 1
        fi
        uv run python scripts/resolve_clone_refs.py \
            --clone-out-dir "$clone_out" \
            --rank "$REF_RANK" \
            --mode "$CLONE_MODE" \
            --field "$field"
    }
    export MOSHI_REF_WAV="${MOSHI_REF_WAV:-$(resolve_clone_ref wav moshi)}"
    if [[ "$CLONE_MODE" == "in-context" ]]; then
        export MOSHI_REF_TEXT="${MOSHI_REF_TEXT:-$(resolve_clone_ref text moshi)}"
    fi
    if [[ "$QWEN_VOICE_MODE" == "clone" ]]; then
        export USER_REF_WAV="${USER_REF_WAV:-$(resolve_clone_ref wav user)}"
        if [[ "$CLONE_MODE" == "in-context" ]]; then
            export USER_REF_TEXT="${USER_REF_TEXT:-$(resolve_clone_ref text user)}"
        fi
    fi
    echo "[fdq-audio] clone mode: $CLONE_MODE (rank=$REF_RANK)"
    echo "[fdq-audio] moshi ref:  $MOSHI_REF_WAV"
    [[ "$QWEN_VOICE_MODE" == "clone" ]] && echo "[fdq-audio] user ref:   $USER_REF_WAV"
fi

export TTS_BACKEND=qwen3
export QWEN_ENGINE=vllm-omni
export TTS_DTYPE="${TTS_DTYPE:-float16}"
export TTS_DEVICE="${TTS_DEVICE:-cuda}"
export WHOLE_UTTERANCE=1
export WHOLE_UTTERANCE_MAX_CHARS="${WHOLE_UTTERANCE_MAX_CHARS:-150}"
export VLLM_STAGE_CONFIG="${VLLM_STAGE_CONFIG:-$PWD/configs/qwen3_tts_v100_batch16.yaml}"
export VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"

echo "===== Qwen3-TTS whole-utterance audio (vllm-omni) ====="
echo "voice_mode:      $QWEN_VOICE_MODE"
echo "vllm_python:     $VLLM_PYTHON"
echo "vllm_stage_cfg:  $VLLM_STAGE_CONFIG"
echo "========================================================"

bash scripts/run_full_duplex_training_data.sh
