#!/usr/bin/env bash
# Shared persistent model cache for local/HPC evaluation drivers. Source this
# before uv or model workers start so every subprocess inherits the same paths.

model_cache_init() {
    local repo_root="$1"
    local cache_root="${MODEL_CACHE_ROOT:-$repo_root/.cache/models}"
    export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"
    echo "[model-cache] HF_HOME=$HF_HOME"
}
