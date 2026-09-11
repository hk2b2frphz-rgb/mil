#!/usr/bin/env bash
# Discover which Qwen3-TTS model (repo id / revision / codec shape) vllm-omni
# 0.21.0rc1 was actually built for. No Qwen3-TTS model revision matches its
# expected architecture (num_code_groups / KV heads), so the installed
# vllm-omni likely targets a different model or codec generation. This greps the
# installed vllm-omni package for the model ids, revisions, example configs, and
# hardcoded codec constants it ships with. No GPU, no network.
#
# Writes a report to vllm_omni_expected_model.txt.
#
# Usage:
#   bash scripts/find_vllm_omni_expected_model.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
REPORT="${REPORT:-$PWD/vllm_omni_expected_model.txt}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VLLM_PYTHON (set VLLM_PYTHON=...)" >&2
    exit 1
fi

OMNI_DIR="$("$VLLM_PYTHON" -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))' 2>/dev/null || true)"

{
    echo "=== what model does vllm-omni 0.21.0rc1 expect? ==="
    date
    echo "vllm_omni dir: ${OMNI_DIR:-<unknown>}"
    "$VLLM_PYTHON" -c 'import vllm_omni; print("vllm-omni:", getattr(vllm_omni, "__version__", "?"))' 2>/dev/null || true

    if [[ -z "${OMNI_DIR:-}" || ! -d "$OMNI_DIR" ]]; then
        echo "ERROR: could not locate the vllm_omni package directory." >&2
        echo "=== end ==="
        exit 1
    fi

    echo
    echo "--- HuggingFace-style model ids referenced in vllm-omni ---"
    grep -rnoE '[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]*[Tt][Tt][Ss][A-Za-z0-9_.-]*' "$OMNI_DIR" 2>/dev/null \
        | sort -u | head -40 || echo "(none)"
    echo "  (broader: any org/name pairs near 'pretrained'/'model'/'repo')"
    grep -rniE "from_pretrained\(|repo_id|model_id|pretrained_model|_name_or_path" "$OMNI_DIR" 2>/dev/null \
        | grep -oE '"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"' | sort -u | head -40 || echo "(none)"

    echo
    echo "--- revision / commit pins in vllm-omni ---"
    grep -rniE "revision[\"'= ]" "$OMNI_DIR" 2>/dev/null | head -20 || echo "(none)"

    echo
    echo "--- example / config / doc files bundled in vllm-omni ---"
    find "$OMNI_DIR" -maxdepth 4 -type f \
        \( -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \
           -o -iname '*example*' -o -iname '*demo*' \) 2>/dev/null | head -60 || echo "(none)"

    echo
    echo "--- codec shape constants (num_code_groups / num_quantizers / codebook) ---"
    grep -rnE 'num_code_groups|num_quantizers|codebook_size|codec_vocab|n_codebooks|code2wav' "$OMNI_DIR" 2>/dev/null \
        | grep -iE '=\s*[0-9]|:\s*[0-9]|16|32|24' | head -40 || echo "(none)"

    echo
    echo "--- README / model card hints inside the qwen3_tts model dir ---"
    QDIR="$(find "$OMNI_DIR" -type d -iname '*qwen3_tts*' 2>/dev/null | head -1 || true)"
    echo "qwen3_tts dir: ${QDIR:-<none>}"
    if [[ -n "${QDIR:-}" ]]; then
        find "$QDIR" -maxdepth 2 -type f | head -40
    fi

    echo
    echo "=== end ==="
} 2>&1 | tee "$REPORT"

echo
echo "レポート: $REPORT"
echo "狙い: vllm-omni が例/テストで使う正しいモデルID・revision、または期待codec形状を特定する。"
