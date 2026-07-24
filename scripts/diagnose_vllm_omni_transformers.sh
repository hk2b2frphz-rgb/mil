#!/usr/bin/env bash
# Diagnose the transformers <-> vllm-omni mismatch behind the Qwen3-TTS Stage1
# failure:
#   "create_causal_mask() got an unexpected keyword argument input_embeds"
#
# Writes a compact report to vllm_omni_diag.txt (shared filesystem) and to
# stdout, so it can be viewed with `less` or shared as a screenshot without
# copy-paste.
#
# Usage (on a node that can see the venv):
#   bash scripts/diagnose_vllm_omni_transformers.sh
#   # override the interpreter if the venv lives elsewhere:
#   VLLM_PYTHON=/path/.venv-vllm-omni/bin/python bash scripts/diagnose_vllm_omni_transformers.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
REPORT="${REPORT:-$PWD/vllm_omni_diag.txt}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VLLM_PYTHON" >&2
    echo "       set VLLM_PYTHON=/path/.venv-vllm-omni/bin/python and retry" >&2
    exit 1
fi

{
    echo "=== vLLM-Omni / transformers diagnostic ==="
    date
    echo "python: $VLLM_PYTHON"

    echo
    echo "--- installed versions ---"
    "$VLLM_PYTHON" - <<'PY'
import importlib
for name in ("torch", "transformers", "tokenizers", "vllm", "vllm_omni", "numpy"):
    try:
        m = importlib.import_module(name)
        print(f"{name}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{name}: IMPORT FAILED: {e!r}")
PY

    echo
    echo "--- transformers.create_causal_mask signature ---"
    "$VLLM_PYTHON" - <<'PY'
import inspect
found = False
for mod in ("transformers.masking_utils",
            "transformers.modeling_attn_mask_utils",
            "transformers"):
    try:
        m = __import__(mod, fromlist=["create_causal_mask"])
    except Exception as e:
        print(f"{mod}: import error: {e!r}")
        continue
    fn = getattr(m, "create_causal_mask", None)
    if fn is not None:
        try:
            print(f"{mod}.create_causal_mask{inspect.signature(fn)}")
        except (TypeError, ValueError):
            print(f"{mod}.create_causal_mask (signature unavailable)")
        found = True
if not found:
    print("create_causal_mask not found in transformers")
PY

    echo
    echo "--- how vllm-omni calls create_causal_mask ---"
    OMNI_DIR="$("$VLLM_PYTHON" -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))' 2>/dev/null || true)"
    echo "vllm_omni dir: ${OMNI_DIR:-<unknown>}"
    if [[ -n "${OMNI_DIR:-}" ]]; then
        grep -rn "create_causal_mask" "$OMNI_DIR" 2>/dev/null | head -40 \
            || echo "(no direct references found under vllm_omni)"
    fi

    echo
    echo "--- pip metadata (Requires) ---"
    for pkg in vllm-omni vllm transformers; do
        echo "## $pkg"
        "$VLLM_PYTHON" -m pip show "$pkg" 2>/dev/null \
            | grep -iE '^(Name|Version|Requires):' \
            || echo "  (not installed / no metadata)"
    done

    echo
    echo "=== end of report ==="
} 2>&1 | tee "$REPORT"

echo
echo "レポートを書き出しました: $REPORT"
echo "確認: less \"$REPORT\"  （またはこの画面のスクショで共有してください）"
