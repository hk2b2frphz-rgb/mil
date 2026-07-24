#!/usr/bin/env bash
# Pin transformers to a version compatible with vllm-omni's bundled Qwen3-TTS
# code. That code calls create_causal_mask(input_embeds=...); transformers
# 5.14.1 renamed the kwarg to inputs_embeds, which breaks Stage1 with
#   "create_causal_mask() got an unexpected keyword argument input_embeds".
#
# This installs a candidate transformers into the vllm-omni venv (holding the
# torch stack fixed), then verifies:
#   1. vllm / vllm-omni's declared transformers requirement (lower bound), and
#   2. create_causal_mask exposes `input_embeds`, and
#   3. vllm and vllm_omni still import.
# The verdict (PASS/FAIL) is written to a report file so it needs no copy-paste.
#
# Usage (on a node that can see the venv and reach the proxy):
#   bash scripts/fix_vllm_omni_transformers.sh                    # default candidate
#   TRANSFORMERS_VERSION=4.56.0 bash scripts/fix_vllm_omni_transformers.sh
#
# On PASS, pin the same version in scripts/setup_vllm_omni_v100_env.sh.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
REPORT="${REPORT:-$PWD/vllm_omni_transformers_fix.txt}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VLLM_PYTHON (set VLLM_PYTHON=...)" >&2
    exit 1
fi

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
cat > "$CONSTRAINTS" <<EOF
torch==$TORCH_VERSION
torchaudio==$TORCHAUDIO_VERSION
torchvision==$TORCHVISION_VERSION
EOF

{
    echo "=== transformers pin attempt: $TRANSFORMERS_VERSION ==="
    date

    echo
    echo "--- vllm / vllm-omni transformers requirement (with specifiers) ---"
    "$VLLM_PYTHON" - <<'PY'
from importlib.metadata import requires
for pkg in ("vllm", "vllm-omni"):
    try:
        reqs = requires(pkg) or []
    except Exception as e:
        print(f"{pkg}: metadata error: {e!r}")
        continue
    hits = [r for r in reqs if r.lower().startswith("transformers")]
    print(f"{pkg}: {hits or 'transformers NOT pinned'}")
PY

    echo
    echo "--- installing transformers==$TRANSFORMERS_VERSION (torch held) ---"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VLLM_PYTHON" --constraint "$CONSTRAINTS" \
            "transformers==$TRANSFORMERS_VERSION" || echo "INSTALL FAILED (see above)"
    else
        PIP_CONSTRAINT="$CONSTRAINTS" "$VLLM_PYTHON" -m pip install \
            "transformers==$TRANSFORMERS_VERSION" || echo "INSTALL FAILED (see above)"
    fi

    echo
    echo "--- verify ---"
    "$VLLM_PYTHON" - <<'PY'
import inspect
ok = True
import transformers
print("transformers:", transformers.__version__)
try:
    from transformers.masking_utils import create_causal_mask
    params = list(inspect.signature(create_causal_mask).parameters)
    print("create_causal_mask params:", params)
    if "input_embeds" in params:
        print("PARAM OK: input_embeds present")
    elif "inputs_embeds" in params:
        print("PARAM MISMATCH: has inputs_embeds, vllm-omni needs input_embeds")
        ok = False
    else:
        print("PARAM UNKNOWN: neither input_embeds nor inputs_embeds")
        ok = False
except Exception as e:
    print("create_causal_mask import error:", repr(e))
    ok = False
for pkg in ("vllm", "vllm_omni"):
    try:
        __import__(pkg)
        print(f"import {pkg}: OK")
    except Exception as e:
        print(f"import {pkg}: FAILED: {e!r}")
        ok = False
print("RESULT:", "PASS" if ok else "FAIL")
PY

    echo "=== end ==="
} 2>&1 | tee "$REPORT"

echo
echo "レポート: $REPORT"
echo "RESULT: PASS なら、この版を setup に固定します。"
echo "RESULT: FAIL で 'import vllm FAILED' なら、その版は vllm に古すぎ → 別版を試すか、"
echo "vllm-omni 側コードを inputs_embeds へパッチする方針に切替えます。"
