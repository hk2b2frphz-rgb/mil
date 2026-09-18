#!/usr/bin/env bash
# Fast, CPU-only compatibility check between the Qwen3-TTS CustomVoice model and
# the vllm-omni implementation. No GPU, no model load: it only reads the cached
# config.json and vllm-omni's bundled config/code, then compares the codec /
# quantizer / vocab settings that would make the talker emit out-of-range token
# ids or code2wav reject a token count "not divisible by num_quantizers".
#
# Writes a report to qwen3_tts_compat.txt (shared FS) and stdout.
#
# Usage:
#   bash scripts/check_qwen3_tts_model_compat.sh
#   MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash scripts/check_qwen3_tts_model_compat.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
REPORT="${REPORT:-$PWD/qwen3_tts_compat.txt}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VLLM_PYTHON (set VLLM_PYTHON=...)" >&2
    exit 1
fi

{
    echo "=== Qwen3-TTS model <-> vllm-omni compatibility check ==="
    date
    echo "model: $MODEL_ID"

    echo
    echo "--- model config.json (cached, no download) ---"
    MODEL_ID="$MODEL_ID" "$VLLM_PYTHON" - <<'PY'
import json, os
model_id = os.environ["MODEL_ID"]
cfg_path = None
try:
    from transformers.utils import cached_file
    cfg_path = cached_file(model_id, "config.json", local_files_only=True)
except Exception as e:
    print("cached_file lookup failed:", repr(e))

if not cfg_path or not os.path.isfile(cfg_path):
    print("config.json not found in local HF cache.")
    print("Run the pipeline once (it downloads the model) or set HF_HOME, then retry.")
else:
    print("config.json:", cfg_path)
    # snapshots/<commit-hash>/config.json  -> the commit is the effective revision
    parent = os.path.basename(os.path.dirname(cfg_path))
    print("effective revision (snapshot dir):", parent)
    cfg = json.load(open(cfg_path))
    for k in ("architectures", "model_type", "transformers_version", "_name_or_path"):
        print(f"  {k}: {cfg.get(k)}")
    print("  -- codec / quantizer / vocab related keys (recursive) --")
    def walk(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    walk(v, key)
                elif any(t in k.lower() for t in
                         ("quant", "codec", "codebook", "vocab", "code2wav",
                          "num_", "hz", "frame", "group")):
                    print(f"    {key} = {v}")
    walk(cfg)
PY

    echo
    echo "--- vllm-omni's expected Qwen3-TTS config defaults ---"
    "$VLLM_PYTHON" - <<'PY'
import inspect
try:
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
except Exception as e:
    print("could not import Qwen3TTSConfig:", repr(e))
else:
    for cls in (Qwen3TTSConfig,) + tuple(
        getattr(Qwen3TTSConfig, "sub_configs", {}).values()
        if hasattr(Qwen3TTSConfig, "sub_configs") else ()
    ):
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        hits = {
            n: p.default for n, p in sig.parameters.items()
            if any(t in n.lower() for t in
                   ("quant", "codec", "codebook", "vocab", "num_", "group", "frame"))
        }
        if hits:
            print(f"{cls.__name__}:")
            for n, d in hits.items():
                print(f"    default {n} = {d}")
PY

    echo
    echo "--- hardcoded codec/quantizer constants in vllm-omni ---"
    OMNI_DIR="$("$VLLM_PYTHON" -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))' 2>/dev/null || true)"
    echo "vllm_omni dir: ${OMNI_DIR:-<unknown>}"
    if [[ -n "${OMNI_DIR:-}" ]]; then
        grep -rnE 'num_quantizers|not divisible|codebook_size|codec_vocab|vocab_size' \
            "$OMNI_DIR" 2>/dev/null | grep -iE 'qwen3_tts|code2wav|quant|codebook' \
            | head -40 || echo "(no matches)"
    fi

    echo
    echo "=== end ==="
} 2>&1 | tee "$REPORT"

echo
echo "レポート: $REPORT  （less で確認、またはスクショで共有）"
echo "見るポイント: モデル config の num_quantizers/codebook 数と、"
echo "vllm-omni 側の期待値（16 など）が一致しているか。ズレていれば revision 不整合。"
