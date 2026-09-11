#!/usr/bin/env bash
# Find a Qwen3-TTS CustomVoice model revision whose architecture matches what
# vllm-omni 0.21.0rc1 expects. The latest model revision changed num_code_groups
# (quantizer groups), the codec token id range, and the talker KV head count, so
# the talker emits out-of-range codes and code2wav rejects "not divisible by
# num_quantizers 16". Pinning an older, matching revision fixes it.
#
# No GPU, no model weights: this only lists the model's commit history and
# downloads each revision's small config.json to fingerprint the architecture.
# Needs network (proxy). Writes a report to qwen3_tts_revisions.txt.
#
# Usage:
#   PROXY_URL=http://<host>:<port> bash scripts/find_qwen3_tts_revision.sh
#   MAX_COMMITS=40 bash scripts/find_qwen3_tts_revision.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_PYTHON="${VLLM_PYTHON:-$PWD/.venv-vllm-omni/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
MAX_COMMITS="${MAX_COMMITS:-30}"
REPORT="${REPORT:-$PWD/qwen3_tts_revisions.txt}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VLLM_PYTHON (set VLLM_PYTHON=...)" >&2
    exit 1
fi

{
    echo "=== Qwen3-TTS revision finder (match vllm-omni 0.21.0rc1) ==="
    date
    echo "model: $MODEL_ID   scanning up to $MAX_COMMITS commits"

    echo
    echo "--- vllm-omni expected fingerprint (config defaults + hardcoded) ---"
    "$VLLM_PYTHON" - <<'PY'
import inspect
KEYS = ("num_code_groups", "num_quantizers", "num_key_value_heads",
        "num_attention_heads", "codebook_size", "codec_vocab_size", "vocab_size")
try:
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    for cls in (Qwen3TTSConfig,) + tuple(
        getattr(Qwen3TTSConfig, "sub_configs", {}).values()
        if hasattr(Qwen3TTSConfig, "sub_configs") else ()
    ):
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        hits = {n: p.default for n, p in sig.parameters.items() if n in KEYS}
        if hits:
            print(f"  {cls.__name__}: {hits}")
except Exception as e:
    print("  Qwen3TTSConfig import failed:", repr(e))
PY

    echo
    echo "--- per-revision config fingerprint (newest first) ---"
    MODEL_ID="$MODEL_ID" MAX_COMMITS="$MAX_COMMITS" "$VLLM_PYTHON" - <<'PY'
import json, os
from huggingface_hub import list_repo_commits, hf_hub_download

model_id = os.environ["MODEL_ID"]
max_commits = int(os.environ["MAX_COMMITS"])
KEYS = ("num_code_groups", "num_quantizers", "num_key_value_heads",
        "num_attention_heads", "codebook_size", "codec_vocab_size", "vocab_size")

def fingerprint(cfg):
    out = {}
    def walk(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    walk(v, key)
                elif k in KEYS:
                    out[key] = v
    walk(cfg)
    return out

try:
    commits = list_repo_commits(model_id)
except Exception as e:
    print("list_repo_commits failed:", repr(e))
    commits = []

for c in commits[:max_commits]:
    rev = c.commit_id
    when = getattr(c, "created_at", "?")
    title = (getattr(c, "title", "") or "").splitlines()[0][:50]
    try:
        p = hf_hub_download(model_id, "config.json", revision=rev)
        cfg = json.load(open(p))
        fp = fingerprint(cfg)
        tv = cfg.get("transformers_version")
        print(f"{rev[:12]}  {str(when)[:10]}  tf={tv}  {title}")
        for k, v in fp.items():
            print(f"        {k} = {v}")
    except Exception as e:
        print(f"{rev[:12]}  {str(when)[:10]}  config fetch failed: {e!r}")
PY

    echo
    echo "=== end ==="
} 2>&1 | tee "$REPORT"

echo
echo "レポート: $REPORT"
echo "上の 'vllm-omni expected' の num_code_groups / KV heads と一致する最新の revision を選び、"
echo "それをモデルのピン留めに使います（コミットハッシュ）。"
