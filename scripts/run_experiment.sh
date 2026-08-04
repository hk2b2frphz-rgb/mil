#!/usr/bin/env bash
# Launch one experiment from experiments/<EXP>/config.yaml.
#
# Usage:
#   bash scripts/run_experiment.sh <EXP_NAME> <SRC_RUN_DIR>
#
# Example:
#   bash scripts/run_experiment.sh lora_base_config ./data/runs/2026-06-02_130539
#
# 挙動:
# - <SRC_RUN_DIR>/training_set/ を experiments/<EXP>/data/training_set へ
#   hardlink で展開（失敗時は cp -r）。
# - manifest を seed 固定で 9:1 split。train.jsonl / eval.jsonl を生成。
# - config.yaml を実パスで埋めた _resolved.yaml にして torchrun に渡す。
# - stdout/stderr は experiments/<EXP>/run.log にも tee。

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: bash で実行してください" >&2
    exit 1
fi
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <EXP_NAME> <SRC_RUN_DIR>" >&2
    echo "  例: bash $0 lora_base_config ./data/runs/2026-06-02_130539" >&2
    exit 1
fi
EXP_NAME="$1"
SRC_RUN_DIR_RAW="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="$REPO_ROOT/experiments/$EXP_NAME"
if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: 実験フォルダがありません: $EXP_DIR" >&2
    exit 1
fi
CONFIG_SRC="$EXP_DIR/config.yaml"
if [[ ! -f "$CONFIG_SRC" ]]; then
    echo "ERROR: config.yaml がありません: $CONFIG_SRC" >&2
    exit 1
fi

SRC_RUN_DIR="$(realpath -m "$SRC_RUN_DIR_RAW")"
SRC_TRAIN_SET="$SRC_RUN_DIR/training_set"
SRC_MANIFEST="$SRC_TRAIN_SET/synthetic_moshi_train.jsonl"
if [[ ! -f "$SRC_MANIFEST" ]]; then
    echo "ERROR: source manifest がありません: $SRC_MANIFEST" >&2
    echo "  <SRC_RUN_DIR> は data/runs/<RUN_ID> を指定してください" >&2
    exit 1
fi

EXP_DATA_DIR="$EXP_DIR/data"
EXP_TRAIN_SET="$EXP_DATA_DIR/training_set"
# split manifest は training_set/ の中に置く。
# sphn.dataset_jsonl は jsonl の "path" を **manifest のあるディレクトリ** を基点に
# 解決するので、ここを training_set/ にしないと sample_xxx.wav が見つからずに
# 空 dataset になり、最初の next(iter(dataset)) で StopIteration を出す。
EXP_TRAIN_MF="$EXP_TRAIN_SET/train.jsonl"
EXP_EVAL_MF="$EXP_TRAIN_SET/eval.jsonl"
RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
EXP_CKPT_DIR="$EXP_DIR/checkpoints/$RUN_TS"
EXP_LOG="$EXP_DIR/run_${RUN_TS}.log"
RESOLVED_CONFIG="$EXP_DIR/_resolved.yaml"

# repo 配下を絶対化
MOSHI_FT_REPO="${MOSHI_FT_REPO:-$REPO_ROOT/../moshi-finetune}"
MOSHI_FT_REPO="$(realpath -m "$MOSHI_FT_REPO")"

mkdir -p "$EXP_DATA_DIR"

if [[ "${REFRESH_EXP_DATA:-0}" == "1" ]]; then
    echo "[exp] REFRESH_EXP_DATA=1: clearing previous experiment data"
    rm -rf "$EXP_TRAIN_SET" "$EXP_TRAIN_MF" "$EXP_EVAL_MF"
fi

# ---------------------------------------------------------------------------
# 1) training_set を実験フォルダにコピー（hardlink を試みて、ダメなら cp -r）
# ---------------------------------------------------------------------------
if [[ ! -d "$EXP_TRAIN_SET" ]]; then
    echo "[exp] training_set を展開: $SRC_TRAIN_SET → $EXP_TRAIN_SET"
    if cp -rl "$SRC_TRAIN_SET" "$EXP_TRAIN_SET" 2>/dev/null; then
        echo "[exp]   hardlink で展開（容量ゼロ）"
    else
        echo "[exp]   hardlink 不可なので cp -r で展開"
        cp -r "$SRC_TRAIN_SET" "$EXP_TRAIN_SET"
    fi
else
    echo "[exp] training_set は既にあります: $EXP_TRAIN_SET"
fi

# ---------------------------------------------------------------------------
# 2) manifest を 9:1 split（seed=0 固定）
# ---------------------------------------------------------------------------
if [[ ! -f "$EXP_TRAIN_MF" || ! -f "$EXP_EVAL_MF" ]]; then
    echo "[exp] manifest を 9:1 split: $EXP_TRAIN_MF / $EXP_EVAL_MF"
    python3 - "$EXP_TRAIN_SET/synthetic_moshi_train.jsonl" "$EXP_TRAIN_MF" "$EXP_EVAL_MF" <<'PY'
import json, random, sys, pathlib
src, train_out, eval_out = map(pathlib.Path, sys.argv[1:])
lines = [l for l in src.read_text().splitlines() if l.strip()]
if len(lines) < 2:
    sys.exit(
        f"[exp] ERROR: need >= 2 samples to split into train/eval, got {len(lines)}. "
        "Increase NUM_CASES / dialogue count."
    )
rng = random.Random(0)
rng.shuffle(lines)
n_eval = min(max(1, len(lines) // 10), len(lines) - 1)
eval_lines = lines[:n_eval]
train_lines = lines[n_eval:]
train_out.write_text("\n".join(train_lines) + "\n")
eval_out.write_text("\n".join(eval_lines) + "\n")
print(f"[exp]   total={len(lines)} train={len(train_lines)} eval={len(eval_lines)}")
PY
else
    echo "[exp] split 済み manifest を再利用"
fi

# ---------------------------------------------------------------------------
# 3) config.yaml の data.train_data / data.eval_data / run_dir を埋める
#    moshi_paths.config_path が HF ファイル名のままだと train.py が file not found
#    になるので、hf_hub_download で落としてからローカル絶対パスに差し替える。
# ---------------------------------------------------------------------------
echo "[exp] _resolved.yaml を生成"
python3 - "$CONFIG_SRC" "$RESOLVED_CONFIG" "$EXP_TRAIN_MF" "$EXP_EVAL_MF" "$EXP_CKPT_DIR" "$MOSHI_FT_REPO" <<'PY'
import re, sys, pathlib, subprocess
src, dst, train, evald, ckpt, ft_repo = sys.argv[1:]
src = pathlib.Path(src); dst = pathlib.Path(dst)
text = src.read_text()

def sub_block_value(text, key, value):
    return re.sub(rf'^(\s*{re.escape(key)}\s*:\s*).*$', rf'\1"{value}"', text, flags=re.M)

text = sub_block_value(text, "train_data", str(train))
text = sub_block_value(text, "eval_data",  str(evald))
text = sub_block_value(text, "run_dir",    str(ckpt))

# moshi_paths.hf_repo_id と config_path を読み、config_path が HF 内のファイル名
# っぽければ落として local 絶対パスに差し替える。
m_repo = re.search(r'^\s*hf_repo_id\s*:\s*["\']?([^"\'\n]+)["\']?\s*$', text, re.M)
m_conf = re.search(r'^(\s*config_path\s*:\s*)["\']?([^"\'\n]+)["\']?\s*$', text, re.M)
if m_repo and m_conf:
    hf_repo = m_repo.group(1).strip()
    cfg     = m_conf.group(2).strip()
    if cfg and "/" not in cfg and not pathlib.Path(cfg).is_absolute():
        # uv 経由で hf_hub_download を呼んで落とす。huggingface_hub は moshi-finetune の
        # 依存に入っているので uv run で叩ける。
        out = subprocess.check_output([
            "uv", "run", "--project", ft_repo, "python", "-c",
            f"from huggingface_hub import hf_hub_download; print(hf_hub_download('{hf_repo}', '{cfg}'))",
        ], text=True).strip().splitlines()
        local_path = out[-1]
        print(f"[exp] config_path: '{cfg}' -> '{local_path}'")
        text = sub_block_value(text, "config_path", local_path)

dst.write_text(text)
PY

# Keep only the best-eval-loss checkpoint (plus the just-saved one) during
# training instead of the num_ckpt_keep most recent ones. Default on; set
# HP_KEEP_BEST_ONLY=false to restore the recency-based num_ckpt_keep pruning.
export HP_KEEP_BEST_ONLY="${HP_KEEP_BEST_ONLY:-true}"

# Optional launch-time hyperparameter overrides.
# Use HP_* env vars from PBS/qsub without editing experiment config files.
python3 - "$RESOLVED_CONFIG" <<'PY'
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

TOP_LEVEL = {
    "HP_BATCH_SIZE": "batch_size",
    "HP_NUM_MICROBATCHES": "num_microbatches",
    "HP_MAX_STEPS": "max_steps",
    "HP_MAX_NORM": "max_norm",
    "HP_DURATION_SEC": "duration_sec",
    "HP_GRADIENT_CHECKPOINTING": "gradient_checkpointing",
    "HP_PARAM_DTYPE": "param_dtype",
    "HP_SEED": "seed",
    "HP_LOG_FREQ": "log_freq",
    "HP_CKPT_FREQ": "ckpt_freq",
    "HP_NUM_CKPT_KEEP": "num_ckpt_keep",
    "HP_DO_EVAL": "do_eval",
    "HP_EVAL_FREQ": "eval_freq",
    "HP_KEEP_BEST_ONLY": "keep_best_only",
    "HP_EARLY_STOPPING_PATIENCE": "early_stopping_patience",
    "HP_EARLY_STOPPING_MIN_DELTA": "early_stopping_min_delta",
}
SECTIONS = {
    "lora": {
        "HP_LORA_ENABLE": "enable",
        "HP_LORA_RANK": "rank",
        "HP_LORA_SCALING": "scaling",
        "HP_LORA_FT_EMBED": "ft_embed",
    },
    "optim": {
        "HP_LR": "lr",
        "HP_WEIGHT_DECAY": "weight_decay",
        "HP_PCT_START": "pct_start",
        "HP_LR_SCHEDULER": "scheduler",
    },
}


def yaml_value(raw: str) -> str:
    low = raw.lower()
    if low in {"true", "false"}:
        return low
    if low in {"null", "none"}:
        return "null"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", raw):
        return raw
    return '"' + raw.replace("\\", "\\\\").replace('"', '\\"') + '"'


def set_top_level(src: str, key: str, value: str) -> str:
    line = f"{key}: {value}"
    pat = rf"^{re.escape(key)}\s*:.*$"
    if re.search(pat, src, flags=re.M):
        return re.sub(pat, line, src, count=1, flags=re.M)
    return src.rstrip() + "\n" + line + "\n"


def set_section_value(src: str, section: str, key: str, value: str) -> str:
    section_pat = rf"^{re.escape(section)}:\s*$"
    match = re.search(section_pat, src, flags=re.M)
    if not match:
        return src.rstrip() + f"\n{section}:\n  {key}: {value}\n"
    start = match.end()
    next_match = re.search(r"^[A-Za-z0-9_][A-Za-z0-9_-]*:\s*", src[start:], flags=re.M)
    end = start + next_match.start() if next_match else len(src)
    block = src[start:end]
    key_pat = rf"^(\s*){re.escape(key)}\s*:.*$"
    if re.search(key_pat, block, flags=re.M):
        block = re.sub(key_pat, rf"\1{key}: {value}", block, count=1, flags=re.M)
    else:
        if block and not block.endswith("\n"):
            block += "\n"
        block += f"  {key}: {value}\n"
    return src[:start] + block + src[end:]


for env_name, key in TOP_LEVEL.items():
    raw = os.environ.get(env_name)
    if raw:
        text = set_top_level(text, key, yaml_value(raw))

for section, mapping in SECTIONS.items():
    for env_name, key in mapping.items():
        raw = os.environ.get(env_name)
        if raw:
            text = set_section_value(text, section, key, yaml_value(raw))

path.write_text(text)
PY

# ---------------------------------------------------------------------------
# 4) moshi-finetune の依存（pyproject patch / venv / setuptools 等）は
#    run_pipeline.sh の finetune ステップと同じ処理を ad-hoc に行う。
#    ここでは「既に run_pipeline.sh で 1 度成功している」ことを前提に、
#    .venv の存在だけ確認する。
# ---------------------------------------------------------------------------
if [[ ! -d "$MOSHI_FT_REPO/.venv" ]]; then
    echo "ERROR: $MOSHI_FT_REPO/.venv がありません。" >&2
    echo "  まず run_pipeline.sh の finetune ステップを 1 度通して依存を準備してください。" >&2
    exit 1
fi

# 単 GPU で BACKEND="nccl" のまま init_process_group → dist.barrier すると
# 環境によっては無言で hang する。WORLD_SIZE=1 のときだけ gloo にする。
# Apply LoRA-side compatibility patches to the Kyutai moshi-finetune checkout.
python3 "$REPO_ROOT/scripts/patch_kyutai_moshi_finetune.py" --ft-repo "$MOSHI_FT_REPO"

DISTRIBUTED_PY="$MOSHI_FT_REPO/finetune/distributed.py"
if [[ -f "$DISTRIBUTED_PY" ]] && ! grep -q 'AUTO_PATCH_BACKEND_WORLD1' "$DISTRIBUTED_PY"; then
    echo "[exp] finetune/distributed.py の BACKEND を WORLD_SIZE=1 で gloo に分岐"
    python3 - "$DISTRIBUTED_PY" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = 'BACKEND = "nccl"'
new = (
    '# AUTO_PATCH_BACKEND_WORLD1: single GPU は gloo にして NCCL barrier hang を回避\n'
    'import os as _os\n'
    'BACKEND = "gloo" if _os.environ.get("WORLD_SIZE", "1") == "1" else "nccl"'
)
if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
fi

# CUDA_VISIBLE_DEVICES 既定値
INTERLEAVER_PY="$MOSHI_FT_REPO/finetune/data/interleaver.py"
if [[ -f "$INTERLEAVER_PY" ]] && ! grep -q 'AUTO_PATCH_NL_FALLBACK' "$INTERLEAVER_PY"; then
    echo "[exp] finetune/data/interleaver.py に tokenizer newline fallback を適用"
    python3 - "$INTERLEAVER_PY" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '    nl_piece = tokenizer.encode("\\n")[-1]'
new = (
    '    # AUTO_PATCH_NL_FALLBACK: JP tokenizer can return [] for "\\n"\n'
    '    _nl_enc = tokenizer.encode("\\n") or tokenizer.encode(" ")\n'
    '    nl_piece = _nl_enc[-1] if _nl_enc else 0'
)
if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
fi

NPROC="${NPROC:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _devs=""
    for ((i=0; i<NPROC; i++)); do
        _devs+="${_devs:+,}$i"
    done
    export CUDA_VISIBLE_DEVICES="$_devs"
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Match the full-FT launcher's fail-fast dataset health check. This verifies
# that both manifests contain usable records, all referenced audio files exist,
# and every distributed rank is expected to receive at least one eval chunk.
LORA_DATASET_HEALTH="$EXP_DIR/lora_dataset_health_${RUN_TS}.json"
echo "[exp] checking LoRA train/eval dataset health"
python3 "$REPO_ROOT/scripts/check_lora_dataset.py" \
    --train-manifest "$EXP_TRAIN_MF" \
    --eval-manifest "$EXP_EVAL_MF" \
    --config "$RESOLVED_CONFIG" \
    --world-size "$NPROC" \
    --output-json "$LORA_DATASET_HEALTH"

# 単 GPU は python 直起動 (torchrun + FSDP 単 GPU の segfault 回避)
# 複数 GPU は torchrun。USE_TORCHRUN=1 で単 GPU でも torchrun を強制可能。
USE_TORCHRUN="${USE_TORCHRUN:-auto}"
if [[ "$USE_TORCHRUN" == "auto" ]]; then
    if [[ "$NPROC" -eq 1 ]]; then USE_TORCHRUN=0; else USE_TORCHRUN=1; fi
fi

if [[ "$USE_TORCHRUN" == "1" ]]; then
    TORCHRUN_ARGS=(
        --nnodes 1
        --node-rank 0
        --nproc-per-node "$NPROC"
        --master-addr "$MASTER_ADDR"
        --master-port "$MASTER_PORT"
    )
    if command -v torchrun >/dev/null 2>&1; then
        LAUNCH_CMD=(torchrun "${TORCHRUN_ARGS[@]}" -m train "$RESOLVED_CONFIG")
    else
        LAUNCH_CMD=(uv run --project "$MOSHI_FT_REPO" torchrun "${TORCHRUN_ARGS[@]}" -m train "$RESOLVED_CONFIG")
    fi
    LAUNCH_DESC="torchrun (nnodes=1 node_rank=0 nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT)"
else
    # 単 GPU 直起動。torchrun が export するはずの分散 env を自前で渡す。
    export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
    LAUNCH_CMD=(uv run --project "$MOSHI_FT_REPO" python -m train "$RESOLVED_CONFIG")
    LAUNCH_DESC="python direct (single GPU, no torchrun)"
fi

echo "[exp] launching: $LAUNCH_DESC"
echo "       exp:     $EXP_NAME"
echo "       config:  $RESOLVED_CONFIG"
echo "       train:   $EXP_TRAIN_MF"
echo "       eval:    $EXP_EVAL_MF"
echo "       ckpt:    $EXP_CKPT_DIR"
echo "       nproc:   $NPROC"
echo "       CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "       log:     $EXP_LOG"

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
    echo "[exp] MLflow live sync enabled"
    echo "[exp] checking MLflow availability"
    if command -v timeout >/dev/null 2>&1; then
        MLFLOW_CHECK_CMD=(timeout "${MLFLOW_IMPORT_TIMEOUT:-120}" uv run python -c "import mlflow")
    else
        MLFLOW_CHECK_CMD=(uv run python -c "import mlflow")
    fi
    if ! "${MLFLOW_CHECK_CMD[@]}" >/dev/null 2>&1; then
        echo "[exp] installing mlflow into project uv environment"
        uv pip install mlflow
    fi
    MLFLOW_RUN_ID_FILE="$EXP_DIR/mlflow_run_${RUN_TS}.id"
    MLFLOW_SYNC_CMD=(uv run python scripts/sync_mlflow_metrics.py
        --run-dir "$EXP_CKPT_DIR" \
        --config "$RESOLVED_CONFIG" \
        --dataset-health "$LORA_DATASET_HEALTH" \
        --experiment "${MLFLOW_EXPERIMENT_NAME:-default}" \
        --run-name "${MLFLOW_RUN_NAME:-$(basename "$EXP_CKPT_DIR")}" \
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
                run_mlflow_sync_once || echo "[exp] WARN: periodic MLflow sync failed" >&2
                sleep "$MLFLOW_LIVE_SYNC_INTERVAL"
            done
        ) &
        MLFLOW_SYNC_PID="$!"
        echo "[exp] MLflow live sync interval: ${MLFLOW_LIVE_SYNC_INTERVAL}s"
    fi
fi

# ---------------------------------------------------------------------------
# Debug プリント: 学習に入る前に、データセットの先頭サンプルの対話本文と
# tokenizer round-trip を流す。DEBUG=1 で有効化。
# ---------------------------------------------------------------------------
if [[ "${DEBUG:-0}" == "1" ]]; then
    echo
    echo "[debug] DEBUG=1 が指定されたのでデータと tokenizer を点検します"
    DEBUG_N="${DEBUG_N:-3}"
    MOSHI_FT_REPO="$MOSHI_FT_REPO" \
    EXP_TRAIN_MF="$EXP_TRAIN_MF" \
    DEBUG_N="$DEBUG_N" \
    uv run --project "$MOSHI_FT_REPO" python <<'PY'
import json, os, pathlib, sys

ft_repo = pathlib.Path(os.environ["MOSHI_FT_REPO"]).resolve()
if (ft_repo / "finetune").is_dir():
    sys.path.insert(0, str(ft_repo))

mf = pathlib.Path(os.environ["EXP_TRAIN_MF"])
n  = int(os.environ.get("DEBUG_N", "3"))

print(f"[debug] manifest: {mf}")
lines = [l for l in mf.read_text().splitlines() if l.strip()]
print(f"[debug] manifest entries: {len(lines)} (先頭 {n} 件を表示)")

# tokenizer 用意（cuda 使わない）
from huggingface_hub import hf_hub_download
from moshi.models import loaders
local_cfg = hf_hub_download("llm-jp/llm-jp-moshi-v1", "moshi_lm_kwargs.json")
ci = loaders.CheckpointInfo.from_hf_repo("llm-jp/llm-jp-moshi-v1", config_path=local_cfg)
spm = ci.get_text_tokenizer()
print(f"[debug] tokenizer vocab_size={spm.vocab_size()} bos={spm.bos_id()} eos={spm.eos_id()}")

manifest_dir = mf.parent
for i, line in enumerate(lines[:n]):
    entry = json.loads(line)
    wav_path = (manifest_dir / entry["path"]).resolve()
    json_path = wav_path.with_suffix(".json")
    print()
    print(f"[debug] --- sample {i+1}/{n} ---")
    print(f"[debug] wav : {wav_path.name}  ({entry.get('duration')} sec)")
    if not json_path.exists():
        print(f"[debug] (対応する {json_path.name} が無いのでテキストは表示できません)")
        continue
    meta = json.loads(json_path.read_text())
    # alignments: [[text, [start, end], label], ...]
    aligns = meta.get("alignments", [])
    print(f"[debug] turns: {len(aligns)}")
    for j, a in enumerate(aligns[:10]):
        text, span, label = a
        print(f"[debug]   [{j:02d}] ({label}) {span[0]:.2f}-{span[1]:.2f}s: {text}")
    if aligns:
        # 1 ターン目のテキストで encode→decode 確認
        first_text = aligns[0][0]
        ids = spm.encode(first_text)
        back = spm.decode(ids)
        print(f"[debug] round-trip check on first turn:")
        print(f"[debug]   in : {first_text!r}")
        print(f"[debug]   ids: {ids[:20]}{'...' if len(ids) > 20 else ''} (len={len(ids)})")
        print(f"[debug]   out: {back!r}")
        if back.strip() != first_text.strip():
            print(f"[debug]   ⚠  round-trip mismatch — tokenizer がテキストを完全に往復できていません")
print()
print("[debug] 確認終了。学習を起動します。")
PY
fi

# ---------------------------------------------------------------------------
# 5) 起動。run.log にも tee。
# ---------------------------------------------------------------------------
set +e
(
    cd "$MOSHI_FT_REPO" && \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${LAUNCH_CMD[@]}"
) 2>&1 | tee "$EXP_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ -n "$MLFLOW_SYNC_PID" ]]; then
    kill "$MLFLOW_SYNC_PID" >/dev/null 2>&1 || true
    wait "$MLFLOW_SYNC_PID" 2>/dev/null || true
fi

if [[ "$MLFLOW_ENABLED" == "1" ]]; then
    echo "[exp] final MLflow sync"
    run_mlflow_sync_once || echo "[exp] WARN: final MLflow sync failed" >&2
fi

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "[exp] ERROR: training failed with status $TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi

# ---------------------------------------------------------------------------
# 6) 学習成功後、eval loss が最小の checkpoint を自動選択して merge/export する。
#    SKIP_AUTO_POSTPROCESS=1 で無効化できる。
# ---------------------------------------------------------------------------
if [[ "${SKIP_AUTO_POSTPROCESS:-0}" != "1" ]]; then
    echo
    echo "[exp] selecting best checkpoint by eval loss"

    SAVE_ADAPTERS="$(python3 - "$RESOLVED_CONFIG" <<'PY'
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r'^\s*save_adapters\s*:\s*(true|false)\s*$', text, re.M | re.I)
print(m.group(1).lower() if m else "true")
PY
)"
    HF_REPO_ID="$(python3 - "$RESOLVED_CONFIG" <<'PY'
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r'^\s*hf_repo_id\s*:\s*["\']?([^"\'\n]+)["\']?\s*$', text, re.M)
print(m.group(1).strip() if m else "llm-jp/llm-jp-moshi-v1")
PY
)"

    if [[ "$SAVE_ADAPTERS" == "false" ]]; then
        CKPT_FILENAME="consolidated/consolidated.safetensors"
    else
        CKPT_FILENAME="consolidated/lora.safetensors"
    fi

    BEST_JSON="$EXP_DIR/best_checkpoint_${RUN_TS}.json"
    if uv run --project "$MOSHI_FT_REPO" python "$REPO_ROOT/scripts/select_best_checkpoint.py" \
        --mode lora \
        --metrics-jsonl "$EXP_CKPT_DIR/metrics.eval.jsonl" \
        --checkpoints-dir "$EXP_CKPT_DIR/checkpoints" \
        --checkpoint-filename "$CKPT_FILENAME" \
        --output-json "$BEST_JSON"; then

        BEST_CKPT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint_path'])" "$BEST_JSON")"
        MERGED_OUT="${MERGED_OUT:-$EXP_DIR/merged/${RUN_TS}/consolidated.safetensors}"
        mkdir -p "$(dirname "$MERGED_OUT")"

        if [[ "$SAVE_ADAPTERS" == "false" ]]; then
            echo "[exp] full fine-tuning checkpoint selected (no merge needed): $BEST_CKPT"
            cp "$BEST_CKPT" "$MERGED_OUT"
            echo "[exp] copied to: $MERGED_OUT"
        else
            echo "[exp] merging best LoRA checkpoint: $BEST_CKPT"
            uv run --project "$MOSHI_FT_REPO" python "$REPO_ROOT/scripts/merge_lora.py" \
                --lora-ckpt "$BEST_CKPT" \
                --out "$MERGED_OUT" \
                --hf-repo "$HF_REPO_ID"
        fi
        echo "[exp] auto-merge complete: $MERGED_OUT"

        # -----------------------------------------------------------------
        # 7) マージ済みモデルで Full-Duplex-Bench-JA 評価を自動実行する。
        #    SKIP_AUTO_EVAL=1 で無効化できる。
        # -----------------------------------------------------------------
        if [[ "${SKIP_AUTO_EVAL:-0}" != "1" ]]; then
            EVAL_MODEL_ID="$(printf '%s' "${EXP_NAME//\//_}_${RUN_TS}" | tr -cd 'A-Za-z0-9._-')"
            echo
            echo "[exp] running Full-Duplex-Bench-JA evaluation: model_id=$EVAL_MODEL_ID"
            if MODEL_ID="$EVAL_MODEL_ID" \
               RUN_ID="$EVAL_MODEL_ID" \
               MODEL_WEIGHT="$MERGED_OUT" \
               HF_REPO="$HF_REPO_ID" \
               bash "$REPO_ROOT/scripts/run_full_duplex_eval.sh"; then
                echo "[exp] auto-eval complete: $REPO_ROOT/eval_runs/full_duplex/$EVAL_MODEL_ID/benchmark_results/summary.json"
            else
                echo "[exp] WARN: auto-eval failed; merged model is still available at $MERGED_OUT" >&2
            fi
        else
            echo "[exp] SKIP_AUTO_EVAL=1: skipping automatic Full-Duplex-Bench evaluation"
        fi
    else
        echo "[exp] WARN: could not select a best checkpoint; skipping auto-merge and auto-eval" >&2
    fi
fi
