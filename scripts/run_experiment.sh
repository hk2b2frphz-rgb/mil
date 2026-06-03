#!/usr/bin/env bash
# Launch one experiment from experiments/<EXP>/config.yaml.
#
# Usage:
#   bash scripts/run_experiment.sh <EXP_NAME> <SRC_RUN_DIR>
#
# Example:
#   bash scripts/run_experiment.sh exp001_lora_baseline ./data/runs/2026-06-02_130539
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
    echo "  例: bash $0 exp001_lora_baseline ./data/runs/2026-06-02_130539" >&2
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
EXP_TRAIN_MF="$EXP_DATA_DIR/train.jsonl"
EXP_EVAL_MF="$EXP_DATA_DIR/eval.jsonl"
EXP_CKPT_DIR="$EXP_DIR/checkpoints"
EXP_LOG="$EXP_DIR/run.log"
RESOLVED_CONFIG="$EXP_DIR/_resolved.yaml"

# repo 配下を絶対化
MOSHI_FT_REPO="${MOSHI_FT_REPO:-$REPO_ROOT/../moshi-finetune}"
MOSHI_FT_REPO="$(realpath -m "$MOSHI_FT_REPO")"

mkdir -p "$EXP_DATA_DIR"

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
rng = random.Random(0)
rng.shuffle(lines)
n_eval = max(1, len(lines) // 10)
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
# ---------------------------------------------------------------------------
echo "[exp] _resolved.yaml を生成"
python3 - "$CONFIG_SRC" "$RESOLVED_CONFIG" "$EXP_TRAIN_MF" "$EXP_EVAL_MF" "$EXP_CKPT_DIR" <<'PY'
import re, sys, pathlib
src, dst, train, evald, ckpt = map(pathlib.Path, sys.argv[1:])
text = src.read_text()

def sub_block_value(text, key, value):
    # data: 下の "  key: ..." を書き換える簡易置換
    return re.sub(rf'^(\s*{re.escape(key)}\s*:\s*).*$', rf'\1"{value}"', text, flags=re.M)

text = sub_block_value(text, "train_data", str(train))
text = sub_block_value(text, "eval_data",  str(evald))
text = sub_block_value(text, "run_dir",    str(ckpt))
dst.write_text(text)
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

# torchrun の解決
TORCHRUN_CMD=()
if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD=(torchrun)
else
    TORCHRUN_CMD=(uv run --project "$MOSHI_FT_REPO" torchrun)
fi

# CUDA_VISIBLE_DEVICES 既定値
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _nproc="${NPROC:-1}"
    _devs=""
    for ((i=0; i<_nproc; i++)); do
        _devs+="${_devs:+,}$i"
    done
    export CUDA_VISIBLE_DEVICES="$_devs"
fi

echo "[exp] launching torchrun"
echo "       exp:     $EXP_NAME"
echo "       config:  $RESOLVED_CONFIG"
echo "       train:   $EXP_TRAIN_MF"
echo "       eval:    $EXP_EVAL_MF"
echo "       ckpt:    $EXP_CKPT_DIR"
echo "       nproc:   ${NPROC:-1}"
echo "       CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "       log:     $EXP_LOG"

# ---------------------------------------------------------------------------
# 5) 起動。run.log にも tee。
# ---------------------------------------------------------------------------
(
    cd "$MOSHI_FT_REPO" && \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${TORCHRUN_CMD[@]}" \
        --nproc-per-node "${NPROC:-1}" \
        --master_port "${MASTER_PORT:-29500}" \
        -m train "$RESOLVED_CONFIG"
) 2>&1 | tee "$EXP_LOG"
