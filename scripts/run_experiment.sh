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
# split manifest は training_set/ の中に置く。
# sphn.dataset_jsonl は jsonl の "path" を **manifest のあるディレクトリ** を基点に
# 解決するので、ここを training_set/ にしないと sample_xxx.wav が見つからずに
# 空 dataset になり、最初の next(iter(dataset)) で StopIteration を出す。
EXP_TRAIN_MF="$EXP_TRAIN_SET/train.jsonl"
EXP_EVAL_MF="$EXP_TRAIN_SET/eval.jsonl"
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
NPROC="${NPROC:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _devs=""
    for ((i=0; i<NPROC; i++)); do
        _devs+="${_devs:+,}$i"
    done
    export CUDA_VISIBLE_DEVICES="$_devs"
fi

# 単 GPU は python 直起動 (torchrun + FSDP 単 GPU の segfault 回避)
# 複数 GPU は torchrun。USE_TORCHRUN=1 で単 GPU でも torchrun を強制可能。
USE_TORCHRUN="${USE_TORCHRUN:-auto}"
if [[ "$USE_TORCHRUN" == "auto" ]]; then
    if [[ "$NPROC" -eq 1 ]]; then USE_TORCHRUN=0; else USE_TORCHRUN=1; fi
fi

if [[ "$USE_TORCHRUN" == "1" ]]; then
    if command -v torchrun >/dev/null 2>&1; then
        LAUNCH_CMD=(torchrun --nproc-per-node "$NPROC" --master_port "${MASTER_PORT:-29500}" -m train "$RESOLVED_CONFIG")
    else
        LAUNCH_CMD=(uv run --project "$MOSHI_FT_REPO" torchrun --nproc-per-node "$NPROC" --master_port "${MASTER_PORT:-29500}" -m train "$RESOLVED_CONFIG")
    fi
    LAUNCH_DESC="torchrun (nproc=$NPROC)"
else
    # 単 GPU 直起動。torchrun が export するはずの分散 env を自前で渡す。
    export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
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

# ---------------------------------------------------------------------------
# 5) 起動。run.log にも tee。
# ---------------------------------------------------------------------------
(
    cd "$MOSHI_FT_REPO" && \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${LAUNCH_CMD[@]}"
) 2>&1 | tee "$EXP_LOG"
