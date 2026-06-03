#!/usr/bin/env bash
# End-to-end pipeline: ~1h of synthetic Japanese counseling audio → Moshi LoRA FT
#
# 想定環境: Linux GPU サーバー、リポジトリのルートで実行。
# Moshi 用と Gemma 用の uv 環境が既に sync 済みであること。
#
# 各ステップは独立に再実行できる。途中で止めたい場合は --steps を絞る。

# bash 強制（sh で起動されると process substitution が壊れる）
if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: bash で実行してください: bash scripts/run_pipeline.sh" >&2
    exit 1
fi

set -euo pipefail

# リポジトリルートを基準にする（どこから呼ばれても動くように）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 実験管理: 実行ごとに日付付きの run dir を作る。
# OUT_ROOT を明示指定した場合はそちらを優先する（再開・再現実行用）。
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/data/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_ROOT/$RUN_ID}"
NUM_CASES="${NUM_CASES:-100}"
FT_CONFIG="${FT_CONFIG:-$REPO_ROOT/configs/moshi_lora_jp_loneliness.yaml}"
MOSHI_FT_REPO="${MOSHI_FT_REPO:-$REPO_ROOT/../moshi-finetune}"

STEPS="${STEPS:-use_cases,dialogues,audio,finetune}"

mkdir -p "$OUT_ROOT"

# 以降のパスは全て絶対化（後で cd しても壊れないように）
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"
USE_CASES_PATH="$OUT_ROOT/use_cases.jsonl"
GEMMA_DIALOGUES_DIR="$OUT_ROOT/gemma_dialogues"
AUDIO_DIR="$OUT_ROOT/training_set"
# FT_CONFIG / MOSHI_FT_REPO も絶対化（cd 後でも有効）
FT_CONFIG="$(realpath -m "$FT_CONFIG")"
MOSHI_FT_REPO_PARENT="$(realpath -m "$(dirname "$MOSHI_FT_REPO")")"
MOSHI_FT_REPO="$MOSHI_FT_REPO_PARENT/$(basename "$MOSHI_FT_REPO")"

# パイプライン全体のログ（標準出力/エラー両方を tee で保存）
PIPELINE_LOG="$LOG_DIR/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

# 実行メタ情報を残す（あとから「何をやったか」を辿れるように）
META_FILE="$OUT_ROOT/run_meta.txt"
{
    echo "run_id: $RUN_ID"
    echo "started_at: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "user: $(whoami)"
    echo "cwd: $(pwd)"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
    echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
    echo "git_status:"
    git status --short 2>/dev/null || echo "  n/a"
    echo "steps: $STEPS"
    echo "num_cases: $NUM_CASES"
    echo "ft_config: $FT_CONFIG"
    echo "env:"
    echo "  GEMMA_MODEL=${GEMMA_MODEL:-}"
    echo "  QWEN_TTS_MODEL=${QWEN_TTS_MODEL:-}"
    echo "  TTS_DEVICE=${TTS_DEVICE:-}"
    echo "  TTS_DTYPE=${TTS_DTYPE:-}"
    echo "  NPROC=${NPROC:-}"
} > "$META_FILE"

echo "===== run_id: $RUN_ID ====="
echo "out_root: $OUT_ROOT"
echo "log:      $PIPELINE_LOG"

# 全体経過の計測
PIPELINE_START_TS=$(date +%s)

# 終了時刻 + 全体所要時間を記録
trap '
    end_ts=$(date +%s)
    elapsed=$((end_ts - PIPELINE_START_TS))
    echo "ended_at: $(date -Iseconds)" >> "$META_FILE"
    echo "elapsed_sec: $elapsed" >> "$META_FILE"
    printf "\n===== pipeline elapsed: %dh%02dm%02ds =====\n" \
        $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
' EXIT

# 色付きヘッダ（ターミナルが対応していれば）
if [[ -t 1 ]]; then
    C_BLUE=$'\e[1;34m'; C_GREEN=$'\e[1;32m'; C_YELLOW=$'\e[1;33m'
    C_GRAY=$'\e[0;90m'; C_RESET=$'\e[0m'
else
    C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_GRAY=""; C_RESET=""
fi

log_info()  { echo "${C_BLUE}[$(date +%H:%M:%S)]${C_RESET} $*"; }
log_ok()    { echo "${C_GREEN}[$(date +%H:%M:%S)] ✓${C_RESET} $*"; }
log_skip()  { echo "${C_GRAY}[$(date +%H:%M:%S)] -${C_RESET} $*"; }
log_warn()  { echo "${C_YELLOW}[$(date +%H:%M:%S)] !${C_RESET} $*"; }

# 各 step は run_step / end_step で挟む
CURRENT_STEP=""
CURRENT_STEP_START=0
TOTAL_STEPS=$(echo "$STEPS" | awk -F',' '{print NF}')
STEP_INDEX=0

run_step() {
    local name="$1"
    STEP_INDEX=$((STEP_INDEX + 1))
    if [[ ",$STEPS," == *",$name,"* ]]; then
        CURRENT_STEP="$name"
        CURRENT_STEP_START=$(date +%s)
        echo
        echo "${C_BLUE}═══════════════════════════════════════════════════════${C_RESET}"
        log_info "step ${STEP_INDEX}/${TOTAL_STEPS}: ${C_YELLOW}${name}${C_RESET} 開始"
        echo "${C_BLUE}═══════════════════════════════════════════════════════${C_RESET}"
        return 0
    fi
    log_skip "step ${STEP_INDEX}/${TOTAL_STEPS}: ${name}（スキップ）"
    return 1
}

end_step() {
    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$((end_ts - CURRENT_STEP_START))
    log_ok "step ${CURRENT_STEP} 完了（$(printf '%dm%02ds' $((elapsed/60)) $((elapsed%60)))）"
}

# ファイル数・サイズの簡易レポート
report_path() {
    local label="$1" path="$2"
    if [[ -f "$path" ]]; then
        local lines size
        lines=$(wc -l < "$path" 2>/dev/null || echo "?")
        size=$(du -h "$path" 2>/dev/null | cut -f1)
        log_info "  → $label: $path  (${lines} lines, ${size})"
    elif [[ -d "$path" ]]; then
        local count size
        count=$(find "$path" -type f 2>/dev/null | wc -l)
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        log_info "  → $label: $path  (${count} files, ${size})"
    fi
}

# ---------------------------------------------------------------------------
# 1. use case 100 件を生成（軸の組み合わせから）
# ---------------------------------------------------------------------------
if run_step use_cases; then
    log_info "use case を ${NUM_CASES} 件生成: $USE_CASES_PATH"
    uv run python scripts/build_use_cases.py \
        --out-path "$USE_CASES_PATH" \
        --num "$NUM_CASES"
    report_path "use_cases" "$USE_CASES_PATH"
    end_step
fi

# ---------------------------------------------------------------------------
# 2. Gemma で対話 JSONL を生成（音声合成はしない）
# ---------------------------------------------------------------------------
if run_step dialogues; then
    # A100 なら bfloat16、V100 なら float16 を環境変数で指定可能
    # 例: GEMMA_DTYPE=float16 ./scripts/run_pipeline.sh
    GEMMA_DTYPE_VAL="${GEMMA_DTYPE:-bfloat16}"
    GEMMA_MAX_NEW_TOKENS_VAL="${GEMMA_MAX_NEW_TOKENS:-900}"
    log_info "Gemma で対話を ${NUM_CASES} 件生成: $GEMMA_DIALOGUES_DIR"
    log_info "  model:          ${GEMMA_MODEL:-google/gemma-4-E2B-it}"
    log_info "  dtype:          ${GEMMA_DTYPE_VAL}"
    log_info "  max_new_tokens: ${GEMMA_MAX_NEW_TOKENS_VAL}"
    uv run python scripts/generate_synthetic_moshi_training_data.py \
        --out-dir "$GEMMA_DIALOGUES_DIR" \
        --use-cases-jsonl "$USE_CASES_PATH" \
        --num-dialogues "$NUM_CASES" \
        --mode dialogues-only \
        --gemma-backend transformers-subprocess \
        --gemma-model "${GEMMA_MODEL:-google/gemma-4-E2B-it}" \
        --gemma-dtype "${GEMMA_DTYPE_VAL}" \
        --gemma-max-new-tokens "${GEMMA_MAX_NEW_TOKENS_VAL}" \
        --allow-template-fallback
    report_path "dialogues" "$GEMMA_DIALOGUES_DIR/dialogues.jsonl"
    end_step
fi

# ---------------------------------------------------------------------------
# 3. Qwen3-TTS で音声化（左=moshi 固定、右=user プールから巡回）
# ---------------------------------------------------------------------------
if run_step audio; then
    log_info "Qwen3-TTS で音声化: $AUDIO_DIR"
    log_info "  model:  ${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
    log_info "  device: ${TTS_DEVICE:-cuda} / dtype: ${TTS_DTYPE:-bfloat16}"
    uv run python scripts/generate_qwen3_tts_data.py \
        --out-dir "$AUDIO_DIR" \
        --dialogues-jsonl "$GEMMA_DIALOGUES_DIR/dialogues.jsonl" \
        --model "${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" \
        --device "${TTS_DEVICE:-cuda}" \
        --dtype "${TTS_DTYPE:-bfloat16}"
    report_path "audio" "$AUDIO_DIR"
    end_step
fi

# ---------------------------------------------------------------------------
# 4. Moshi-finetune を起動（LoRA）
# ---------------------------------------------------------------------------
if run_step finetune; then
    # 1) repo を用意（なければ clone）
    if [[ ! -d "$MOSHI_FT_REPO" ]]; then
        log_warn "moshi-finetune repo not found at $MOSHI_FT_REPO — clone します"
        mkdir -p "$MOSHI_FT_REPO_PARENT"
        git clone https://github.com/kyutai-labs/moshi-finetune.git "$MOSHI_FT_REPO"
    fi

    # 2) 依存を用意（.venv が無ければ uv sync）
    if [[ ! -d "$MOSHI_FT_REPO/.venv" ]]; then
        # openai-whisper 20240930 の sdist は build-system.requires が
        # 不完全で、uv の隔離ビルド環境で安定して落ちる。setuptools/wheel を
        # extra-build-dependencies で渡しても setup.py が他の依存を
        # import するため救済できない。
        # whisper_timestamped 経由でしか入らず、annotate.py 専用 (train.py は
        # 不要) なので、学習を通すために依存から落とす。
        if grep -q '"whisper_timestamped"' "$MOSHI_FT_REPO/pyproject.toml"; then
            log_info "moshi-finetune の pyproject から whisper_timestamped を除外（学習には不要、annotate.py 専用）"
            sed -i.bak '/"whisper_timestamped"/d' "$MOSHI_FT_REPO/pyproject.toml"
            rm -f "$MOSHI_FT_REPO/uv.lock"
        fi

        # sphn==0.1.12 の pin は moshi 本体 (sphn>=0.2.0,<0.3.0) と衝突する。
        # 緩めて moshi に合わせる。
        if grep -q '"sphn==0.1.12"' "$MOSHI_FT_REPO/pyproject.toml"; then
            log_info "sphn の pin を moshi 本体に合わせて緩和（==0.1.12 → >=0.2.0,<0.3.0）"
            sed -i.bak3 's/"sphn==0.1.12"/"sphn>=0.2.0,<0.3.0"/' "$MOSHI_FT_REPO/pyproject.toml"
            rm -f "$MOSHI_FT_REPO/uv.lock"
        fi
        # 過去に追加した extra-build-dependencies ブロックを撤去（効かなかった）
        if grep -q '\[tool.uv.extra-build-dependencies\]' "$MOSHI_FT_REPO/pyproject.toml"; then
            log_info "古い extra-build-dependencies ブロックを削除"
            sed -i.bak2 '/^\[tool\.uv\.extra-build-dependencies\]/,$d' "$MOSHI_FT_REPO/pyproject.toml"
        fi
        rm -f "$MOSHI_FT_REPO/uv.lock"

        # 旧式 setup.py が pkg_resources を import するので、隔離ビルドを切って
        # venv に setuptools を入れた状態で sync する（uv 自身が勧める方法）。
        log_info "venv を作成し setuptools/wheel/pip を先入れ"
        ( cd "$MOSHI_FT_REPO" && uv venv && uv pip install --upgrade pip setuptools wheel )
        log_info "moshi-finetune の依存を uv sync --no-build-isolation で準備"
        ( cd "$MOSHI_FT_REPO" && uv sync --no-build-isolation )
    fi

    # interleaver.py への防御パッチ: 日本語 Moshi (llm-jp/llm-jp-moshi-v1) の
    # tokenizer は encode("\n") が空 list を返すことがあり、`[-1]` で
    # IndexError になる。空ならスペースを fallback にする。
    INTERLEAVER="$MOSHI_FT_REPO/finetune/data/interleaver.py"
    if [[ -f "$INTERLEAVER" ]] && ! grep -q 'AUTO_PATCH_NL_FALLBACK' "$INTERLEAVER"; then
        log_info "interleaver.py に nl_piece fallback パッチを適用"
        python3 - "$INTERLEAVER" <<'PY'
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

    # 3) torchrun を解決: PATH にあれば直接、無ければ uv run 経由
    TORCHRUN_CMD=()
    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN_CMD=(torchrun)
    else
        log_info "torchrun が PATH に無いため uv run --project '$MOSHI_FT_REPO' 経由で実行"
        TORCHRUN_CMD=(uv run --project "$MOSHI_FT_REPO" torchrun)
    fi

    # moshi-finetune の train.py は os.environ["CUDA_VISIBLE_DEVICES"] を
    # 直読みするので、未設定だと KeyError になる。
    # 未指定なら nproc に合わせて 0..N-1 をデフォルトにする。
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        _nproc="${NPROC:-1}"
        _devs=""
        for ((i=0; i<_nproc; i++)); do
            _devs+="${_devs:+,}$i"
        done
        export CUDA_VISIBLE_DEVICES="$_devs"
        log_info "CUDA_VISIBLE_DEVICES 未設定 → '$CUDA_VISIBLE_DEVICES' に自動設定"
    fi

    # 元 config はリポジトリ固定のパスを持っているので、この run の実体に
    # 合わせた一時コピーを作って渡す（train_data / run_dir を patch）。
    RUN_FT_CONFIG="$OUT_ROOT/ft_config.yaml"
    RUN_CKPT_DIR="$OUT_ROOT/checkpoints"
    TRAIN_MANIFEST="$AUDIO_DIR/synthetic_moshi_train.jsonl"
    if [[ ! -f "$TRAIN_MANIFEST" ]]; then
        log_warn "train manifest が見つかりません: $TRAIN_MANIFEST"
        log_warn "  audio ステップが完了している必要があります。"
        exit 1
    fi
    # train.py が run_dir を自前で作るので、こちらでは作らない。
    # train_data / run_dir を実パスに書き換え、overwrite_run_dir=true を
    # 末尾に追加（既存があっても再実行で詰まらないように）。
    python3 - "$FT_CONFIG" "$RUN_FT_CONFIG" "$TRAIN_MANIFEST" "$RUN_CKPT_DIR" <<'PY'
import re, sys, pathlib
src, dst, manifest, ckpt = sys.argv[1:]
text = pathlib.Path(src).read_text()
text = re.sub(r'^(\s*train_data:\s*).*$', rf'\1"{manifest}"', text, flags=re.M)
text = re.sub(r'^(\s*run_dir:\s*).*$',    rf'\1"{ckpt}"',    text, flags=re.M)
if re.search(r'^\s*overwrite_run_dir\s*:', text, re.M):
    text = re.sub(r'^(\s*overwrite_run_dir\s*:\s*).*$', r'\1true', text, flags=re.M)
else:
    text = text.rstrip() + "\n\n# auto-added by run_pipeline.sh\noverwrite_run_dir: true\n"
pathlib.Path(dst).write_text(text)
PY

    log_info "Moshi LoRA fine-tune を起動"
    log_info "  ft repo:   $MOSHI_FT_REPO"
    log_info "  config:    $RUN_FT_CONFIG  (元: $FT_CONFIG)"
    log_info "  manifest:  $TRAIN_MANIFEST"
    log_info "  ckpt:      $RUN_CKPT_DIR"
    log_info "  nproc:     ${NPROC:-1}"
    log_info "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    ( cd "$MOSHI_FT_REPO" && \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "${TORCHRUN_CMD[@]}" \
            --nproc-per-node "${NPROC:-1}" \
            --master_port "${MASTER_PORT:-29500}" \
            -m train "$RUN_FT_CONFIG" )
    end_step
fi

echo
echo "${C_GREEN}═══════════════════════════════════════════════════════${C_RESET}"
log_ok "pipeline finished"
echo "  out_root: $OUT_ROOT"
echo "  audio:    $AUDIO_DIR"
echo "  meta:     $META_FILE"
echo "  log:      $PIPELINE_LOG"
echo "  ft cfg:   $(grep run_dir "$FT_CONFIG" || true)"
