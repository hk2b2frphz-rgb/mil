#!/usr/bin/env bash
# End-to-end pipeline: ~1h of synthetic Japanese counseling audio → Moshi LoRA FT
#
# 想定環境: Linux GPU サーバー、リポジトリのルートで実行。
# Moshi 用と Gemma 用の uv 環境が既に sync 済みであること。
#
# 各ステップは独立に再実行できる。途中で止めたい場合は --steps を絞る。

set -euo pipefail

# 実験管理: 実行ごとに日付付きの run dir を作る。
# OUT_ROOT を明示指定した場合はそちらを優先する（再開・再現実行用）。
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-./data/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_ROOT/$RUN_ID}"
NUM_CASES="${NUM_CASES:-100}"
USE_CASES_PATH="$OUT_ROOT/use_cases.jsonl"
GEMMA_DIALOGUES_DIR="$OUT_ROOT/gemma_dialogues"
AUDIO_DIR="$OUT_ROOT/training_set"
LOG_DIR="$OUT_ROOT/logs"
FT_CONFIG="${FT_CONFIG:-./configs/moshi_lora_jp_loneliness.yaml}"
MOSHI_FT_REPO="${MOSHI_FT_REPO:-../moshi-finetune}"

STEPS="${STEPS:-use_cases,dialogues,audio,finetune}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

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
    log_info "Gemma で対話を ${NUM_CASES} 件生成: $GEMMA_DIALOGUES_DIR"
    log_info "  model: ${GEMMA_MODEL:-google/gemma-4-E2B-it}"
    uv run python scripts/generate_synthetic_moshi_training_data.py \
        --out-dir "$GEMMA_DIALOGUES_DIR" \
        --use-cases-jsonl "$USE_CASES_PATH" \
        --num-dialogues "$NUM_CASES" \
        --mode dialogues-only \
        --gemma-backend transformers-subprocess \
        --gemma-model "${GEMMA_MODEL:-google/gemma-4-E2B-it}" \
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
    if [[ ! -d "$MOSHI_FT_REPO" ]]; then
        log_warn "moshi-finetune repo not found at $MOSHI_FT_REPO"
        echo "  Clone it first: git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune"
        exit 1
    fi
    log_info "Moshi LoRA fine-tune を起動"
    log_info "  ft repo: $MOSHI_FT_REPO"
    log_info "  config:  $FT_CONFIG"
    log_info "  nproc:   ${NPROC:-1}"
    # manifest のパスが config に書かれているので、相対パスを fix する。
    # 上記 config はリポジトリルートから FT を起動する想定。
    cd "$MOSHI_FT_REPO"
    torchrun \
        --nproc-per-node "${NPROC:-1}" \
        --master_port "${MASTER_PORT:-29500}" \
        -m train "$(realpath -m "$OLDPWD/$FT_CONFIG")"
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
