#!/usr/bin/env bash
# End-to-end pipeline: ~1h of synthetic Japanese counseling audio → Moshi LoRA FT
#
# 想定環境: Linux GPU サーバー、リポジトリのルートで実行。
# Moshi 用と Gemma 用の uv 環境が既に sync 済みであること。
#
# 各ステップは独立に再実行できる。途中で止めたい場合は --steps を絞る。

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./data/v1}"
NUM_CASES="${NUM_CASES:-100}"
USE_CASES_PATH="$OUT_ROOT/use_cases.jsonl"
GEMMA_DIALOGUES_DIR="$OUT_ROOT/gemma_dialogues"
AUDIO_DIR="$OUT_ROOT/training_set"
FT_CONFIG="${FT_CONFIG:-./configs/moshi_lora_jp_loneliness.yaml}"
MOSHI_FT_REPO="${MOSHI_FT_REPO:-../moshi-finetune}"

STEPS="${STEPS:-use_cases,dialogues,audio,finetune}"

run_step() {
    local name="$1"
    if [[ ",$STEPS," == *",$name,"* ]]; then
        echo
        echo "===== step: $name ====="
        return 0
    fi
    echo "(skipping step: $name)"
    return 1
}

mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------------------
# 1. use case 100 件を生成（軸の組み合わせから）
# ---------------------------------------------------------------------------
if run_step use_cases; then
    python scripts/build_use_cases.py \
        --out-path "$USE_CASES_PATH" \
        --num "$NUM_CASES"
fi

# ---------------------------------------------------------------------------
# 2. Gemma で対話 JSONL を生成（音声合成はしない）
# ---------------------------------------------------------------------------
if run_step dialogues; then
    uv run python scripts/generate_synthetic_moshi_training_data.py \
        --out-dir "$GEMMA_DIALOGUES_DIR" \
        --use-cases-jsonl "$USE_CASES_PATH" \
        --num-dialogues "$NUM_CASES" \
        --mode dialogues-only \
        --gemma-backend transformers-subprocess \
        --gemma-model "${GEMMA_MODEL:-google/gemma-4-E2B-it}" \
        --allow-template-fallback
fi

# ---------------------------------------------------------------------------
# 3. Qwen3-TTS で音声化（左=moshi 固定、右=user プールから巡回）
# ---------------------------------------------------------------------------
if run_step audio; then
    uv run python scripts/generate_qwen3_tts_data.py \
        --out-dir "$AUDIO_DIR" \
        --dialogues-jsonl "$GEMMA_DIALOGUES_DIR/dialogues.jsonl" \
        --model "${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" \
        --device "${TTS_DEVICE:-cuda}" \
        --dtype "${TTS_DTYPE:-bfloat16}"
fi

# ---------------------------------------------------------------------------
# 4. Moshi-finetune を起動（LoRA）
# ---------------------------------------------------------------------------
if run_step finetune; then
    if [[ ! -d "$MOSHI_FT_REPO" ]]; then
        echo "ERROR: moshi-finetune repo not found at $MOSHI_FT_REPO"
        echo "Clone it first: git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune"
        exit 1
    fi
    # manifest のパスが config に書かれているので、相対パスを fix する。
    # 上記 config はリポジトリルートから FT を起動する想定。
    cd "$MOSHI_FT_REPO"
    torchrun \
        --nproc-per-node "${NPROC:-1}" \
        --master_port "${MASTER_PORT:-29500}" \
        -m train "$(realpath -m "$OLDPWD/$FT_CONFIG")"
fi

echo
echo "pipeline finished. audio: $AUDIO_DIR, ft run_dir: $(grep run_dir "$FT_CONFIG" || true)"
