#!/usr/bin/env bash
set -euo pipefail

# 実データ応答評価のカスケード(ASR->LLM->TTS)版。scripts/run_real_eval.sh の
# 兄弟で、同じ MODEL_ID / RUN_ID / REAL_OUT_DIR の作法に従い、同じ
# benchmark_results/summary.json を書く。したがって同じ表に並べられる。
#
#   MODEL_ID=cascade_gemma2b bash scripts/run_real_cascade_eval.sh
#
# Moshi 系と同じテストデータ(人手アノテーション由来)を、同じ指標で測る。
# 合成 Full-Duplex-Bench-JA を走る scripts/run_full_duplex_cascade_eval.sh とは
# 別物なので混同しないこと。あちらはデータもプロトコルも違う。
#
# 土俵を揃えるうえで肝心な点:
#   応答音声は「入力を全部聞き終えてから、ASR+LLM+TTS の実処理が終わるまで」の
#   位置に置かれる(eval/run_local_baseline_full_duplex.py)。つまり応答速度に
#   カスケードの実際の処理時間が入る。Moshi 側は生成そのものが応答開始なので、
#   どちらも「音が鳴り始めた時刻」という同じ定義で測られる。
#
# 構造上の注意:
#   カスケードはターンが終わってから動くので、相槌はほぼ 0 件になる。これは
#   欠陥ではなく構造の差であり、相槌軸に出る差はそう解釈すること。
#
# OpenAI/Azure API は呼ばない(judge 入力を書き出すだけ)。

trap '
    status=$?
    echo "[real-cascade] ERROR: command failed (exit $status): $BASH_COMMAND" >&2
    case "$status" in
        137) echo "[real-cascade]   exit 137 = SIGKILL -- GPU/host OOM か walltime 超過。" >&2 ;;
        139) echo "[real-cascade]   exit 139 = SIGSEGV -- CUDA ドライバ/カーネルのクラッシュ。" >&2 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv >&2
' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/setup_proxy.sh"

MODEL_ID="${MODEL_ID:?Set MODEL_ID (e.g. cascade_gemma2b)}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi

RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/data/test_data/real_dialogue}"
REAL_DATASET_DIR="${REAL_DATASET_DIR:-$REPO_ROOT/data/eval_sets/real_response}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"
REAL_SEEDS="${REAL_SEEDS:-0}"
REAL_CONTEXT_SEC="${REAL_CONTEXT_SEC:-0}"
REAL_CASES_PER_TASK="${REAL_CASES_PER_TASK:-}"
REAL_MOS_BACKEND="${REAL_MOS_BACKEND:-utmos}"
REAL_MOS_DEVICE="${REAL_MOS_DEVICE:-cpu}"
REAL_MAX_LATENCY_SEC="${REAL_MAX_LATENCY_SEC:-}"
REBUILD_REAL_DATASET="${REBUILD_REAL_DATASET:-0}"

CASCADE_ASR_MODEL="${CASCADE_ASR_MODEL:-large-v3}"
CASCADE_ASR_DEVICE="${CASCADE_ASR_DEVICE:-cuda}"
CASCADE_ASR_COMPUTE_TYPE="${CASCADE_ASR_COMPUTE_TYPE:-float16}"
CASCADE_LLM_MODEL="${CASCADE_LLM_MODEL:-google/gemma-2-2b-it}"
# これは安全網であって長さの制御手段ではない。上限で切ると文の途中で終わり、
# その断片がそのまま音声になる。長さはシステムプロンプト側で指示している
# (1〜2文・60文字程度)。言い切らずに終わった応答は打ち切りの疑いとして
# 件数がログに出るので、そこが多いなら上限ではなくプロンプトを見直すこと。
CASCADE_LLM_MAX_NEW_TOKENS="${CASCADE_LLM_MAX_NEW_TOKENS:-200}"
CASCADE_TTS_BACKEND="${CASCADE_TTS_BACKEND:-kokoro}"
CASCADE_TTS_MODEL="${CASCADE_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
# kokoro は固定の日本語話者を使う(jf_/jm_ で始まる ID)。qwen3 のときは Serena。
if [[ "$CASCADE_TTS_BACKEND" == "kokoro" ]]; then
    CASCADE_TTS_SPEAKER="${CASCADE_TTS_SPEAKER:-jf_alpha}"
else
    CASCADE_TTS_SPEAKER="${CASCADE_TTS_SPEAKER:-Serena}"
fi
CASCADE_DEVICE="${CASCADE_DEVICE:-cuda}"
CASCADE_DTYPE="${CASCADE_DTYPE:-float16}"

export TORCHINDUCTOR_DISABLE=1
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export TORCH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1
export REAL_GIT_COMMIT="${REAL_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

mkdir -p "$REAL_OUT_DIR"
exec > >(tee "$REAL_OUT_DIR/run.log") 2>&1

echo "===== Real dialogue response evaluation (cascade) ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "dataset:      $REAL_DATASET_DIR"
echo "output:       $REAL_OUT_DIR"
echo "asr:          $CASCADE_ASR_MODEL"
echo "llm:          $CASCADE_LLM_MODEL"
if [[ "$CASCADE_TTS_BACKEND" == "kokoro" ]]; then
    echo "tts:          kokoro (hexgrad/Kokoro-82M) voice=$CASCADE_TTS_SPEAKER"
else
    echo "tts:          $CASCADE_TTS_BACKEND / $CASCADE_TTS_MODEL"
fi
echo "git_commit:   $REAL_GIT_COMMIT"
echo "started_at:   $(date -Iseconds)"
echo "======================================================="

# 1) データセット。Moshi 系と同じものを共有する。
if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "$REBUILD_REAL_DATASET" == "1" ]]; then
    build_args=(
        uv run python eval/build_real_test_dataset.py
        --test-data-dir "$TEST_DATA_DIR"
        --out-dir "$REAL_DATASET_DIR"
        --context-sec "$REAL_CONTEXT_SEC"
        --lead-in-sec "${REAL_LEAD_IN_SEC:-3.0}"
    )
    [[ "$REBUILD_REAL_DATASET" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
else
    echo "[real-cascade] 既存のデータセットを使います: $REAL_DATASET_DIR"
fi

# 2) 推論。ASR->LLM->TTS。応答の置き位置に実処理時間が入る。
#
# kokoro は本体の依存に入っていないので、その場で足す。misaki[ja] が引く
# unidic は辞書データを同梱しないため、推論の前に一度 download が要る
# (scripts/run_tts_comparison.pbs と同じ手順)。--with の組が同じなら uv は
# 環境をキャッシュするので、辞書は次の呼び出しにも残る。
UV_RUN=(uv run)
if [[ "$CASCADE_TTS_BACKEND" == "kokoro" ]]; then
    UV_RUN=(uv run --with "kokoro>=0.9.4" --with "misaki[ja]" --with unidic)
    echo "[real-cascade] kokoro の辞書データを確認します"
    "${UV_RUN[@]}" python -m unidic download || {
        echo "ERROR: unidic download に失敗しました。kokoro は日本語辞書なしでは動きません。" >&2
        exit 1
    }
fi

inference_args=(
    "${UV_RUN[@]}" python eval/run_local_baseline_full_duplex.py
    --system cascade
    --dataset-dir "$REAL_DATASET_DIR"
    --out-dir "$REAL_OUT_DIR/inference"
    --model-id "$MODEL_ID"
    --tasks all
    --seeds "$REAL_SEEDS"
    --overwrite
    --asr-model "$CASCADE_ASR_MODEL"
    --asr-device "$CASCADE_ASR_DEVICE"
    --asr-compute-type "$CASCADE_ASR_COMPUTE_TYPE"
    --llm-model "$CASCADE_LLM_MODEL"
    --llm-max-new-tokens "$CASCADE_LLM_MAX_NEW_TOKENS"
    --tts-backend "$CASCADE_TTS_BACKEND"
    --tts-model "$CASCADE_TTS_MODEL"
    --tts-speaker "$CASCADE_TTS_SPEAKER"
    --device "$CASCADE_DEVICE"
    --dtype "$CASCADE_DTYPE"
)
if [[ -n "$REAL_CASES_PER_TASK" ]]; then
    inference_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
fi
# 出力音声の再書き起こし(2 回目の ASR)を省く。応答テキストは LLM 側にあるので
# 内容は失われないが、時刻つきチャンクが無くなる。相槌の種類判定はチャンクの
# 文字を見るので、カスケードの相槌をきちんと採点したいときは省かないこと
# (構造上ほぼ 0 件なので、実害が出るのは稀)。
if [[ "${CASCADE_SKIP_OUTPUT_ALIGNMENT:-0}" == "1" ]]; then
    inference_args+=(--skip-output-alignment)
    echo "[real-cascade] 出力アライメント ASR を省きます"
fi
"${inference_args[@]}"

# 3) 評価。Moshi 系と同じスクリプト、同じ指標。
eval_args=(
    uv run python eval/evaluate_real_response.py
    --run-dir "$REAL_OUT_DIR/inference"
    --out-dir "$REAL_OUT_DIR/benchmark_results"
    --mos-backend "$REAL_MOS_BACKEND"
    --mos-device "$REAL_MOS_DEVICE"
)
if [[ -n "$REAL_MAX_LATENCY_SEC" ]]; then
    eval_args+=(--max-latency-sec "$REAL_MAX_LATENCY_SEC")
fi
"${eval_args[@]}"

# 3b) 相槌。カスケードはほぼ 0 件になるが、それ自体が結果なので必ず出す。
if [[ "${REAL_BACKCHANNEL:-1}" == "1" ]]; then
    uv run python eval/evaluate_real_dialogue_backchannel.py \
        --run-dir "$REAL_OUT_DIR/inference" \
        --out "$REAL_OUT_DIR/benchmark_results/backchannel.json" \
        --tolerance-sec "${REAL_BC_TOLERANCE_SEC:-1.0}"
fi

# 4) LLM-as-a-judge 入力。
uv run python eval/pack_real_dialogue_judge_input.py \
    --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$REAL_OUT_DIR/real_judge_input.jsonl"

echo "[real-cascade] summary:     $REAL_OUT_DIR/benchmark_results/summary.json"
echo "[real-cascade] backchannel: $REAL_OUT_DIR/benchmark_results/backchannel.json"
echo "[real-cascade] judge input: $REAL_OUT_DIR/real_judge_input.jsonl"
echo "finished_at: $(date -Iseconds)"
