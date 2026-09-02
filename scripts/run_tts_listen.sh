#!/usr/bin/env bash
# Synthesize a few dialogues on a single GPU, just to listen to them.
#
# The production TTS path (run_qwen_tts_vllm_*.pbs -> run_qwen_tts_vllm_common.sh)
# wants environment modules and a V100-patched vLLM-Omni across four GPUs, so it
# only runs on the cluster. That is a throughput path. For the question "does
# this sound right", generate_qwen3_tts_data.py can load Qwen3-TTS directly
# through the qwen-tts package, which needs nothing but torch -- slower per
# dialogue, and usable on any single GPU box (this was written on a rented
# A100). Nothing here replaces the production path; it shortens the loop
# between changing timing and hearing it.
#
#   bash scripts/run_tts_listen.sh
#   SRC=data/runs/<batch>/llm_dialogues/dialogues.jsonl N=5 bash scripts/run_tts_listen.sh
#
# --whole-utterance is the flag that matters: it synthesizes each speaker's
# turns as one utterance and recovers the boundaries with forced alignment,
# which is what lets a backchannel be laid over the speaker instead of queued
# after them. Without it you are listening to a different pipeline.
#
# CustomVoice by default, because reference audio for the clone lives on the
# cluster. The voice will not be the one we ship; the timing will be.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf}"
cd "$(dirname "$0")/.."

VENV="${TTS_VENV:-$HOME/.venv-tts}"
if [ ! -d "$VENV" ]; then
    echo "=== creating $VENV ==="
    uv venv "$VENV" --python 3.11
fi
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu129}"
# sphn writes the Moshi-format audio; without it every dialogue synthesizes
# fully and then dies on the way to disk.
uv pip install --python "$VENV/bin/python" --quiet \
    torch torchaudio qwen-tts soundfile numpy uroman sphn

SRC="${SRC:-data/runs/qwen_aizuchi_1000_normal_v11/llm_dialogues/dialogues.jsonl}"
OUT="${OUT:-data/runs/tts_listen}"
N="${N:-3}"

echo "=== synthesizing $N dialogues from $SRC ==="
"$VENV/bin/python" scripts/generate_qwen3_tts_data.py \
    --dialogues-jsonl "$SRC" \
    --out-dir "$OUT" \
    --num-dialogues "$N" \
    --tts-backend qwen3 \
    --qwen-voice-mode "${QWEN_VOICE_MODE:-customvoice}" \
    --whole-utterance \
    --auto-overlap-aizuchi \
    --gap-sec "${GAP_SEC:-0.2}" \
    --style-preset "${STYLE_PRESET:-counseling_neutral}" \
    --fa-fallback skip \
    --device cuda

echo "=== done ==="
find "$OUT" -name '*.wav' | head -20
