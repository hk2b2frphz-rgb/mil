#!/usr/bin/env bash
# Synthesize the screw PoC listening set on a single rented GPU.
#
# screw_poc/pbs/run_tts_smoke.pbs does the same work, but it wants PBS, the
# site proxy and the cluster's module environment.  When the question is only
# "does this sound right", none of that is available and none of it is needed:
# Qwen3-TTS-12Hz-1.7B loads through the qwen-tts package on any single GPU with
# 16GB, and the whole listening set takes minutes.
#
#   bash screw_poc/scripts/run_tts_listen.sh
#   SRC=screw_poc/artifacts/train_dialogues.jsonl N=3 bash screw_poc/scripts/run_tts_listen.sh
#
# These dialogues are strictly sequential -- nobody talks over anybody -- so
# unlike the listening-model path there is no --whole-utterance and no forced
# alignment here.  Turns are laid end to end with GAP_SEC between them.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf}"
cd "$(dirname "$0")/../.."

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

if [ ! -f screw_poc/artifacts/train_dialogues.jsonl ]; then
    echo "=== generating dialogues ==="
    "$VENV/bin/python" screw_poc/scripts/generate_dialogues.py >/dev/null
fi
if [ ! -f screw_poc/artifacts/listening_set.jsonl ]; then
    "$VENV/bin/python" screw_poc/scripts/pick_listening_set.py
fi

SRC="${SRC:-screw_poc/artifacts/listening_set.jsonl}"
OUT="${OUT:-screw_poc/artifacts/tts_listen}"
N="${N:-11}"

echo "=== synthesizing $N dialogues from $SRC ==="
"$VENV/bin/python" scripts/generate_qwen3_tts_data.py \
    --dialogues-jsonl "$SRC" \
    --out-dir "$OUT/training_set" \
    --num-dialogues "$N" \
    --model "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" \
    --speaker-moshi "Ono_Anna" \
    --speaker-user "Serena" \
    --user-speaker-pool "Serena,Sohee,Vivian,Dylan,Eric,Aiden,Uncle_Fu,Ryan" \
    --speaker-other "Dylan" \
    --speaker-background "Ryan" \
    --device "${TTS_DEVICE:-cuda}" \
    --dtype "${TTS_DTYPE:-float16}" \
    --lead-in-sec "${LEAD_IN_SEC:-0.3}" \
    --gap-sec "${GAP_SEC:-0.2}" \
    --no-opening-greeting \
    --no-emotion

echo "=== done ==="
find "$OUT" -name '*.wav' | head -20
