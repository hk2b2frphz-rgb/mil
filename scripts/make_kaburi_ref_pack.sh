#!/usr/bin/env bash
set -euo pipefail

# Build a KABURI-TTS speaker reference pack from the moshi clone voice this
# repo already uses, so the KABURI corpus keeps the same listener voice as the
# Qwen3 corpora (data/clone_examples/<id>, chosen by clone_voice_examples.py).
#
# Channel A = left = moshi (the listener).  Channel B = right = user.
#
#   MOSHI reference (channel A), in order of precedence:
#     MOSHI_REF_WAV           explicit wav
#     CLONE_OUT_DIR_MOSHI     clone_voice_examples.py output, resolved through
#                             scripts/resolve_clone_refs.py with REF_RANK /
#                             CLONE_MODE -- exactly what the Qwen3 jobs pass as
#                             --qwen-clone-ref-audio-moshi
#
#   USER reference (channel B), in order of precedence:
#     USER_REF_WAV            explicit wav
#     CLONE_OUT_DIR_USER      a second clone_voice_examples.py output
#     (default)               the bundled DEMO_F2 voice, decoded back to a wav
#                             from assets/test_refpack. It is an Irodori
#                             VoiceDesign voice -- nobody real, no license
#                             strings attached. The Qwen3 path has no user
#                             reference wav to reuse (it drives the user side
#                             with a CustomVoice preset), so there is nothing
#                             else to inherit here.
#
# Usage:
#   CLONE_OUT_DIR_MOSHI=$PWD/data/clone_examples/99999 \
#     bash scripts/make_kaburi_ref_pack.sh data/kaburi_ref_packs/99999_ref01
#
#   REFINE=1 ...    run kaburi's bootstrap refinement afterwards (needs a GPU).
#                   Only worth it for read-style references; the clone refs cut
#                   out of real dialogue are already in domain.
#
# The pack is small (two latents plus a template chunk) and deterministic, so
# rebuilding over an existing directory is harmless. It is skipped when the
# manifest is already there unless FORCE=1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-${KABURI_REF_PACK:-}}"
if [[ -z "$OUT_DIR" ]]; then
    echo "usage: bash scripts/make_kaburi_ref_pack.sh <out-dir>" >&2
    exit 1
fi

KABURI_REPO="${KABURI_REPO:-$REPO_ROOT/../kaburi-tts}"
if [[ ! -d "$KABURI_REPO/kaburi_tts" ]]; then
    echo "ERROR: KABURI-TTS checkout not found: $KABURI_REPO" >&2
    echo "Run scripts/setup_kaburi_env.sh first." >&2
    exit 1
fi

REF_RANK="${REF_RANK:-1}"
CLONE_MODE="${CLONE_MODE:-in-context}"
SPEAKER_A_LABEL="${SPEAKER_A_LABEL:-moshi_clone}"
SPEAKER_B_LABEL="${SPEAKER_B_LABEL:-user_ref}"
MAX_REF_SEC="${MAX_REF_SEC:-10.0}"
REF_PACK_DEVICE="${REF_PACK_DEVICE:-cpu}"

if [[ -s "$OUT_DIR/manifest.jsonl" && "${FORCE:-0}" != "1" ]]; then
    echo "[ref-pack] already built: $OUT_DIR (FORCE=1 to rebuild)"
    exit 0
fi

resolve_clone_wav() {  # resolve_clone_wav <clone-out-dir>
    uv run python scripts/resolve_clone_refs.py \
        --clone-out-dir "$1" \
        --rank "$REF_RANK" \
        --mode "$CLONE_MODE" \
        --field wav
}

if [[ -z "${MOSHI_REF_WAV:-}" ]]; then
    if [[ -z "${CLONE_OUT_DIR_MOSHI:-}" ]]; then
        echo "ERROR: set CLONE_OUT_DIR_MOSHI (or MOSHI_REF_WAV) to the moshi clone voice." >&2
        echo "The Qwen3 jobs use CLONE_OUT_DIR_MOSHI=\$PWD/data/clone_examples/99999." >&2
        exit 1
    fi
    MOSHI_REF_WAV="$(resolve_clone_wav "$CLONE_OUT_DIR_MOSHI")"
fi
if [[ ! -s "$MOSHI_REF_WAV" ]]; then
    echo "ERROR: moshi reference wav not found: $MOSHI_REF_WAV" >&2
    exit 1
fi

export KABURI_REPO
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaburi_uv_env.sh"
kaburi_uv_run KABURI_UV

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

NO_DENSIFY=()
if [[ -z "${USER_REF_WAV:-}" && -n "${CLONE_OUT_DIR_USER:-}" ]]; then
    USER_REF_WAV="$(resolve_clone_wav "$CLONE_OUT_DIR_USER")"
fi
if [[ -z "${USER_REF_WAV:-}" ]]; then
    BUNDLED_REF="${BUNDLED_USER_REF:-$KABURI_REPO/assets/test_refpack/refs/DEMO_F2_ref_000.pt}"
    if [[ ! -s "$BUNDLED_REF" ]]; then
        echo "ERROR: bundled user reference not found: $BUNDLED_REF" >&2
        echo "Pass USER_REF_WAV or CLONE_OUT_DIR_USER instead." >&2
        exit 1
    fi
    USER_REF_WAV="$WORK_DIR/user_ref.wav"
    SPEAKER_B_LABEL="${SPEAKER_B_LABEL_BUNDLED:-DEMO_F2}"
    echo "[ref-pack] no user reference given; decoding the bundled voice: $BUNDLED_REF"
    # Already a densified 10s reference latent, so re-densifying it would only
    # chew on decoder artefacts.
    NO_DENSIFY=(--no-densify)
    "${KABURI_UV[@]}" - "$BUNDLED_REF" "$USER_REF_WAV" <<'PY'
import sys
from pathlib import Path

import torch
import torchaudio
from irodori_tts.codec import DACVAECodec

ref_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
payload = torch.load(str(ref_path), map_location="cpu", weights_only=False)
latent = payload["latent"].float().unsqueeze(0)
codec = DACVAECodec.load(device="cpu", dtype=torch.float32)
wav = codec.decode_latent(latent)[0].float().cpu()
torchaudio.save(str(out_path), wav, int(codec.sample_rate), channels_first=True)
print(f"[ref-pack] decoded {payload.get('speaker')} -> {out_path} "
      f"({wav.shape[-1] / codec.sample_rate:.1f}s)")
PY
fi
if [[ ! -s "$USER_REF_WAV" ]]; then
    echo "ERROR: user reference wav not found: $USER_REF_WAV" >&2
    exit 1
fi

echo "[ref-pack] A=$SPEAKER_A_LABEL (left/moshi):  $MOSHI_REF_WAV"
echo "[ref-pack] B=$SPEAKER_B_LABEL (right/user):  $USER_REF_WAV"

BUILD_DIR="$OUT_DIR"
if [[ "${REFINE:-0}" == "1" ]]; then
    BUILD_DIR="$WORK_DIR/pack_raw"
fi
mkdir -p "$BUILD_DIR"

"${KABURI_UV[@]}" "$KABURI_REPO/scripts/make_ref_pack.py" \
    --ref-a "$MOSHI_REF_WAV" \
    --ref-b "$USER_REF_WAV" \
    --speaker-a "$SPEAKER_A_LABEL" \
    --speaker-b "$SPEAKER_B_LABEL" \
    --out-dir "$BUILD_DIR" \
    --device "$REF_PACK_DEVICE" \
    --max-ref-sec "$MAX_REF_SEC" \
    ${NO_DENSIFY[@]+"${NO_DENSIFY[@]}"}

if [[ "${REFINE:-0}" == "1" ]]; then
    echo "[ref-pack] bootstrap refinement (needs a GPU)"
    mkdir -p "$OUT_DIR"
    "${KABURI_UV[@]}" "$KABURI_REPO/scripts/refine_ref_pack.py" \
        --ref-pack "$BUILD_DIR" \
        --out-dir "$OUT_DIR" \
        --device "${REFINE_DEVICE:-cuda:0}"
fi

test -s "$OUT_DIR/manifest.jsonl"
echo "[ref-pack] done: $OUT_DIR"
cat "$OUT_DIR/manifest.jsonl"
