#!/usr/bin/env bash
# Shared helper: build the `uv run` prefix that a cascade driver must use when
# its TTS backend is kokoro.
#
# kokoro is deliberately NOT in the project dependencies (see
# docs/real_response_evaluation.md), so a plain `uv run` cannot import it and
# scripts/tts_comparison_backends.py raises
#   "Kokoro requires kokoro>=0.9.4 and misaki[ja] in its isolated env."
# The packages have to be layered in at call time.  Use the same fully
# isolated uv environment as the known-good Qwen-TTS PBS jobs
# (run_qwen_tts_whole_utterance_*_4gpu.pbs): importing Kokoro into the Moshi
# project environment lets that project's Transformers pin leak in and break
# Kokoro's custom ALBERT module. misaki[ja] pulls `unidic`,
# which ships code only -- the dictionary itself must be downloaded once before
# inference (same procedure as scripts/run_tts_comparison.pbs). uv caches the
# environment per --with set, so the dictionary survives into later calls.
#
# Usage:
#   source scripts/kokoro_uv_env.sh
#   kokoro_uv_run UV_RUN "$CASCADE_TTS_BACKEND" "[tag]"
#   "${UV_RUN[@]}" python eval/run_local_baseline_full_duplex.py ...
#
# For a non-kokoro backend the array is just (uv run), so callers can use it
# unconditionally.

kokoro_uv_run() {
    local __out_name="$1"
    local backend="$2"
    local tag="${3:-kokoro}"

    if [[ "$backend" != "kokoro" ]]; then
        eval "$__out_name=(uv run)"
        return 0
    fi

    # Match the working Qwen-TTS job exactly: a no-project environment with
    # the allocation's own CUDA torch/torchaudio versions.  The extra runtime
    # packages are what the real-response evaluator imports before it reaches
    # the Kokoro backend; faster-whisper is only used by cascade but harmless
    # for SpeechLLM and keeps this helper's contract uniform.
    local torch_version torchaudio_version
    torch_version="$(uv run python -c 'import torch; print(torch.__version__)')"
    torchaudio_version="$(uv run python -c 'import torchaudio; print(torchaudio.__version__)')"
    eval "$__out_name=(uv run --isolated --no-project --index 'pytorch-cu121=https://download.pytorch.org/whl/cu121' --with 'kokoro>=0.9.4' --with 'misaki[ja]' --with unidic --with pyopenjtalk --with numpy --with sphn --with soundfile --with uroman --with faster-whisper --with 'torch==$torch_version' --with 'torchaudio==$torchaudio_version')"
    local -n kokoro_uv="$__out_name"

    echo "[$tag] checking the kokoro Japanese dictionary (unidic)"
    if ! "${kokoro_uv[@]}" python -m unidic download; then
        echo "ERROR: unidic download failed. kokoro cannot run without the Japanese dictionary." >&2
        return 1
    fi

    # `from kokoro import KPipeline` imports Misaki lazily.  Check it here in
    # the *same* isolated environment used by the evaluator so a missing
    # Japanese extra, a broken fugashi wheel, or an unavailable dictionary is
    # reported before a long model run starts.
    echo "[$tag] preflighting kokoro + misaki[ja] import"
    if ! "${kokoro_uv[@]}" python -c 'import transformers; from kokoro import KPipeline; print(f"kokoro/misaki[ja] import: OK (transformers={transformers.__version__})")'; then
        echo "ERROR: Kokoro preflight failed in the uv isolated environment above." >&2
        echo "       Re-run the displayed uv command after fixing its exact ImportError." >&2
        return 1
    fi
}
