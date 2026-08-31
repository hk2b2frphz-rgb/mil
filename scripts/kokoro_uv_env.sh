#!/usr/bin/env bash
# Shared helper: build the `uv run` prefix that a cascade driver must use when
# its TTS backend is kokoro.
#
# kokoro is deliberately NOT in the project dependencies (see
# docs/real_response_evaluation.md), so a plain `uv run` cannot import it and
# scripts/tts_comparison_backends.py raises
#   "Kokoro requires kokoro>=0.9.4 and misaki[ja] in its isolated env."
# The packages have to be layered in at call time. misaki[ja] pulls `unidic`,
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

    eval "$__out_name=(uv run --with 'kokoro>=0.9.4' --with 'misaki[ja]' --with unidic)"

    echo "[$tag] checking the kokoro Japanese dictionary (unidic)"
    if ! uv run --with "kokoro>=0.9.4" --with "misaki[ja]" --with unidic \
        python -m unidic download; then
        echo "ERROR: unidic download failed. kokoro cannot run without the Japanese dictionary." >&2
        return 1
    fi

    # `from kokoro import KPipeline` imports Misaki lazily.  Check it here in
    # the *same* uv overlay used by the evaluator so a missing Japanese extra,
    # a broken fugashi wheel, or an unavailable dictionary is reported before
    # a long model run starts.  Do not replace stderr with a generic message:
    # the underlying ImportError identifies the package/system dependency that
    # must be fixed on this particular HPC image.
    echo "[$tag] preflighting kokoro + misaki[ja] import"
    if ! uv run --with "kokoro>=0.9.4" --with "misaki[ja]" --with unidic \
        python -c 'from kokoro import KPipeline; print("kokoro/misaki[ja] import: OK")'; then
        echo "ERROR: Kokoro preflight failed in the uv isolated environment above." >&2
        echo "       Re-run the displayed uv command after fixing its exact ImportError." >&2
        return 1
    fi
}
