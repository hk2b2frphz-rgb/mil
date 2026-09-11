#!/usr/bin/env bash
# Shared helper: build the `uv` invocation for the KABURI-TTS environment.
#
# KABURI lives in its own uv project (torch >=2.10 cu128, python <3.12) next to
# this repo, so every caller needs the same three decisions. They live here so
# a fix lands on setup_kaburi_env.sh and the PBS jobs at once.
#
#   TLS      On a cluster behind a TLS-inspecting proxy, uv's bundled
#            certificate store does not know the proxy's CA and every download
#            dies with "invalid peer certificate: UnknownIssuer". --system-certs
#            (--native-tls on older uv; kaburi_tls_flag asks uv which name it
#            takes, since the old one now warns) makes uv use the machine's own
#            certificate store instead, which is where such a CA is installed.
#            It is a no-op elsewhere, so it is on by default; set
#            KABURI_NATIVE_TLS=0 to drop it.
#
#            If the CA is not in the system store either, point uv at it
#            directly instead -- uv honours both of these:
#              export SSL_CERT_FILE=/path/to/corp-ca.pem
#              export SSL_CERT_DIR=/etc/pki/tls/certs
#
#   Index    KABURI pins torch/torchaudio to https://download.pytorch.org/whl/cu128,
#            which redirects to download-r2.pytorch.org. A proxy that allows
#            the first host but not the second fails there no matter what the
#            certificates say. KABURI_TORCH_FROM_PYPI=1 adds --no-sources, so
#            the pin is ignored and torch comes from PyPI, whose wheels bundle
#            CUDA 12.8 anyway. The flag must be passed to every uv call for the
#            same environment -- `uv run` re-resolves otherwise -- which is the
#            other reason this helper exists.
#
#   Extras   pyopenjtalk is layered on top of the project environment. It is
#            not a KABURI dependency; scripts/alignment_words.py uses it to
#            split utterances into morphemes, and falls back to a regex if it
#            is missing (coarser word timings in the training data).
#
#   Codec    torchaudio 2.9+ reads and writes audio through torchcodec, whose
#            wheels are per CUDA major version. A mismatch with torch shows up
#            as "libnvrtc.so.<N>: cannot open shared object file" on the first
#            wav access, not at import. setup_kaburi_env.sh detects that and
#            repairs the venv, so `uv run` here passes --no-sync: an exact
#            re-sync would put the broken wheel back.
#
#   Path     KABURI's pyproject sets `package = false`, so `uv sync` installs
#            its dependencies and never the kaburi_tts / irodori_tts packages
#            themselves -- upstream runs its own scripts from inside that
#            checkout, where the interpreter puts them on sys.path for free.
#            Anything run from THIS repo gets "No module named kaburi_tts"
#            instead, so kaburi_uv_run exports PYTHONPATH with the checkout and
#            its scripts/ directory (where synth_with_predictor lives, which
#            the paper timing mode imports).
#
# Usage:
#   source scripts/kaburi_uv_env.sh
#   kaburi_uv_run KABURI_UV            # -> KABURI_UV=(uv run --project ... python)
#   "${KABURI_UV[@]}" scripts/generate_kaburi_tts_data.py ...
#
#   kaburi_uv_sync_args SYNC_ARGS      # -> flags for `uv sync --project ...`
#
# KABURI_PYTHON overrides everything: if it is set, the arrays become just that
# interpreter, for an environment built by hand outside uv.

kaburi_uv_flags() {
    # The internal name is prefixed so a caller passing its own "flags" array
    # does not collide with it -- a nameref onto a same-named local in the same
    # scope is a circular reference and bash refuses it.
    local -a __kaburi_flags=()
    if [[ "${KABURI_NATIVE_TLS:-1}" == "1" ]]; then
        __kaburi_flags+=(--native-tls)
    fi
    if [[ "${KABURI_TORCH_FROM_PYPI:-0}" == "1" ]]; then
        __kaburi_flags+=(--no-sources)
    fi
    local -n __kaburi_flags_out="$1"
    __kaburi_flags_out=(${__kaburi_flags[@]+"${__kaburi_flags[@]}"})
}

kaburi_export_pythonpath() {
    # uv passes the environment through, so this reaches both `uv run` and a
    # bare KABURI_PYTHON. Idempotent: re-sourcing must not stack duplicates.
    local want="$KABURI_REPO:$KABURI_REPO/scripts"
    case ":${PYTHONPATH:-}:" in
        *":$KABURI_REPO:"*) ;;
        *) export PYTHONPATH="${PYTHONPATH:+$want:$PYTHONPATH}"
           export PYTHONPATH="${PYTHONPATH:-$want}" ;;
    esac
}

kaburi_export_ldpath() {
    # torchcodec's shared objects look for the CUDA runtime libraries shipped
    # as nvidia-* wheels. They normally resolve through an RPATH relative to
    # site-packages; when a repair installs one afterwards that can miss, so
    # name the directories explicitly. Same trick as setup_gemma_runtime.sh.
    local venv_lib="$KABURI_REPO/.venv/lib"
    [[ -d "$venv_lib" ]] || return 0
    local dir
    while IFS= read -r dir; do
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$dir:"*) ;;
            *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$dir:$LD_LIBRARY_PATH}"
               export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$dir}" ;;
        esac
    done < <(find "$venv_lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | sort)
}

kaburi_uv_run() {
    local __out_name="$1"
    local -n __run_out="$__out_name"

    if [[ -n "${KABURI_PYTHON:-}" ]]; then
        if [[ ! -x "$KABURI_PYTHON" ]]; then
            echo "ERROR: KABURI_PYTHON is not executable: $KABURI_PYTHON" >&2
            return 1
        fi
        if [[ -n "${KABURI_REPO:-}" ]]; then
            kaburi_export_pythonpath
            kaburi_export_ldpath
        fi
        __run_out=("$KABURI_PYTHON")
        return 0
    fi

    if [[ -z "${KABURI_REPO:-}" ]]; then
        echo "ERROR: KABURI_REPO must be set before calling kaburi_uv_run." >&2
        return 1
    fi

    kaburi_export_pythonpath
    kaburi_export_ldpath

    local -a run_flags
    kaburi_uv_flags run_flags
    # --no-sync: setup_kaburi_env.sh owns this environment, and it may have had
    # to repair torchcodec on top of what the lockfile says. uv sync is exact,
    # so letting `uv run` re-sync would undo that repair on every call. Re-run
    # the setup script after changing dependencies.
    __run_out=(uv run --project "$KABURI_REPO" --no-sync
               ${run_flags[@]+"${run_flags[@]}"} --with pyopenjtalk python)
}

kaburi_uv_sync_args() {
    kaburi_uv_flags "$1"
}
