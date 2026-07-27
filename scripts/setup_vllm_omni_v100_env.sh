#!/usr/bin/env bash
# Build the isolated Python 3.12 runtime used by the V100 Qwen3-TTS jobs.
#
# PyTorch's cu128/cu129 wheels no longer contain Volta (sm70) kernels.  Install
# the cu126 wheel explicitly, verify sm70 support, and build vLLM against that
# PyTorch/CUDA combination instead of using vLLM's prebuilt cu129 wheel.
#
# Every torch pin below carries the CUDA local version segment (2.11.0+cu126).
# PyPI's torch 2.11.0 is a CUDA 13 build published under the *same* version
# number, so a bare "torch==2.11.0" pin lets any later resolution step swap the
# verified cu126/sm70 wheel for the cu130 one. That is not hypothetical: it left
# torch on cu130 while torchaudio and the source-built vLLM stayed on cu126, and
# the mismatch only surfaced when the compiled vLLM extension failed to load.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PWD/.venv-vllm-omni}"
# 0.22.0 is the first vllm-omni whose Qwen3-TTS talker matches the current
# CustomVoice/Base model generation (0.21.0rc1's talker predates it and emits
# out-of-range codec ids), and it adds the Base voice-clone task
# (ref_audio/ref_text/x_vector_only_mode). vllm 0.22.0 still pins
# torch==2.11.0, so the verified cu126/sm70 stack is unchanged.
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$PWD/.vendor/vllm-v${VLLM_VERSION}}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.22.0}"
VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-cuda12.6_cudnn9.7.1_nccl2.24.3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
# Wheel tag of that index ("cu126"), used as the local version segment on every
# torch pin and to derive the CUDA runtime version we assert against.
CUDA_WHEEL_TAG="${CUDA_WHEEL_TAG:-$(basename "$PYTORCH_INDEX_URL")}"
# cu126 -> 12.6, cu130 -> 13.0. This is what torch.version.cuda reports.
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-$(echo "$CUDA_WHEEL_TAG" | sed -E 's/^cu([0-9]+)([0-9])$/\1.\2/')}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
# vllm-omni's bundled Qwen3-TTS code calls create_causal_mask(input_embeds=...).
# transformers 5.x renamed that kwarg to inputs_embeds, so the newest transformers
# vllm pulls in breaks Stage1. Pin to the last version that still uses
# input_embeds (verified importing vllm + vllm-omni).
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
# Default the compile parallelism to the node's core count. The old fixed
# default of 4 made a healthy build look stalled for hours under uv's silent
# "Preparing packages (0/1)" line. Override MAX_JOBS to cap it if memory-bound.
DEFAULT_MAX_JOBS="$( { command -v nproc >/dev/null 2>&1 && nproc; } || echo 4 )"
MAX_JOBS="${MAX_JOBS:-$DEFAULT_MAX_JOBS}"
NVCC_THREADS="${NVCC_THREADS:-2}"
VLLM_PYTHON="$VLLM_ENV_DIR/bin/python"
# Stream the vLLM source-build progress here. On the compute node this file
# lives on the shared filesystem, so `tail -f` from a login node shows the
# ninja/cmake progress even when you cannot open a second shell on the node.
VLLM_BUILD_LOG="${VLLM_BUILD_LOG:-$PWD/vllm_build.log}"
# Persistent build/cache root on the shared filesystem. PBS sets TMPDIR to a
# per-job dir (/var/tmp/pbs.<jobid>) that is wiped at job end, so a build that
# lands there recompiles from scratch every submit. Pin the CMake build dir and
# a ccache store here instead, so the sm70 CUDA objects survive across jobs.
VLLM_BUILD_ROOT="${VLLM_BUILD_ROOT:-$PWD/.vllm-build}"

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$init" ]]; then
            # shellcheck source=/dev/null
            source "$init"
            break
        fi
    done
fi
command -v module >/dev/null 2>&1 || {
    echo "ERROR: environment-modules is unavailable" >&2
    exit 1
}
module load "$VLLM_CUDA_MODULE"

command -v nvcc >/dev/null 2>&1 || {
    echo "ERROR: $VLLM_CUDA_MODULE did not provide nvcc" >&2
    exit 1
}
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required" >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required to build vLLM" >&2
    exit 1
}

NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n 1)"
if [[ "$NVCC_RELEASE" != "$EXPECTED_TORCH_CUDA"* ]]; then
    echo "ERROR: expected CUDA $EXPECTED_TORCH_CUDA from $VLLM_CUDA_MODULE, got ${NVCC_RELEASE:-unknown}" >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS NVCC_THREADS

# The first torch install is not the only step that resolves torch: the vLLM
# source build and the vllm-omni install both pull it in transitively. Those
# steps used to run against PyPI only, where "2.11.0+cu126" does not exist at
# all, so they resolved the cu130 wheel instead. Make the CUDA wheel index
# visible to every uv/pip invocation below, and tell uv to look at all indexes
# (its default first-index strategy would stop at PyPI and never see the +cu126
# wheel). Both indexes are trusted, so best-match across them is safe here.
export UV_EXTRA_INDEX_URL="$PYTORCH_INDEX_URL"
export UV_INDEX_STRATEGY="unsafe-best-match"
export PIP_EXTRA_INDEX_URL="$PYTORCH_INDEX_URL"

# Assert the whole torch stack is still the CUDA-wheel build we verified. Called
# after every step that can resolve torch so a swap is attributed to the step
# that caused it, instead of surfacing much later as an "undefined symbol" when
# the source-built vLLM extension is finally imported.
verify_torch_stack() {
    STAGE="$1" \
    EXPECT_TORCH="$TORCH_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_TORCHAUDIO="$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_TORCHVISION="$TORCHVISION_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_CUDA="$EXPECTED_TORCH_CUDA" \
    "$VLLM_PYTHON" - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

stage = os.environ["STAGE"]
expected = {
    "torch": os.environ["EXPECT_TORCH"],
    "torchaudio": os.environ["EXPECT_TORCHAUDIO"],
    "torchvision": os.environ["EXPECT_TORCHVISION"],
}

# Read versions from the installed distribution metadata rather than by
# importing: if torch was swapped, importing torchaudio/torchvision (still
# linked against the previous torch) can fail before we get to report why.
problems = []
for dist, want in expected.items():
    try:
        got = version(dist)
    except PackageNotFoundError:
        problems.append(f"{dist}: not installed (expected {want})")
        continue
    print(f"[{stage}] {dist}: {got}")
    if got != want:
        problems.append(f"{dist}: {got} != {want}")

import torch

print(f"[{stage}] torch CUDA runtime: {torch.version.cuda}")
if torch.version.cuda != os.environ["EXPECT_CUDA"]:
    problems.append(f"torch.version.cuda: {torch.version.cuda} != {os.environ['EXPECT_CUDA']}")
arches = torch.cuda.get_arch_list()
print(f"[{stage}] compiled CUDA arches: {arches}")
if "sm_70" not in arches:
    problems.append("installed torch has no V100 sm70 kernels")
if not torch.cuda.is_available():
    problems.append("CUDA is unavailable on this node")
else:
    print(f"[{stage}] gpu: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")

if problems:
    raise SystemExit(
        f"{stage}: the pinned torch stack was replaced:\n  "
        + "\n  ".join(problems)
        + "\nA bare torch==<version> pin also matches PyPI's CUDA 13 wheel of the "
        "same version. Keep the +cuXXX local segment on every pin and make the "
        "PyTorch index reachable from the install step above."
    )
PY
}

echo "CUDA module: $VLLM_CUDA_MODULE"
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(command -v nvcc) (release $NVCC_RELEASE)"
echo "PyTorch index: $PYTORCH_INDEX_URL (wheel tag $CUDA_WHEEL_TAG, CUDA $EXPECTED_TORCH_CUDA)"
echo "PyTorch pins: torch==$TORCH_VERSION+$CUDA_WHEEL_TAG torchaudio==$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG torchvision==$TORCHVISION_VERSION+$CUDA_WHEEL_TAG"
echo "vLLM source: $VLLM_SRC_DIR (v$VLLM_VERSION)"
echo "CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "Build parallelism: MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"
echo "Build log: $VLLM_BUILD_LOG (tail -f from a login node to watch progress)"

# Fast resume: if a previous run already produced a complete, valid env, do
# nothing. Re-submitting the (long) build job is then cheap. FRESH=1 forces a
# full clean rebuild.
# The torch-stack half of this check is deliberately the same assertion the
# build steps use, so an env whose torch was swapped is never treated as valid.
if [[ "${FRESH:-0}" != "1" && -x "$VLLM_PYTHON" ]] && \
   verify_torch_stack "resume check" >/dev/null 2>&1 && \
   EXPECT_TRANSFORMERS="$TRANSFORMERS_VERSION" \
   "$VLLM_PYTHON" - >/dev/null 2>&1 <<'PY'
import os, transformers, vllm, vllm_omni  # noqa: F401
assert transformers.__version__ == os.environ["EXPECT_TRANSFORMERS"], transformers.__version__
PY
then
    echo "vLLM-Omni environment already complete and valid: $VLLM_ENV_DIR"
    echo "Nothing to do. Set FRESH=1 to rebuild from scratch."
    exit 0
fi

# Recreate the venv only for a fresh build or when it is missing/broken.
# Reusing a healthy venv on resume keeps the torch install and, crucially, lets
# the vLLM build reuse ninja's compiled objects in the source tree rather than
# recompiling from scratch. --clear guarantees a clean slate when we do rebuild;
# a stale env may hold CPU-only or cu129 PyTorch, or a half-built vLLM.
if [[ "${FRESH:-0}" == "1" || ! -x "$VLLM_PYTHON" ]]; then
    uv venv --python 3.12 --seed --clear "$VLLM_ENV_DIR"
else
    echo "Reusing existing venv: $VLLM_ENV_DIR (set FRESH=1 to recreate)"
fi

# Only touch torch when it is actually wrong. --reinstall is intentional when we
# do (an earlier setup may have left CPU-only, cu129, or PyPI's cu130 PyTorch
# here), but it rewrites every torch header and library. ninja decides staleness
# by mtime, so an unconditional reinstall marks essentially every vLLM CUDA
# object out of date and turns a resume into a full recompile. Skipping the
# no-op case keeps later re-submits genuinely incremental.
if verify_torch_stack "existing torch" >/dev/null 2>&1; then
    echo "torch stack already correct; skipping reinstall to keep the vLLM build incremental"
    verify_torch_stack "existing torch"
else
    uv pip install --python "$VLLM_PYTHON" \
        --reinstall \
        --index-url "$PYTORCH_INDEX_URL" \
        "torch==$TORCH_VERSION+$CUDA_WHEEL_TAG" \
        "torchaudio==$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG" \
        "torchvision==$TORCHVISION_VERSION+$CUDA_WHEEL_TAG"

    verify_torch_stack "torch install"
fi

mkdir -p "$(dirname "$VLLM_SRC_DIR")"
if [[ ! -e "$VLLM_SRC_DIR" ]]; then
    git clone --branch "v$VLLM_VERSION" --depth 1 \
        https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
elif [[ ! -d "$VLLM_SRC_DIR/.git" ]]; then
    echo "ERROR: $VLLM_SRC_DIR exists but is not a vLLM git checkout" >&2
    exit 1
fi

VLLM_SOURCE_TAG="$(git -C "$VLLM_SRC_DIR" describe --tags --exact-match 2>/dev/null || true)"
if [[ "$VLLM_SOURCE_TAG" != "v$VLLM_VERSION" ]]; then
    echo "ERROR: $VLLM_SRC_DIR is at ${VLLM_SOURCE_TAG:-an untagged commit}, expected v$VLLM_VERSION" >&2
    echo "Use an empty VLLM_SRC_DIR for the requested version." >&2
    exit 1
fi

# Pin the whole torch stack so no later install step can swap it. vLLM (and
# vllm-omni) still declare torch as a dependency; without this, pip/uv resolve
# to the newest torch (e.g. 2.13.0) during `pip install -e .`, uninstall the
# verified cu126/sm70 2.11.0, and then collide with the pinned torchvision.
#
# The +$CUDA_WHEEL_TAG local segment is the load-bearing part. Without it the
# constraint is satisfied by PyPI's CUDA 13 wheel of the very same version, so
# the constraint file looks respected while torch is silently swapped.
TORCH_CONSTRAINTS="$VLLM_ENV_DIR/torch-constraints.txt"
cat > "$TORCH_CONSTRAINTS" <<EOF
torch==$TORCH_VERSION+$CUDA_WHEEL_TAG
torchaudio==$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG
torchvision==$TORCHVISION_VERSION+$CUDA_WHEEL_TAG
EOF
echo "torch constraints ($TORCH_CONSTRAINTS):"
sed 's/^/  /' "$TORCH_CONSTRAINTS"

(
    cd "$VLLM_SRC_DIR"
    # Remove only torch/torchaudio/torchvision pins. The source build must use
    # the already verified cu126 PyTorch rather than resolving another wheel.
    "$VLLM_PYTHON" use_existing_torch.py --prefix
    uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" \
        -r requirements/build/cuda.txt
    verify_torch_stack "vLLM build requirements"
    uv pip uninstall --python "$VLLM_PYTHON" vllm >/dev/null 2>&1 || true
    # vLLM needs CMake >= 3.26, but the node's system CMake is older (e.g.
    # 3.22.1) and the build picks it up off PATH. Install a modern CMake (and
    # ninja) into the venv and put the venv's bin first on PATH so the build
    # uses them instead of the system CMake. Cap below 4.0 to stay on a range
    # this vLLM version is known to configure with.
    uv pip install --python "$VLLM_PYTHON" "cmake>=3.26,<4" ninja
    export PATH="$VLLM_ENV_DIR/bin:$PATH"
    echo "Using CMake: $(command -v cmake) ($(cmake --version | head -n1))"

    # Keep the CUDA build off the per-job TMPDIR so it persists and rebuilds
    # incrementally. CMAKE_BUILD_DIR is read by vLLM's build; a stable path here
    # means ninja reuses already-compiled sm70 objects on the next submit.
    #
    # Prefer a build directory that survived a previous successful compile so its
    # objects are reused instead of recompiled. vLLM's default build_temp lands
    # under the source tree; if a configured build (CMakeCache.txt) is still
    # there, point at it. Otherwise use the persistent shared-FS location. Both
    # live outside the per-job TMPDIR, so future submits stay incremental.
    if [[ -z "${CMAKE_BUILD_DIR:-}" ]]; then
        # Guard the search: a missing build/ makes find exit non-zero, and the
        # head pipe can SIGPIPE it — either aborts the script under pipefail.
        prev_build=""
        if [[ -d "$VLLM_SRC_DIR/build" ]]; then
            prev_build="$(find "$VLLM_SRC_DIR/build" -maxdepth 3 -name CMakeCache.txt \
                -printf '%h\n' 2>/dev/null | head -n1 || true)"
        fi
        if [[ -n "$prev_build" ]]; then
            export CMAKE_BUILD_DIR="$prev_build"
            echo "Reusing surviving vLLM build dir: $CMAKE_BUILD_DIR"
        else
            export CMAKE_BUILD_DIR="$VLLM_BUILD_ROOT/cmake"
            echo "No prior build dir found; using persistent $CMAKE_BUILD_DIR"
        fi
    fi
    mkdir -p "$CMAKE_BUILD_DIR"
    # ccache is content-addressed, so it survives even if the build dir name
    # changes between runs (pip picks a fresh temp each time). vLLM's CMake wires
    # these launchers automatically when ccache is present; point its store at
    # the shared filesystem so the cache is not lost with the job's TMPDIR.
    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_DIR="$VLLM_BUILD_ROOT/ccache"
        mkdir -p "$CCACHE_DIR"
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
        echo "ccache active: $(ccache --version | head -n1) (dir=$CCACHE_DIR)"
    else
        echo "WARNING: ccache not found on PATH. A from-scratch rebuild will"
        echo "         recompile every CUDA object. Install/module-load ccache"
        echo "         to make subsequent builds fast, or reuse CMAKE_BUILD_DIR."
    fi

    # Build vLLM from source. uv hides the build backend's output, so the long
    # CUDA compile looks like a hang at "Preparing packages (0/1)". Use pip -v
    # (the venv is --seeded so pip exists) to stream the ninja "[N/M]" progress,
    # and tee it to the shared-filesystem log so it can be tailed remotely.
    # pipefail (set -o) keeps a build failure fatal despite the tee.
    echo "Building vLLM from source with streaming progress (MAX_JOBS=$MAX_JOBS)."
    echo "Follow along: tail -f $VLLM_BUILD_LOG"
    # PIP_CONSTRAINT holds torch at the verified cu126/sm70 build even though
    # vLLM lists torch as a runtime dependency. If vLLM hard-pins a different
    # torch this fails fast at resolution (before the compile), surfacing a real
    # version mismatch rather than silently clobbering torch.
    PIP_CONSTRAINT="$TORCH_CONSTRAINTS" \
        "$VLLM_PYTHON" -m pip install --no-build-isolation -v -e . 2>&1 \
        | tee "$VLLM_BUILD_LOG"
    # The extension that was just compiled links against this exact torch build.
    verify_torch_stack "vLLM source build"
    # Cache stats help confirm the next build will reuse compiled objects.
    if command -v ccache >/dev/null 2>&1; then
        ccache -s 2>/dev/null | sed 's/^/ccache: /' || true
    fi
)

# more-itertools is a runtime dependency of openai-whisper (pulled in via
# vllm-omni) that can be left out of the resolved set; import then fails with
# "openai-whisper requires more-itertools". Install it explicitly to be safe.
#
# This is the step that used to break the environment: vllm-omni pulls torch in
# transitively (torchsde, x-transformers, openai-whisper), and with a bare
# torch==<version> constraint against PyPI it reinstalled the CUDA 13 wheel over
# the verified cu126 one -- leaving torchaudio and the source-built vLLM on
# cu126. The +$CUDA_WHEEL_TAG constraint plus the exported PyTorch index keep it
# pinned, and the check below fails loudly if anything still moves it.
uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" \
    "vllm-omni==$VLLM_OMNI_VERSION" \
    numpy soundfile sphn uroman scipy PyYAML more-itertools

verify_torch_stack "vllm-omni install"

# Force transformers to the vllm-omni-compatible version last, overriding the
# newer one vllm pulls in. Done as a separate step (rather than a constraint on
# the installs above) so it cannot conflict with vllm's declared lower bound;
# vllm and vllm-omni were verified to import at this version.
uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" \
    "transformers==$TRANSFORMERS_VERSION"

verify_torch_stack "transformers pin"

# Final end-to-end check: importing vllm loads the C extension that was compiled
# against the torch above, so this is where a surviving torch/CUDA mismatch
# would show up as an undefined-symbol ImportError.
EXPECT_TRANSFORMERS="$TRANSFORMERS_VERSION" "$VLLM_PYTHON" - <<'PY'
import os
import torch
import torchaudio
import transformers
import vllm
import vllm_omni

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("vllm-omni:", getattr(vllm_omni, "__version__", "unknown"))
expected_tf = os.environ["EXPECT_TRANSFORMERS"]
if transformers.__version__ != expected_tf:
    raise SystemExit(
        f"transformers {transformers.__version__} != pinned {expected_tf}; "
        "vllm-omni Stage1 needs the input_embeds create_causal_mask signature"
    )
print("CUDA smoke:", (torch.ones(1, device="cuda") + 1).item())
PY

echo "vLLM-Omni environment ready: $VLLM_ENV_DIR"
