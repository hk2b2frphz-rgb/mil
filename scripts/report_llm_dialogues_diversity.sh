#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/report_llm_dialogues_diversity.sh <jsonl-or-folder> [output-file]

The first argument may be any of:
  - a dialogue JSONL file, such as llm_dialogues/dialogues.jsonl
  - a run folder containing llm_dialogues/dialogues.jsonl
  - the llm_dialogues folder itself

By default, the report is written beside llm_dialogues as
<run-folder>/diversity_report.txt. Per-dialogue nearest neighbors are written
to <run-folder>/diversity_neighbors.jsonl.

Environment variables:
  SAMPLES=8          Number of full dialogue samples included in the report.
  SIMILARITY_BLOCK_SIZE=128
                     Rows per exact Jaccard work block. Lower this to reduce
                     temporary memory; it does not reduce comparison coverage.
  SCALE_POINTS=100,300,1000,3000,10000
                     Nested sample sizes shown in the scaling table.
  SIMILARITY_SEED=0  Seed used to build reproducible nested scaling samples.
  DIVERSITY_NEIGHBORS_JSONL=<run-folder>/diversity_neighbors.jsonl
                     Per-dialogue nearest-neighbor output path.
  PYTHON_BIN=python  Python executable used when uv is unavailable.
  USE_UV=0           Disable uv even when it is available.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

TARGET="${1%/}"
SAMPLES="${SAMPLES:-8}"
SIMILARITY_BLOCK_SIZE="${SIMILARITY_BLOCK_SIZE:-128}"
SCALE_POINTS="${SCALE_POINTS:-100,300,1000,3000,10000}"
SIMILARITY_SEED="${SIMILARITY_SEED:-0}"
if ! [[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SAMPLES must be a positive integer: $SAMPLES" >&2
    exit 2
fi
if ! [[ "$SIMILARITY_BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SIMILARITY_BLOCK_SIZE must be a positive integer: $SIMILARITY_BLOCK_SIZE" >&2
    exit 2
fi
if ! [[ "$SIMILARITY_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: SIMILARITY_SEED must be an integer: $SIMILARITY_SEED" >&2
    exit 2
fi
if [[ -f "$TARGET" ]]; then
    if [[ "$TARGET" != *.jsonl ]]; then
        echo "ERROR: input file must have a .jsonl extension: $TARGET" >&2
        exit 1
    fi
    INPUT_JSONL="$TARGET"
    INPUT_DIR="$(dirname "$TARGET")"
    if [[ "$(basename "$INPUT_DIR")" == "llm_dialogues" ]]; then
        RUN_DIR="$(dirname "$INPUT_DIR")"
    else
        RUN_DIR="$INPUT_DIR"
    fi
elif [[ -f "$TARGET/llm_dialogues/dialogues.jsonl" ]]; then
    RUN_DIR="$TARGET"
    INPUT_JSONL="$TARGET/llm_dialogues/dialogues.jsonl"
elif [[ -f "$TARGET/dialogues.jsonl" && "$(basename "$TARGET")" == "llm_dialogues" ]]; then
    RUN_DIR="$(dirname "$TARGET")"
    INPUT_JSONL="$TARGET/dialogues.jsonl"
else
    echo "ERROR: no dialogue JSONL input was found for: $TARGET" >&2
    echo "       Pass a .jsonl file, a run folder, or an llm_dialogues folder." >&2
    exit 1
fi

OUTPUT_REPORT="${2:-$RUN_DIR/diversity_report.txt}"
OUTPUT_NEIGHBORS="${DIVERSITY_NEIGHBORS_JSONL:-$RUN_DIR/diversity_neighbors.jsonl}"

if [[ "${USE_UV:-1}" != "0" ]] && command -v uv >/dev/null 2>&1; then
    python_cmd=(uv run python)
elif [[ -n "${PYTHON_BIN:-}" ]]; then
    python_cmd=("$PYTHON_BIN")
elif command -v python3 >/dev/null 2>&1; then
    python_cmd=(python3)
elif command -v python >/dev/null 2>&1; then
    python_cmd=(python)
else
    echo "ERROR: uv or Python is required." >&2
    exit 1
fi

"${python_cmd[@]}" "$SCRIPT_DIR/report_dialogue_diversity.py" \
    --in "$INPUT_JSONL" \
    --out "$OUTPUT_REPORT" \
    --nearest-jsonl "$OUTPUT_NEIGHBORS" \
    --samples "$SAMPLES" \
    --similarity-block-size "$SIMILARITY_BLOCK_SIZE" \
    --scale-points "$SCALE_POINTS" \
    --similarity-seed "$SIMILARITY_SEED"

echo "[diversity] input:  $INPUT_JSONL"
echo "[diversity] report: $OUTPUT_REPORT"
echo "[diversity] nearest neighbors: $OUTPUT_NEIGHBORS"
