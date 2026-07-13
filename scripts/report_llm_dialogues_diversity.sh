#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/report_llm_dialogues_diversity.sh <folder> [output-file]

<folder> may be either:
  - a run folder containing llm_dialogues/dialogues.jsonl
  - the llm_dialogues folder itself

By default, the report is written beside llm_dialogues as
<run-folder>/diversity_report.txt.

Environment variables:
  SAMPLES=8          Number of full dialogue samples included in the report.
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

TARGET_DIR="${1%/}"
SAMPLES="${SAMPLES:-8}"
if ! [[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SAMPLES must be a positive integer: $SAMPLES" >&2
    exit 2
fi
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "ERROR: folder does not exist: $TARGET_DIR" >&2
    exit 1
fi

if [[ -f "$TARGET_DIR/llm_dialogues/dialogues.jsonl" ]]; then
    RUN_DIR="$TARGET_DIR"
    INPUT_JSONL="$TARGET_DIR/llm_dialogues/dialogues.jsonl"
elif [[ -f "$TARGET_DIR/dialogues.jsonl" && "$(basename "$TARGET_DIR")" == "llm_dialogues" ]]; then
    RUN_DIR="$(dirname "$TARGET_DIR")"
    INPUT_JSONL="$TARGET_DIR/dialogues.jsonl"
else
    echo "ERROR: dialogues.jsonl was not found under: $TARGET_DIR/llm_dialogues" >&2
    echo "       You may also pass the llm_dialogues folder directly." >&2
    exit 1
fi

OUTPUT_REPORT="${2:-$RUN_DIR/diversity_report.txt}"

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
    --samples "$SAMPLES"

echo "[diversity] input:  $INPUT_JSONL"
echo "[diversity] report: $OUTPUT_REPORT"
