#!/usr/bin/env bash
# Smoke test for the experiment DAG: run the whole pipeline locally (no GPU,
# no qsub) with fake instant stages, and confirm each stage clears with its
# output wired to the next. Also checks starting mid-chain (from tts, reusing
# the already-created dialogue folder).
#
#   bash scripts/smoke_experiment_dag.sh
#
# PYTHON can override how the driver is invoked (default: uv run python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CFG="configs/experiment.smoke.yaml"
OUT="data/runs/_smoke"
PYTHON="${PYTHON:-uv run python}"
DRIVER="scripts/run_experiment_dag.py"

pass=0
fail=0
check() {  # check <label> <path-that-must-exist>
    if [[ -e "$2" ]]; then
        echo "  [ok]   $1: $2"
        pass=$((pass + 1))
    else
        echo "  [MISS] $1: MISSING $2"
        fail=$((fail + 1))
    fi
}

echo "========================================================"
echo "SMOKE 1/2: full chain (dialogue -> tts -> train -> eval), --local"
echo "========================================================"
rm -rf "$OUT"
$PYTHON "$DRIVER" "$CFG" --start --local

echo
echo "-- verify stage outputs --"
check dialogue "$OUT/dialogue/dialogues.jsonl"
check tts      "$OUT/tts/synthetic_moshi_train.jsonl"
check train    "$OUT/checkpoints/consolidated.safetensors"
check eval     "$OUT/eval/summary.txt"

echo
echo "========================================================"
echo "SMOKE 2/2: start mid-chain from tts (reuse existing dialogue folder)"
echo "========================================================"
# Keep dialogue output, wipe downstream, then start at tts.
rm -rf "$OUT/tts" "$OUT/checkpoints" "$OUT/eval"
$PYTHON "$DRIVER" "$CFG" --start-at tts --local
check tts-from-mid "$OUT/tts/synthetic_moshi_train.jsonl"
check eval-from-mid "$OUT/eval/summary.txt"

echo
echo "========================================================"
if [[ "$fail" -eq 0 ]]; then
    echo "[SMOKE OK] 全ステージ通過 ($pass checks)"
    rm -rf "$OUT"
    exit 0
else
    echo "[SMOKE FAILED] $fail 件の欠落 ($pass ok)"
    echo "   出力は $OUT に残しています(調査用)"
    exit 1
fi
