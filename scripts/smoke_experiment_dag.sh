#!/usr/bin/env bash
# Smoke test for the experiment DAG. Two modes:
#
#   bash scripts/smoke_experiment_dag.sh          # local: inline, no qsub
#   bash scripts/smoke_experiment_dag.sh pbs      # real qsub self-chain (tiny jobs)
#
# local: runs the whole chain in one process (--local) and verifies outputs.
#        Fast, no scheduler; checks orchestration + output wiring.
# pbs:   kicks off --start (real qsub), so each tiny stage job submits the next
#        from the compute node. This is what verifies the self-chaining actually
#        works on this cluster. Waits for the final stage's output, then --check.
#
# PYTHON overrides how the driver is invoked (default: uv run python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-local}"
CFG="configs/experiment.smoke.yaml"
OUT="data/runs/_smoke"
FINAL_MARKER="$OUT/eval/summary.txt"
PYTHON="${PYTHON:-uv run python}"
DRIVER="scripts/run_experiment_dag.py"
WAIT_SECS="${WAIT_SECS:-1800}"   # pbs mode: max seconds to wait for the chain

if [[ "$MODE" == "local" ]]; then
    echo "== SMOKE (local, --local): full chain =="
    rm -rf "$OUT"
    $PYTHON "$DRIVER" "$CFG" --start --local
    echo
    echo "== SMOKE (local): start mid-chain from tts =="
    rm -rf "$OUT/tts" "$OUT/checkpoints" "$OUT/eval"
    $PYTHON "$DRIVER" "$CFG" --start-at tts --local
    echo
    echo "== verify =="
    $PYTHON "$DRIVER" "$CFG" --check
    rc=$?
    if [[ "$rc" -eq 0 ]]; then rm -rf "$OUT"; echo "[SMOKE OK] local"; fi
    exit "$rc"

elif [[ "$MODE" == "pbs" ]]; then
    command -v qsub >/dev/null 2>&1 || { echo "ERROR: qsub not on PATH" >&2; exit 1; }
    echo "== SMOKE (pbs): real qsub self-chain =="
    rm -rf "$OUT"
    $PYTHON "$DRIVER" "$CFG" --start
    echo
    echo "waiting up to ${WAIT_SECS}s for the chain to reach eval ($FINAL_MARKER)..."
    waited=0
    while [[ ! -f "$FINAL_MARKER" ]]; do
        sleep 15
        waited=$((waited + 15))
        if [[ "$waited" -ge "$WAIT_SECS" ]]; then
            echo "[SMOKE TIMEOUT] chain did not finish in ${WAIT_SECS}s." >&2
            echo "  Check: qstat -u \$USER  and the dag_* job logs." >&2
            $PYTHON "$DRIVER" "$CFG" --check || true
            exit 1
        fi
        echo "  ... ${waited}s (qstat -u \$USER to watch)"
    done
    echo "chain reached eval. verifying all stages:"
    $PYTHON "$DRIVER" "$CFG" --check
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
        rm -rf "$OUT"
        echo "[SMOKE OK] pbs self-chaining works (compute-node qsub confirmed)"
    fi
    exit "$rc"

else
    echo "usage: $0 [local|pbs]" >&2
    exit 2
fi
