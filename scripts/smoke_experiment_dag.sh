#!/usr/bin/env bash
# Smoke test for the experiment DAG. Two modes:
#
#   bash scripts/smoke_experiment_dag.sh          # local: inline, no qsub
#   bash scripts/smoke_experiment_dag.sh pbs      # real qsub self-chain (tiny jobs)
#
# local: runs the whole chain in one process (--local). The final `verify`
#        stage checks outputs and prints [SMOKE OK]. Fast, no scheduler.
# pbs:   kicks off --start (real qsub), so each tiny stage job submits the next
#        from the compute node -- this is what verifies self-chaining works on
#        this cluster. There is NO login-node polling: the final `verify` stage
#        (dag_verify job) does the checking and its log ends with [SMOKE OK].
#
# PYTHON overrides how the driver is invoked (default: uv run python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-local}"
CFG="configs/experiment.smoke.yaml"
OUT="data/runs/_smoke"
PYTHON="${PYTHON:-uv run python}"
DRIVER="scripts/run_experiment_dag.py"

if [[ "$MODE" == "local" ]]; then
    echo "== SMOKE (local, --local): full chain =="
    rm -rf "$OUT"
    $PYTHON "$DRIVER" "$CFG" --start --local
    echo
    echo "== SMOKE (local): start mid-chain from tts =="
    rm -rf "$OUT/tts" "$OUT/checkpoints" "$OUT/eval"
    $PYTHON "$DRIVER" "$CFG" --start-at tts --local
    rc=$?
    if [[ "$rc" -eq 0 ]]; then rm -rf "$OUT"; echo "[SMOKE OK] local"; fi
    exit "$rc"

elif [[ "$MODE" == "pbs" ]]; then
    command -v qsub >/dev/null 2>&1 || { echo "ERROR: qsub not on PATH" >&2; exit 1; }
    echo "== SMOKE (pbs): real qsub self-chain (no login-node polling) =="
    rm -rf "$OUT"
    # One instant submit; the chain (dialogue->...->eval->verify) self-runs as
    # tiny qsub jobs, each submitting the next from the compute node.
    $PYTHON "$DRIVER" "$CFG" --start
    echo
    echo "Submitted. The chain now runs as qsub jobs and self-verifies."
    echo "  watch:   qstat -u \$USER"
    echo "  result:  the final 'dag_verify' job log ends with [SMOKE OK]"
    echo "  or once qstat is empty:  $PYTHON $DRIVER $CFG --check"
    exit 0

else
    echo "usage: $0 [local|pbs]" >&2
    exit 2
fi
