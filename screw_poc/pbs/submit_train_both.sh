#!/usr/bin/env bash
# Submit the two training jobs against TTS output that already exists.
#
# The normal path is submit_pipeline.sh, where the TTS job chains these two
# itself.  Use this script to re-run training on a finished TTS corpus, or to
# hang training off a TTS job that is still queued by passing its job id:
#
#   bash screw_poc/pbs/submit_train_both.sh
#   TTS_RUN_DIR=screw_poc/artifacts/tts_0300/merged bash screw_poc/pbs/submit_train_both.sh
#   bash screw_poc/pbs/submit_train_both.sh 12345.pbsserver
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

command -v qsub >/dev/null 2>&1 || {
    echo "ERROR: qsub is not available on PATH." >&2
    exit 1
}

TTS_RUN_DIR="${TTS_RUN_DIR:-$REPO_ROOT/screw_poc/artifacts/tts_1000/merged}"
DEPENDENCY_JOB_ID="${1:-}"

MANIFEST="$TTS_RUN_DIR/training_set/synthetic_moshi_train.jsonl"
if [[ -z "$DEPENDENCY_JOB_ID" && ! -s "$MANIFEST" ]]; then
    echo "ERROR: training manifest not found: $MANIFEST" >&2
    echo "Run the TTS job first, or pass its job id to chain on afterok." >&2
    exit 1
fi

# NPROC is set per job so neither run inherits a wider CUDA_VISIBLE_DEVICES
# through -V.  Lowercase -v wins over the -V environment.
qsub_args=(-V)
if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    qsub_args+=(-W "depend=afterok:${DEPENDENCY_JOB_ID}")
fi

lora_job="$(qsub "${qsub_args[@]}" -v "TTS_RUN_DIR=$TTS_RUN_DIR,NPROC=1" \
    screw_poc/pbs/run_train.pbs)"
full_job="$(qsub "${qsub_args[@]}" -v "TTS_RUN_DIR=$TTS_RUN_DIR,NPROC=2" \
    screw_poc/pbs/run_train_full.pbs)"

echo "dataset:      $TTS_RUN_DIR"
echo "LoRA job:     $lora_job"
echo "Full-FT job:  $full_job"
if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    echo "dependency:   afterok:$DEPENDENCY_JOB_ID"
fi
