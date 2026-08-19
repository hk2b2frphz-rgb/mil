#!/usr/bin/env bash
# Submit the screw PoC pipeline: TTS now, LoRA and Full FT by self-chaining.
#
# Only the TTS job is submitted from here.  It qsubs the two training jobs
# itself, from its own tail, once the merged manifest exists -- see the
# "chain: submit training" step at the end of run_tts_1000_4gpu.pbs.
#
# This replaces the earlier `qsub -W depend=afterok` form, which queued both
# training jobs up front: when TTS failed they stayed in the queue as held
# jobs that had to be qdel'd by hand.  Nothing is queued speculatively now.
#
#   bash screw_poc/pbs/submit_pipeline.sh
#
# Comparison subsets are submitted through the same path -- the variables below
# are forwarded to the TTS job with lowercase -v, and the chain carries the
# matching TTS_RUN_DIR into training automatically:
#
#   DIALOGUES_JSONL=screw_poc/artifacts/subsets/train_0300.jsonl \
#   NUM_DIALOGUES=300 \
#   OUT_ROOT=screw_poc/artifacts/tts_0300 \
#       bash screw_poc/pbs/submit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

command -v qsub >/dev/null 2>&1 || {
    echo "ERROR: qsub is not available on PATH." >&2
    exit 1
}

# Forward only what the caller actually set, so the job keeps its own defaults
# for everything else.  CHAIN_TRAIN=1 is explicit: this entry point exists to
# run the whole pipeline.
chain_env="CHAIN_TRAIN=1"
for var in DIALOGUES_JSONL NUM_DIALOGUES OUT_ROOT NUM_SHARDS; do
    if [[ -n "${!var:-}" ]]; then
        chain_env+=",${var}=${!var}"
    fi
done

tts_job="$(qsub -V -v "$chain_env" screw_poc/pbs/run_tts_1000_4gpu.pbs)"

echo "TTS job:  $tts_job"
echo "env:      $chain_env"
echo
echo "LoRA and Full-FT are submitted by the TTS job itself once its merged"
echo "manifest is verified.  Their job ids appear under '[chain]' at the end of"
echo "screw_poc/artifacts/pbs_logs/tts_1000_<jobid>.log."
