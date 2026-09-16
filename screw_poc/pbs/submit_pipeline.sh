#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

command -v qsub >/dev/null 2>&1 || {
    echo "ERROR: qsub is not available on PATH." >&2
    exit 1
}

tts_job="$(qsub -V screw_poc/pbs/run_tts_1000_4gpu.pbs)"
lora_job="$(qsub -V -W "depend=afterok:${tts_job}" screw_poc/pbs/run_train.pbs)"
full_job="$(qsub -V -W "depend=afterok:${tts_job}" screw_poc/pbs/run_train_full.pbs)"

echo "TTS job:       $tts_job"
echo "LoRA job:      $lora_job (starts after successful TTS completion)"
echo "Full-FT job:   $full_job (starts after successful TTS completion)"
