#!/usr/bin/env bash
set -euo pipefail

# Interactive-node equivalent of scripts/2026-09-15/real_aizuchi_dialogue_500.pbs.
# Same environment variables, exec'd directly with bash instead of through
# qsub. Needs one GPU (vLLM serves Qwen3.6-27B in bf16, A100-class).
#
#   bash scripts/2026-09-15/real_aizuchi_dialogue_500.sh
#   AIZUCHI_ENABLE_THINKING=1 bash scripts/2026-09-15/real_aizuchi_dialogue_500.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/2026-09-15/real_aizuchi_dialogue_500.sh
#
# See real_aizuchi_dialogue_500.pbs for why each setting is what it is; this
# file only carries the same values so there is nothing to keep in sync by
# hand. Progress is tee'd by the underlying script to
# experiments/pbs_logs/real_aizuchi_500_<version>_local.log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CORPUS_VERSION="${CORPUS_VERSION:-v1}"
export BATCH_ID="real_aizuchi_500_${CORPUS_VERSION}"
export OUT_ROOT="$PWD/data/runs/real_aizuchi_500_${CORPUS_VERSION}/dialogue"
export NUM_CASES="${NUM_CASES:-500}"

export DIALOGUE_GENERATION_MODE="aizuchi-only"
export AIZUCHI_ONLY_PLACEMENT="llm"
export AIZUCHI_VOCAB_FILE="$PWD/scripts/2026-09-15/listening_vocab.tsv"
export AIZUCHI_PROBE_REPLIES_FILE="$PWD/scripts/2026-09-15/listening_probe_replies.txt"
export AIZUCHI_NO_REPEAT_WINDOW="0"
# Shows aizuchiAI one real example (non-backchannel content stripped from an
# actual transcript excerpt) instead of zero-shot. On by default here: the
# real recording showed the model's job is closer to "match this rhythm" than
# "follow this rule", which a worked example teaches better than more prose.
export AIZUCHI_ONLY_EXAMPLE="${AIZUCHI_ONLY_EXAMPLE:-1}"

export AIZUCHI_ENABLE_THINKING="${AIZUCHI_ENABLE_THINKING:-0}"
export AIZUCHI_THINKING_MAX_TOKENS="${AIZUCHI_THINKING_MAX_TOKENS:-1600}"

export AIZUCHI_ONLY_FREQUENCY="normal"
export AIZUCHI_ONLY_MIN_BLOCKS="${AIZUCHI_ONLY_MIN_BLOCKS:-4}"
export AIZUCHI_ONLY_MAX_BLOCKS="${AIZUCHI_ONLY_MAX_BLOCKS:-6}"

export AIZUCHI_ONLY_MAX_SILENCES="${AIZUCHI_ONLY_MAX_SILENCES:-2}"
export AIZUCHI_ONLY_SILENCE_RATE="${AIZUCHI_ONLY_SILENCE_RATE:-0.25}"
export AIZUCHI_ONLY_SILENCE_MIN_SEC="${AIZUCHI_ONLY_SILENCE_MIN_SEC:-2.0}"
export AIZUCHI_ONLY_SILENCE_MAX_SEC="${AIZUCHI_ONLY_SILENCE_MAX_SEC:-5.0}"
export AIZUCHI_ONLY_PROBE_RATE="${AIZUCHI_ONLY_PROBE_RATE:-0.5}"

echo "===== real_aizuchi_dialogue_500 (interactive) ====="
echo "repo:       $REPO_ROOT"
echo "batch_id:   $BATCH_ID"
echo "out_root:   $OUT_ROOT"
echo "num_cases:  $NUM_CASES"
echo "thinking:   $AIZUCHI_ENABLE_THINKING (max_tokens=$AIZUCHI_THINKING_MAX_TOKENS)"
echo "gpu:        ${CUDA_VISIBLE_DEVICES:-0 (default)}"
echo "===================================================="

exec bash scripts/run_dialogues_qwen_3000.pbs
