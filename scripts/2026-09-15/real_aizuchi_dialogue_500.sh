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
# Position: probability per clause boundary, like the original rule-based
# route, with one change -- the end of the utterance is now guaranteed
# (previously it was drawn from the same rates as mid-utterance positions, so
# a listener could still go quiet after the speaker finished). Word: chosen
# by the same LLM call the rule-based route used, not at random -- deciding
# among a handful of short acknowledgement words still benefits from seeing
# what was just said (e.g. preferring an acknowledgement word right after a
# fact/feeling lands), which a uniform random pick can't do.
export AIZUCHI_ONLY_PLACEMENT="density"
# 0 = no backchannels at all, 0.25/0.5/0.75 = reserved/normal/eager, 1 = flood
# (the "too much" upper bound; not the inference/production default, but fine
# to use when generating training data). Values between anchors interpolate
# continuously, replacing the discrete preset choice.
export AIZUCHI_DENSITY="${AIZUCHI_DENSITY:-0.5}"
# Vocabulary is the written default (AIZUCHI_ONLY_VOCAB: hai/ee/sou-nan-desu-ne/
# aa... etc) minus one entry. The real-recording-derived listening_vocab.tsv
# put "sokka" in reach for almost every position and it stood out; reverting
# to the plain written list traded that for a smaller version of the same
# problem, bare "un." alone standing out the same way, while "un, un." and
# "hai, hai." (also in the written list) did not.
# aizuchi_vocab_no_bare_un.tsv is the written list with only that one entry
# removed. scripts/2026-09-15/listening_vocab.tsv and real_aizuchi_examples.md
# are kept on disk if the real-vocabulary direction is worth revisiting.
export AIZUCHI_VOCAB_FILE="$PWD/scripts/2026-09-15/aizuchi_vocab_no_bare_un.tsv"
# Examples apply here too -- density mode calls the same word-choice prompt
# builder (build_aizuchi_only_agent_prompt) as rule mode, only the position
# logic differs. Thinking (below) is still llm-mode-only: density's LLM call
# is a short word pick from a handful of options, not open-ended placement.
export AIZUCHI_ONLY_EXAMPLE="${AIZUCHI_ONLY_EXAMPLE:-0}"

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
