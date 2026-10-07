#!/usr/bin/env bash
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"
source scripts/2026-10-07/common.sh
stage="${1:?stage required}"
case "$stage" in
source)
    export CUDA_VISIBLE_DEVICES=0
    candidate="${COMPARE_SOURCE_INPUT:-$PWD/data/runs/real_aizuchi_${COMPARE_N}_v2/dialogue/llm_dialogues/dialogues.jsonl}"
    if [[ ! -s "$candidate" ]]; then
        if [[ -n "${COMPARE_SOURCE_INPUT:-}" ]]; then
            echo "Requested source does not exist: $candidate" >&2; exit 1
        fi
        export BATCH_ID="${COMPARE_ID}_shared"
        export OUT_ROOT="$COMPARE_ROOT/shared/generated"
        export DIALOGUE_GENERATION_MODE=aizuchi-only AIZUCHI_ONLY_PLACEMENT=density
        export AIZUCHI_DENSITY_MIXED=1 AIZUCHI_DENSITY=0.5
        export AIZUCHI_ONLY_EXAMPLE=0 AIZUCHI_ENABLE_THINKING=0
        export AIZUCHI_ONLY_MIN_BLOCKS=4 AIZUCHI_ONLY_MAX_BLOCKS=6
        export AIZUCHI_ONLY_MAX_SILENCES=2 AIZUCHI_ONLY_SILENCE_RATE=0.25
        export AIZUCHI_ONLY_SILENCE_MIN_SEC=2.0 AIZUCHI_ONLY_SILENCE_MAX_SEC=5.0
        export AIZUCHI_ONLY_PROBE_RATE=0.5 DIALOGUE_RESUME=1 OVERWRITE=1
        unset ALLOW_TEMPLATE_FALLBACK
        bash scripts/run_dialogues_qwen_3000.pbs
        candidate="$OUT_ROOT/llm_dialogues/dialogues.jsonl"
    fi
    python3 scripts/aizuchi_comparison.py freeze --source "$candidate" --out "$COMPARE_SOURCE" --count "$COMPARE_N"
    ;;
bank)
    if [[ -s "$COMPARE_BANK/samples.jsonl" ]]; then
        echo "Reusing shared v2 bank: $COMPARE_BANK"
    else
        export BATCH_ID="${COMPARE_ID}_bank" OUT_ROOT="$COMPARE_BANK"
        export TEXTS_FILE="$COMPARE_VOCAB" REPEATS="${COMPARE_BANK_REPEATS:-160}"
        bash scripts/run_qwen3_bank_4gpu.sh
    fi
    python3 scripts/2026-10-07/check_bank.py "$COMPARE_BANK/samples.jsonl" "$COMPARE_VOCAB"
    ;;
ai)
    export CUDA_VISIBLE_DEVICES=0
    test -s "$COMPARE_SOURCE"
    export BATCH_ID="${COMPARE_ID}_ai_placement"
    export OUT_ROOT="$COMPARE_ROOT/ai_placement/dialogue"
    export AIZUCHI_REPOSITION_SOURCE="$COMPARE_SOURCE" OVERWRITE=1
    export MULTI_AGENT_CONCURRENCY="${COMPARE_AI_CONCURRENCY:-16}"
    bash scripts/run_dialogues_qwen_3000.pbs
    ;;
render)
    test -s "$COMPARE_SOURCE"
    export DIALOGUES_JSONL="$COMPARE_SOURCE"
    [[ "$ARM" != ai_placement ]] || export DIALOGUES_JSONL="$ARM_ROOT/dialogue/llm_dialogues/dialogues.jsonl"
    test -s "$DIALOGUES_JSONL"
    export SOURCE_BATCH_ID="${RUN_ID}_dialogue" BATCH_ID="${RUN_ID}_tts"
    export OUT_ROOT="$ARM_ROOT/tts"
    if [[ "$ARM" == traditional_overlap ]]; then
        # Whole-utterance Qwen TTS with the existing near-clause-end overlap.
        # Synthesizes both voices. No bank rewrite, KABURI or bank splice.
        bash scripts/run_qwen_tts_vllm_3000_4gpu.pbs
    else
        python3 scripts/2026-10-07/check_bank.py "$COMPARE_BANK/samples.jsonl" "$COMPARE_VOCAB"
        export BANK_DIR="$COMPARE_BANK"
        bash scripts/run_bank_stereo_test.sh "$COMPARE_N"
    fi
    ;;
condition)
    if [[ "$ARM" == traditional_overlap ]]; then
        # The production merge holds absolute manifest paths, not data_stereo.
        # Condition each real shard, then merge their conditioned manifests.
        for ((i=0; i<NUM_SHARDS; i++)); do
            printf -v shard 'shard_%03d' "$i"
            export SRC_SHARD="$ARM_ROOT/tts/$shard"
            export DST_SHARD="$ARM_ROOT/tts/conditioned/$shard"
            bash scripts/2026-09-15/real_aizuchi_condition_10000_v2.pbs
        done
        uv run python scripts/merge_training_shards.py --batch-dir "$ARM_ROOT/tts/conditioned" \
            --out-dir "$ARM_ROOT/tts/merged_conditioned" --expected-shards "$NUM_SHARDS"
    else
        export SRC_SHARD="$ARM_ROOT/tts/placement_bank/shard_000"
        export DST_SHARD="${SRC_SHARD}_conditioned"
        bash scripts/2026-09-15/real_aizuchi_condition_10000_v2.pbs
    fi
    ;;
pair)
    python3 scripts/aizuchi_comparison.py assemble --source "$COMPARE_SOURCE" --root "$COMPARE_ROOT"
    ;;
train) exec bash scripts/2026-10-07/train_chain.sh ;;
eval) exec bash scripts/2026-10-07/evaluate.sh ;;
report) python3 scripts/2026-10-07/report_comparison.py --root "$COMPARE_ROOT" --id "$COMPARE_ID" ;;
*) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac
