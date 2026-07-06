#!/usr/bin/env bash
# Runs scripts/run_full_duplex_eval.sh once per model listed in MODELS_FILE,
# writing each model's results under BATCH_OUT_DIR/<model_id>/ and a final
# combined_summary.json comparing every model side by side. See
# scripts/models_manifest.example.txt for the manifest format.
#
# Deliberately not `set -e`: one model's failure must not abort the rest of
# the batch. Per-model failures are recorded in the summary table printed at
# the end, and the script's own exit code is non-zero iff any model failed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODELS_FILE="${MODELS_FILE:?Set MODELS_FILE to a manifest listing models to evaluate (see scripts/models_manifest.example.txt)}"
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "ERROR: MODELS_FILE not found: $MODELS_FILE" >&2
    exit 1
fi

BATCH_ID="${BATCH_ID:-batch_$(date +%Y%m%d_%H%M%S)}"
BATCH_OUT_DIR="${BATCH_OUT_DIR:-$REPO_ROOT/eval_runs/full_duplex_batches/$BATCH_ID}"
mkdir -p "$BATCH_OUT_DIR"
exec > >(tee -a "$BATCH_OUT_DIR/batch.log") 2>&1

echo "===== Full-Duplex-Bench-JA batch eval ====="
echo "batch_id:    $BATCH_ID"
echo "models_file: $MODELS_FILE"
echo "out_dir:     $BATCH_OUT_DIR"
echo "started_at:  $(date -Iseconds)"
echo "============================================"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

RESULT_LINES=()
TOTAL=0
FAILED=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue

    IFS='|' read -r model_id model_weight model_config hf_repo extra_env opening_greeting <<< "$line"
    model_id="$(echo "$model_id" | xargs)"
    model_weight="$(echo "$model_weight" | xargs)"
    model_config="$(echo "$model_config" | xargs)"
    hf_repo="$(echo "$hf_repo" | xargs)"
    extra_env="$(echo "$extra_env" | xargs)"
    opening_greeting="$(echo "$opening_greeting" | xargs)"
    if [[ -n "$opening_greeting" && "$opening_greeting" != "0" && "$opening_greeting" != "1" ]]; then
        echo "[batch] WARN: opening_greeting must be 0 or 1 for $model_id, ignoring invalid value: $opening_greeting" >&2
        opening_greeting=""
    fi

    if [[ -z "$model_id" ]]; then
        echo "[batch] WARN: skipping manifest line with empty model_id: $raw_line" >&2
        continue
    fi
    if ! [[ "$model_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[batch] WARN: skipping invalid model_id (letters/digits/dot/underscore/hyphen only): $model_id" >&2
        continue
    fi

    TOTAL=$((TOTAL + 1))

    run_env=(
        "MODEL_ID=$model_id"
        "RUN_ID=$model_id"
        "FDB_OUT_DIR=$BATCH_OUT_DIR/$model_id"
    )
    [[ -n "$model_weight" ]] && run_env+=("MODEL_WEIGHT=$model_weight")
    [[ -n "$model_config" ]] && run_env+=("MODEL_CONFIG=$model_config")
    [[ -n "$hf_repo" ]] && run_env+=("HF_REPO=$hf_repo")
    if [[ -n "$extra_env" ]]; then
        IFS=';' read -ra pairs <<< "$extra_env"
        for pair in "${pairs[@]}"; do
            pair="$(echo "$pair" | xargs)"
            [[ -n "$pair" ]] && run_env+=("$pair")
        done
    fi

    # Precedence for FDB_OPENING_GREETING, highest first:
    #   1. this row's dedicated opening_greeting manifest column
    #   2. FDB_OPENING_GREETING=... inside this row's extra_env
    #   3. auto-default: a row with no model_weight has no local checkpoint
    #      override, i.e. it pulls the unmodified HF checkpoint (e.g.
    #      model_id=base) -- which was never trained to say the fixed
    #      opening greeting (see run_full_duplex_eval.sh /
    #      docs/full_duplex_evaluation.md). Default that row to
    #      FDB_OPENING_GREETING=0 so base/llm-jp rows don't sit through a
    #      dead-air lead-in reserved for a greeting they'll never say.
    greeting_source=""
    if [[ -n "$opening_greeting" ]]; then
        for i in "${!run_env[@]}"; do
            [[ "${run_env[$i]}" == FDB_OPENING_GREETING=* ]] && unset 'run_env[i]'
        done
        run_env+=("FDB_OPENING_GREETING=$opening_greeting")
        greeting_source="manifest column ($opening_greeting)"
    else
        for kv in "${run_env[@]}"; do
            if [[ "$kv" == FDB_OPENING_GREETING=* ]]; then
                greeting_source="extra_env (${kv#FDB_OPENING_GREETING=})"
            fi
        done
        if [[ -z "$greeting_source" && -z "$model_weight" ]]; then
            run_env+=("FDB_OPENING_GREETING=0")
            greeting_source="auto: no model_weight -> unmodified HF checkpoint (0)"
        fi
    fi

    echo
    echo "===== [$TOTAL] model_id=$model_id ====="
    echo "model_weight: ${model_weight:-<hf default>}"
    echo "model_config: ${model_config:-<hf default>}"
    [[ -n "$extra_env" ]] && echo "extra_env:    $extra_env"
    [[ -n "$greeting_source" ]] && echo "opening_greeting: $greeting_source"

    model_start=$(date +%s)
    if env "${run_env[@]}" bash scripts/run_full_duplex_eval.sh; then
        status="ok"
    else
        status="FAILED"
        FAILED=$((FAILED + 1))
    fi
    model_end=$(date +%s)
    elapsed=$((model_end - model_start))
    echo "[batch] model_id=$model_id status=$status elapsed_sec=$elapsed"
    RESULT_LINES+=("$model_id|$status|$elapsed")
done < "$MODELS_FILE"

echo
echo "===== Batch summary ====="
printf '%-24s %-8s %s\n' "model_id" "status" "elapsed_sec"
for line in "${RESULT_LINES[@]}"; do
    IFS='|' read -r m s e <<< "$line"
    printf '%-24s %-8s %s\n' "$m" "$s" "$e"
done

uv run python eval/combine_full_duplex_summaries.py \
    --batch-dir "$BATCH_OUT_DIR" \
    --out "$BATCH_OUT_DIR/combined_summary.json" \
    || echo "[batch] WARN: combined_summary.json step failed (see above)" >&2

echo
echo "[batch] per-model results: $BATCH_OUT_DIR/<model_id>/"
echo "[batch] combined summary:  $BATCH_OUT_DIR/combined_summary.json"
echo "[batch] total: $TOTAL failed: $FAILED"
echo "finished_at: $(date -Iseconds)"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
