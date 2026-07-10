#!/usr/bin/env bash
# Runs scripts/run_full_duplex_eval.sh once per model listed in MODELS_FILE,
# writing each model's results under BATCH_OUT_DIR/<output_name>/ and a final
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
OUTPUT_NAMES=()
MODEL_IDS=()
TOTAL=0
FAILED=0
MANIFEST_ERRORS=0

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
PARALLELISM="${FDB_BATCH_PARALLELISM:-${#GPU_IDS[@]}}"
if ! [[ "$PARALLELISM" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FDB_BATCH_PARALLELISM must be a positive integer: $PARALLELISM" >&2
    exit 1
fi
if [[ "$PARALLELISM" -gt "${#GPU_IDS[@]}" ]]; then
    echo "ERROR: FDB_BATCH_PARALLELISM=$PARALLELISM exceeds visible GPU count ${#GPU_IDS[@]} ($CUDA_VISIBLE_DEVICES)." >&2
    exit 1
fi
# REFRESH_FDB_DATA deliberately rebuilds shared inputs. Keep that exceptional
# mode serial so no model can read the dataset while another rebuilds it.
if [[ "${REFRESH_FDB_DATA:-0}" == "1" && "$PARALLELISM" -gt 1 ]]; then
    echo "[batch] REFRESH_FDB_DATA=1: forcing serial execution for dataset consistency."
    PARALLELISM=1
fi
echo "gpus:         ${GPU_IDS[*]}"
echo "parallelism:  $PARALLELISM"

WAVE_PIDS=()

wait_for_wave() {
    local pid
    for pid in "${WAVE_PIDS[@]}"; do
        # Workers always record their own status. A missing status file after
        # an external kill is handled as FAILED in the final collection.
        wait "$pid" || true
    done
    WAVE_PIDS=()
}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue

    IFS='|' read -r model_id model_weight model_config hf_repo extra_env opening_greeting output_name <<< "$line"
    model_id="$(echo "$model_id" | xargs)"
    model_weight="$(echo "$model_weight" | xargs)"
    model_config="$(echo "$model_config" | xargs)"
    hf_repo="$(echo "$hf_repo" | xargs)"
    extra_env="$(echo "$extra_env" | xargs)"
    opening_greeting="$(echo "$opening_greeting" | xargs)"
    output_name="$(echo "${output_name:-}" | xargs)"

    # Backward-compatible shorthand: in a 6-column manifest, a final value
    # other than 0/1 is treated as output_name, not opening_greeting.
    if [[ -z "$output_name" && -n "$opening_greeting" && "$opening_greeting" != "0" && "$opening_greeting" != "1" ]]; then
        output_name="$opening_greeting"
        opening_greeting=""
    fi
    if [[ -n "$opening_greeting" && "$opening_greeting" != "0" && "$opening_greeting" != "1" ]]; then
        echo "[batch] WARN: opening_greeting must be 0 or 1 for $model_id, ignoring invalid value: $opening_greeting" >&2
        opening_greeting=""
    fi

    if [[ -z "$model_id" ]]; then
        echo "[batch] WARN: skipping manifest line with empty model_id: $raw_line" >&2
        MANIFEST_ERRORS=1
        continue
    fi
    if ! [[ "$model_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[batch] WARN: skipping invalid model_id (letters/digits/dot/underscore/hyphen only): $model_id" >&2
        MANIFEST_ERRORS=1
        continue
    fi
    output_name="${output_name:-$model_id}"
    if ! [[ "$output_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[batch] WARN: skipping invalid output_name for $model_id (letters/digits/dot/underscore/hyphen only): $output_name" >&2
        MANIFEST_ERRORS=1
        continue
    fi
    duplicate_output=0
    for seen_output_name in "${OUTPUT_NAMES[@]}"; do
        if [[ "$seen_output_name" == "$output_name" ]]; then
            duplicate_output=1
            break
        fi
    done
    if [[ "$duplicate_output" == "1" ]]; then
        echo "[batch] WARN: skipping duplicate output_name (would overwrite previous result): $output_name" >&2
        MANIFEST_ERRORS=1
        continue
    fi
    OUTPUT_NAMES+=("$output_name")
    MODEL_IDS+=("$model_id")

    TOTAL=$((TOTAL + 1))

    run_env=(
        "MODEL_ID=$model_id"
        "RUN_ID=$output_name"
        "FDB_OUT_DIR=$BATCH_OUT_DIR/$output_name"
    )
    [[ -n "$model_weight" ]] && run_env+=("MODEL_WEIGHT=$model_weight")
    [[ -n "$model_config" ]] && run_env+=("MODEL_CONFIG=$model_config")
    [[ -n "$hf_repo" ]] && run_env+=("HF_REPO=$hf_repo")
    # Which eval pipeline to run for this row. Default "moshi" runs the Moshi
    # full-duplex script; "cascade" (set via FDB_SYSTEM=cascade in extra_env)
    # runs the local ASR->LLM->TTS baseline instead. Both write the same
    # $FDB_OUT_DIR/benchmark_results/summary.json, so combine_full_duplex_
    # summaries.py lists them side by side with no special-casing.
    row_system="moshi"
    if [[ -n "$extra_env" ]]; then
        IFS=';' read -ra pairs <<< "$extra_env"
        for pair in "${pairs[@]}"; do
            pair="$(echo "$pair" | xargs)"
            [[ -z "$pair" ]] && continue
            if [[ "$pair" == FDB_SYSTEM=* ]]; then
                row_system="${pair#FDB_SYSTEM=}"
                continue
            fi
            run_env+=("$pair")
        done
    fi
    if [[ "$row_system" != "moshi" && "$row_system" != "cascade" ]]; then
        echo "[batch] WARN: unknown FDB_SYSTEM='$row_system' for $model_id, treating as moshi" >&2
        row_system="moshi"
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

    if [[ "$row_system" == "cascade" ]]; then
        eval_script="scripts/run_full_duplex_cascade_eval.sh"
    else
        eval_script="scripts/run_full_duplex_eval.sh"
    fi

    echo
    echo "===== [$TOTAL] model_id=$model_id output_name=$output_name ====="
    echo "system:       $row_system"
    [[ "$row_system" == "moshi" ]] && echo "model_weight: ${model_weight:-<hf default>}"
    [[ "$row_system" == "moshi" ]] && echo "model_config: ${model_config:-<hf default>}"
    [[ "$output_name" != "$model_id" ]] && echo "output_name:  $output_name"
    [[ -n "$extra_env" ]] && echo "extra_env:    $extra_env"
    [[ -n "$greeting_source" ]] && echo "opening_greeting: $greeting_source"

    row_requires_serial=0
    for kv in "${run_env[@]}"; do
        [[ "$kv" == "REFRESH_FDB_DATA=1" ]] && row_requires_serial=1
    done
    if [[ "$row_requires_serial" == "1" ]]; then
        # A row-level refresh can overwrite a dataset that other rows read.
        # Drain all workers and run this one alone.
        wait_for_wave
    fi

    # Fill one GPU wave at a time. Every worker is a separate process with a
    # single visible GPU, so model state and random-number streams remain as
    # isolated as in the former serial execution.
    if [[ "${#WAVE_PIDS[@]}" -ge "$PARALLELISM" ]]; then
        wait_for_wave
    fi
    gpu_id="${GPU_IDS[${#WAVE_PIDS[@]}]}"
    status_file="$BATCH_OUT_DIR/.${output_name}.batch_status"
    rm -f "$status_file"
    mkdir -p "$BATCH_OUT_DIR/$output_name"
    # Never let a failed rerun contribute a stale summary from an older run.
    rm -f "$BATCH_OUT_DIR/$output_name/benchmark_results/summary.json"
    (
        model_start=$(date +%s)
        if env CUDA_VISIBLE_DEVICES="$gpu_id" "${run_env[@]}" bash "$eval_script"; then
            run_config="$BATCH_OUT_DIR/$output_name/inference/run_config.json"
            if [[ -f "$run_config" ]] && grep -q "\"system\": \"$row_system\"" "$run_config"; then
                status="ok"
            else
                echo "[batch] ERROR: expected system=$row_system in $run_config" >&2
                status="FAILED"
            fi
        else
            status="FAILED"
        fi
        model_end=$(date +%s)
        elapsed=$((model_end - model_start))
        printf '%s|%s\n' "$status" "$elapsed" > "$status_file"
        echo "[batch] model_id=$model_id output_name=$output_name system=$row_system gpu=$gpu_id status=$status elapsed_sec=$elapsed"
    ) > >(tee -a "$BATCH_OUT_DIR/$output_name/batch_worker.log") 2>&1 &
    WAVE_PIDS+=("$!")
    echo "[batch] started model_id=$model_id on gpu=$gpu_id pid=$!"
    if [[ "$row_requires_serial" == "1" ]]; then
        wait_for_wave
    fi
done < "$MODELS_FILE"

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: manifest contains no valid model rows: $MODELS_FILE" >&2
    exit 1
fi

wait_for_wave

for i in "${!OUTPUT_NAMES[@]}"; do
    model_id="${MODEL_IDS[$i]}"
    output_name="${OUTPUT_NAMES[$i]}"
    status_file="$BATCH_OUT_DIR/.${output_name}.batch_status"
    if [[ -f "$status_file" ]]; then
        IFS='|' read -r status elapsed < "$status_file"
        rm -f "$status_file"
    else
        status="FAILED"
        elapsed="unknown"
    fi
    if [[ "$status" != "ok" ]]; then
        FAILED=$((FAILED + 1))
    fi
    RESULT_LINES+=("$model_id|$output_name|$status|$elapsed")
done

echo
echo "===== Batch summary ====="
printf '%-24s %-24s %-8s %s\n' "model_id" "output_name" "status" "elapsed_sec"
for line in "${RESULT_LINES[@]}"; do
    IFS='|' read -r m o s e <<< "$line"
    printf '%-24s %-24s %-8s %s\n' "$m" "$o" "$s" "$e"
done

if ! uv run python eval/combine_full_duplex_summaries.py \
    --batch-dir "$BATCH_OUT_DIR" \
    --out "$BATCH_OUT_DIR/combined_summary.json"; then
    echo "[batch] ERROR: combined_summary.json step failed" >&2
    FAILED=$((FAILED + 1))
fi

echo
echo "[batch] per-model results: $BATCH_OUT_DIR/<output_name>/"
echo "[batch] combined summary:  $BATCH_OUT_DIR/combined_summary.json"
echo "[batch] total: $TOTAL failed: $FAILED"
echo "finished_at: $(date -Iseconds)"

if [[ "$FAILED" -gt 0 || "$MANIFEST_ERRORS" -gt 0 ]]; then
    exit 1
fi
