#!/usr/bin/env bash
# Prepare and queue all three training chains. Evaluation is submitted by each
# final training link, so it cannot accidentally run after an intermediate link.
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"
unset ARM CHAIN_INDEX
source scripts/2026-10-07/common.sh
mkdir -p "$COMPARE_ROOT/jobs"
submit() {
    local file="$1" deps="${2:-}" job
    local args=(-V -v "COMPARE_ID=$COMPARE_ID,COMPARE_N=$COMPARE_N")
    [[ -z "$deps" ]] || args+=(-W "depend=afterok:$deps")
    job="$(qsub "${args[@]}" "scripts/2026-10-07/$file.pbs")"
    [[ "$job" =~ ^[0-9]+([.][A-Za-z0-9_.-]+)?$ ]] || { echo "Invalid qsub result: $job" >&2; return 1; }
    printf '%s\t%s\n' "$file" "$job" >> "$COMPARE_ROOT/jobs/submissions.tsv"
    printf '%s' "$job"
}
source_job="$(submit prepare_dialogues)"
bank_job="$(submit prepare_bank)"
ai_job="$(submit prepare_ai_placement "$source_job")"
condition_jobs=()
for arm in real_v2 traditional_overlap ai_placement; do
    deps="$source_job"
    [[ "$arm" != real_v2 ]] || deps="$deps:$bank_job"
    [[ "$arm" != ai_placement ]] || deps="$deps:$ai_job"
    render_job="$(submit "render_$arm" "$deps")"
    condition_jobs+=("$(submit "condition_$arm" "$render_job")")
done
IFS=:; deps="${condition_jobs[*]}"; unset IFS
pair_job="$(submit assemble_paired_data "$deps")"
for arm in real_v2 traditional_overlap ai_placement; do
    job="$(submit "train_$arm" "$pair_job")"
    echo "$arm: first training job $job"
done
echo "Submission log: $COMPARE_ROOT/jobs/submissions.tsv"
