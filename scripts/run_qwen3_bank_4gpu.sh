#!/usr/bin/env bash
set -euo pipefail

# Build a large backchannel bank across four GPUs.
#
#   bash scripts/run_qwen3_bank_4gpu.sh
#   REPEATS=400 bash scripts/run_qwen3_bank_4gpu.sh
#
# scripts/qwen3_tts_diversity_probe.py drives one engine on one GPU. This runs
# four of them, one per GPU, each with its own seed and its own output
# directory, and then writes a single samples.jsonl at the root whose wav paths
# point into the shards. The result is an ordinary bank directory: anything
# that reads a bank (splice_aizuchi_bank.py) takes it unchanged.
#
# Sharding is by seed, not by text: every shard draws the whole vocabulary, so
# a shard failing costs coverage of nothing, only depth. REPEATS is per shard,
# so four shards at 400 over seven words is 11,200 clips.
#
# At the measured 0.355 s/sample that is about twenty minutes of wall clock per
# shard, plus the engine load. The per-shard reports are kept; the merged bank
# has no report of its own, because the diversity numbers are per-engine and
# averaging them would hide a shard that went wrong. Re-measure the merge with
# --reanalyze if a single set of numbers is wanted.
#
# Overrides: REPEATS, NUM_SHARDS, CUDA_VISIBLE_DEVICES, TEXTS_FILE, TEXTS,
# TEMPERATURE, TEMPERATURE_SWEEP, OUT_ROOT, BATCH_ID, CLONE_OUT_DIR_MOSHI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PBS_O_WORKDIR="$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/run_id_utils.sh"

REPEATS="${REPEATS:-400}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STAMP="$(run_id_stamp)"
BATCH_ID="${BATCH_ID:-qwen3_bank_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/runs/diversity/$BATCH_ID}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" -lt "$NUM_SHARDS" ]]; then
    echo "ERROR: ${#GPU_IDS[@]} GPU(s) for NUM_SHARDS=$NUM_SHARDS" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT/logs"
echo "===== Qwen3 bank (${NUM_SHARDS} GPU) ====="
echo "repeats/shard: $REPEATS"
echo "out:           $OUT_ROOT"
echo "started_at:    $(date -Iseconds)"
echo "========================================="

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    shard="shard_$(printf '%03d' "$i")"
    (
        export CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}"
        export REPEATS SEED="$i"
        export OUT_DIR="$OUT_ROOT/$shard"
        export BATCH_ID="${BATCH_ID}_${shard}"
        bash scripts/run_qwen3_tts_diversity.sh
    ) >"$OUT_ROOT/logs/${shard}.log" 2>&1 &
    pids+=("$!")
    echo "[launch] $shard on GPU ${GPU_IDS[$i]} -> $OUT_ROOT/logs/${shard}.log"
done

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: shard $i failed. See $OUT_ROOT/logs/" >&2
        failed=1
    fi
done

# Merge whatever finished. A failed shard costs depth, not coverage, so a
# partial bank is still usable -- but say so rather than letting it pass as a
# full one.
MERGED="$OUT_ROOT/samples.jsonl"
: >"$MERGED"
merged_shards=0
for ((i = 0; i < NUM_SHARDS; i++)); do
    shard="shard_$(printf '%03d' "$i")"
    src="$OUT_ROOT/$shard/samples.jsonl"
    [[ -s "$src" ]] || continue
    uv run python - "$src" "$shard" "$MERGED" <<'PY'
import json
import sys
from pathlib import Path

source, prefix, target = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
with source.open("r", encoding="utf-8") as handle, target.open("a", encoding="utf-8") as out:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        # wav paths become relative to the merged root, so the bank reads like
        # any single-shard one.
        entry["wav"] = f"{prefix}/{entry['wav']}"
        entry["shard"] = prefix
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
    merged_shards=$((merged_shards + 1))
done

TOTAL="$(wc -l <"$MERGED" | tr -d ' ')"
echo
echo "bank:    $OUT_ROOT"
echo "clips:   $TOTAL (from $merged_shards/$NUM_SHARDS shards)"
if [[ "$merged_shards" -lt "$NUM_SHARDS" ]]; then
    echo "WARNING: only $merged_shards shard(s) merged; the bank is shallower than asked for." >&2
fi
echo "measure: uv run python scripts/qwen3_tts_diversity_probe.py --out-dir $OUT_ROOT --reanalyze"
echo "finished_at: $(date -Iseconds)"
exit "$failed"
