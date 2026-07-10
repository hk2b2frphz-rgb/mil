#!/usr/bin/env bash
# Runs the pinned, unmodified upstream FDB v1/v1.5 evaluation code against a
# Japanese-adapted data layout. Only the staging adapter changes language data.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${FDB_UPSTREAM_DIR:?Set FDB_UPSTREAM_DIR to Full-Duplex-Bench@3e799c45... checkout}"
RUN_DIR="${1:?pass inference run directory}"
OUT_DIR="${2:?pass official evaluation output directory}"
UP="$UPSTREAM_DIR/v1_v1.5"
[[ -f "$UP/evaluation/evaluate.py" && -f "$UP/get_transcript/asr.py" ]] || { echo "Invalid pinned upstream checkout" >&2; exit 1; }
python "$REPO_ROOT/eval/prepare_fdb_ja_official_layout.py" --run-dir "$RUN_DIR" --out-dir "$OUT_DIR" --overwrite
for task in pause_handling smooth_turn_taking backchannel user_interruption; do
  [[ -d "$OUT_DIR/$task" ]] || continue
  ( cd "$UP/evaluation"; python evaluate.py --task "$task" --root_dir "$OUT_DIR/$task" ) | tee "$OUT_DIR/${task}_official.log"
done
for task in user_interruption user_backchannel talking_to_other background_speech; do
  [[ -d "$OUT_DIR/$task" ]] || continue
  python "$UP/get_transcript/asr.py" --root_dir "$OUT_DIR/$task" --task default --audio_name output.wav
  python "$UP/get_transcript/asr.py" --root_dir "$OUT_DIR/$task" --task default --audio_name clean_output.wav
  ( cd "$UP/evaluation"; python get_timing.py --root_dir "$OUT_DIR/$task" ) | tee "$OUT_DIR/${task}_timing.log"
done
echo "[fdb-ja] upstream evaluation complete: $OUT_DIR"
