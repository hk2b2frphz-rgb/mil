#!/usr/bin/env bash
# Download SLURP metadata (GitHub) and real-recording audio (Zenodo) to
# ICASSP2027/data/slurp/. Run on a login node with outbound network (or
# source scripts/setup_proxy.sh first). ~2 GB download for slurp_real.
#
#   bash ICASSP2027/scripts/download_slurp.sh [DEST_DIR]
#
# Produces:
#   DEST_DIR/metadata/{train,devel,test}.jsonl
#   DEST_DIR/audio/slurp_real/*.flac
#
# License note: SLURP is CC BY 4.0 (audio) / CC BY-NC 4.0 (some parts) --
# check LICENSE in the upstream repo before redistribution.

set -euo pipefail

DEST="${1:-ICASSP2027/data/slurp}"
META_BASE="https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp"
# Official SLURP audio release (verify the record is still current:
# https://zenodo.org/record/4274930).
AUDIO_URL="https://zenodo.org/record/4274930/files/slurp_real.tar.gz"

mkdir -p "$DEST/metadata" "$DEST/audio"

for split in train devel test; do
    if [[ -s "$DEST/metadata/${split}.jsonl" ]]; then
        echo "[slurp] metadata ${split}.jsonl exists, skipping"
    else
        echo "[slurp] downloading ${split}.jsonl"
        curl -fL --retry 3 -o "$DEST/metadata/${split}.jsonl" \
            "$META_BASE/${split}.jsonl"
    fi
done

if compgen -G "$DEST/audio/slurp_real/*.flac" >/dev/null; then
    echo "[slurp] audio already extracted, skipping"
else
    echo "[slurp] downloading slurp_real.tar.gz (large)"
    curl -fL --retry 3 -o "$DEST/audio/slurp_real.tar.gz" "$AUDIO_URL"
    echo "[slurp] extracting"
    tar -xzf "$DEST/audio/slurp_real.tar.gz" -C "$DEST/audio"
    rm -f "$DEST/audio/slurp_real.tar.gz"
fi

echo "[slurp] done."
echo "  metadata: $DEST/metadata/test.jsonl"
echo "  audio:    $DEST/audio/slurp_real"
echo "Build the benchmark with:"
echo "  qsub -v BENCH_ID=bench_en_slurp,LANGUAGE=en,CORPUS=slurp,\\"
echo "  SLURP_JSONL=$DEST/metadata/test.jsonl,SLURP_AUDIO_DIR=$DEST/audio/slurp_real \\"
echo "  ICASSP2027/pbs/p1_build_benchmark.pbs"
