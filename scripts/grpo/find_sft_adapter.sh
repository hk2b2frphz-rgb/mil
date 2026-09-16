#!/usr/bin/env bash
# Print the newest eligible SFT adapter; diagnostics go to stderr.
set -euo pipefail
ROOT="${1:?Usage: find_sft_adapter.sh SEARCH_ROOT}"
test -d "$ROOT" || { echo "ERROR: SFT search directory missing: $ROOT" >&2; exit 1; }
ROOT="$(cd "$ROOT" && pwd)"
# NUL records preserve spaces in paths. Sort by write time, then path for ties.
# Numeric checkpoint names distinguish SFT from checkpoint_epoch_* GRPO saves.
CANDIDATES="$(mktemp)"
trap 'rm -f -- "$CANDIDATES"' EXIT
find "$ROOT" -type d -iname '*grpo*' -prune -o \
    -type f -name lora.safetensors -size +0c -printf '%T@ %p\0' \
    | LC_ALL=C sort -z -k1,1nr -k2 > "$CANDIDATES"
while IFS= read -r -d '' RECORD; do
    ADAPTER="${RECORD#* }"
    [[ "$ADAPTER" =~ /checkpoint_[0-9]+/consolidated/lora\.safetensors$ ]] || continue
    [[ -s "$(dirname "$ADAPTER")/config.json" ]] || continue
    printf '%s\n' "$ADAPTER"
    exit 0
done < "$CANDIDATES"
echo "ERROR: no nonempty SFT adapter with config.json found under $ROOT." >&2
echo "Expected checkpoint_<step>/consolidated/lora.safetensors; GRPO is excluded." >&2
echo "Set SFT_SEARCH_ROOT to your SFT directory, or SFT_LORA_CKPT to an explicit adapter." >&2
exit 1
