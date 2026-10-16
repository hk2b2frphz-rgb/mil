#!/usr/bin/env python3
"""Fail before splicing if any generated word has no matching bank audio."""
import json
from pathlib import Path
import sys


def check_bank(manifest, vocab):
    words = {line.split("\t")[0].strip() for line in vocab.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")}
    found = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wav = (manifest.parent / row["wav"]).resolve()
        if not wav.is_file():
            raise ValueError(f"Missing bank audio: {wav}")
        found.add(row["text"])
    if words - found:
        raise ValueError(f"Bank lacks exact-word clips: {sorted(words - found)}")
    print(f"[bank] all {len(words)} vocabulary entries have audio")


if __name__ == "__main__":
    check_bank(Path(sys.argv[1]), Path(sys.argv[2]))
