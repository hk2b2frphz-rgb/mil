"""Corpus loaders: MASSIVE (any locale, TTS-rendered) and SLURP (English,
real recordings).

Both corpora share the ``[slot_type : value]`` annotation format (MASSIVE
localized SLURP), so every loader returns the same row shape consumed by
`clarify.slots.select_base_items`:

    {"id", "intent", "annot_utt", "utt", "scenario",
     "source",                       # "massive" | "slurp"
     "audio_file": str | None}       # real recording (SLURP only)

MASSIVE loads from HF (`AmazonScience/massive`, CC BY 4.0; needs network/
proxy on the cluster) or from a pre-exported JSONL for offline nodes.
SLURP loads from a local checkout of the official metadata JSONL plus the
`slurp_real` audio directory (see ICASSP2027/scripts/download_slurp.sh).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# MASSIVE
# ---------------------------------------------------------------------------

def load_massive(locale: str, split: str = "test") -> list[dict[str, Any]]:
    """Load MASSIVE rows for a locale ('en-US', 'ja-JP', ...) via HF."""
    from datasets import load_dataset  # lazy; cluster-side

    ds = load_dataset("AmazonScience/massive", locale, split=split)
    intent_names = ds.features["intent"].names
    rows: list[dict[str, Any]] = []
    for raw in ds:
        row = dict(raw)
        if isinstance(row.get("intent"), int):
            row["intent"] = intent_names[row["intent"]]
        row["source"] = "massive"
        row["audio_file"] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SLURP (real English speech)
# ---------------------------------------------------------------------------

def _slurp_pick_recording(
    recordings: list[dict[str, Any]], audio_dir: Path,
    prefer_headset: bool = True,
) -> Path | None:
    """Pick one existing audio file for an utterance.

    SLURP distributes close-talk (headset) and far-field recordings; the
    close-talk one is the cleanest carrier for span-localized corruption
    (the corruption grid then adds degradation on top of clean speech).
    """
    candidates = []
    for rec in recordings or []:
        fname = rec.get("file")
        if not fname:
            continue
        path = audio_dir / fname
        if path.exists():
            candidates.append((fname, path))
    if not candidates:
        return None
    if prefer_headset:
        for fname, path in candidates:
            if "headset" in fname:
                return path
    return candidates[0][1]


def load_slurp(
    metadata_jsonl: Path,
    audio_dir: Path,
    require_audio: bool = True,
) -> list[dict[str, Any]]:
    """Load SLURP rows from the official metadata JSONL.

    Official schema per line: slurp_id, sentence, sentence_annotation
    (bracket format), scenario, action, recordings[{file, ...}].
    intent is scenario_action, matching MASSIVE intent names.
    """
    rows: list[dict[str, Any]] = []
    skipped_no_audio = 0
    for raw in load_jsonl_rows(metadata_jsonl):
        scenario = raw.get("scenario")
        action = raw.get("action")
        if not scenario or not action:
            continue
        audio = _slurp_pick_recording(raw.get("recordings", []), audio_dir)
        if audio is None and require_audio:
            skipped_no_audio += 1
            continue
        rows.append({
            "id": raw.get("slurp_id"),
            "intent": f"{scenario}_{action}",
            "annot_utt": raw.get("sentence_annotation") or "",
            "utt": raw.get("sentence") or "",
            "scenario": scenario,
            "source": "slurp",
            "audio_file": str(audio) if audio else None,
        })
    if skipped_no_audio:
        print(f"[corpora] slurp: skipped {skipped_no_audio} rows with no "
              f"local audio under {audio_dir}")
    return rows


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def load_corpus(
    corpus: str,
    language: str,
    split: str = "test",
    massive_jsonl: Path | None = None,
    slurp_jsonl: Path | None = None,
    slurp_audio_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Unified entry: corpus in {"massive", "slurp"}."""
    if corpus == "massive":
        if massive_jsonl:
            rows = load_jsonl_rows(massive_jsonl)
            for row in rows:
                row.setdefault("source", "massive")
                row.setdefault("audio_file", None)
            return rows
        from .lang import get_pack

        return load_massive(get_pack(language).massive_locale, split)
    if corpus == "slurp":
        if language != "en":
            raise ValueError("SLURP is English-only")
        if not slurp_jsonl or not slurp_audio_dir:
            raise ValueError(
                "corpus=slurp needs --slurp-jsonl and --slurp-audio-dir"
            )
        return load_slurp(Path(slurp_jsonl), Path(slurp_audio_dir))
    raise ValueError(f"unknown corpus {corpus!r}")
