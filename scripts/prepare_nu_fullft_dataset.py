#!/usr/bin/env python3
"""Prepare existing Moshi synthetic data for nu-dialogue/moshi-finetune.

The local data generator writes stereo wav files plus sidecar JSON files with
utterance-level alignments:

    [text, [start_sec, end_sec], "SPEAKER_MAIN" | "SPEAKER_USER"]

nu-dialogue/moshi-finetune expects per-dialogue stereo wav files and a JSON
transcript list using speakers "A" and "B". The generated audio convention in
this repository is left channel = moshi/main, right channel = user, so this
script maps SPEAKER_MAIN to A and SPEAKER_USER to B.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PreparedSample:
    source_wav: Path
    stem: str
    duration: float | None
    transcript: list[dict[str, str | float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synthetic_moshi_train.jsonl to nu-dialogue raw data."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "copy"],
        default="auto",
        help="How to place wav files into the nu raw data directory.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--no-normalize-text",
        action="store_true",
        help="Disable Japanese punctuation/ellipsis normalization for transcript text.",
    )
    parser.add_argument(
        "--normalization-examples",
        type=int,
        default=20,
        help="Number of before/after text normalization examples to keep in manifest.json.",
    )
    return parser.parse_args()


def speaker_from_label(label: str) -> str | None:
    normalized = label.strip().upper()
    if normalized in {"SPEAKER_MAIN", "MAIN", "MOSHI", "ASSISTANT", "A"}:
        return "A"
    if normalized in {"SPEAKER_USER", "USER", "CLIENT", "B"}:
        return "B"
    return None


def normalize_transcript_text(text: str) -> tuple[str, list[str]]:
    """Normalize Japanese dialogue punctuation before nu text tokenization."""
    rules: list[str] = []

    def remember(rule: str, before: str, after: str) -> str:
        if after != before and rule not in rules:
            rules.append(rule)
        return after

    text = text.strip()
    text = remember("nfkc", text, unicodedata.normalize("NFKC", text))

    before = text
    text = re.sub(r"(?:\.{2,}|…+|‥+|・{3,})", "…", text)
    text = remember("ellipsis", before, text)

    before = text
    text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
    text = remember("period_to_japanese_full_stop", before, text)

    before = text
    text = text.translate(
        str.maketrans(
            {
                ",": "、",
                "?": "？",
                "!": "！",
                ":": "：",
                ";": "；",
                "(": "（",
                ")": "）",
                "[": "［",
                "]": "］",
            }
        )
    )
    text = remember("ascii_punctuation_to_japanese", before, text)

    before = text
    text = re.sub(r"([。、！？])\1+", r"\1", text)
    text = remember("repeated_japanese_punctuation", before, text)

    before = text
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([。、！？：；…（）［］])\s*", r"\1", text)
    text = remember("whitespace_around_punctuation", before, text)

    return text, rules


def alignment_to_segment(
    item: Any,
    normalize_text_enabled: bool,
) -> tuple[dict[str, str | float], dict[str, Any] | None] | None:
    if isinstance(item, dict):
        text = str(item.get("word") or item.get("text") or "").strip()
        speaker = str(item.get("speaker") or item.get("label") or "").strip()
        start = item.get("start")
        end = item.get("end")
        if speaker not in {"A", "B"}:
            mapped = speaker_from_label(speaker)
            if mapped is None:
                return None
            speaker = mapped
    elif isinstance(item, list) and len(item) >= 3:
        text = str(item[0]).strip()
        span = item[1]
        label = str(item[2])
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            return None
        start, end = span[0], span[1]
        speaker = speaker_from_label(label)
        if speaker is None:
            return None
    else:
        return None

    if not text:
        return None
    raw_text = text
    rules: list[str] = []
    if normalize_text_enabled:
        text, rules = normalize_transcript_text(raw_text)
    if not text:
        return None
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        return None
    segment = {
        "speaker": speaker,
        "word": text,
        "start": round(start_f, 4),
        "end": round(end_f, 4),
    }
    change = (
        {"before": raw_text, "after": text, "rules": rules}
        if rules and raw_text != text
        else None
    )
    return segment, change


def read_samples(
    manifest: Path,
    normalize_text_enabled: bool,
    normalization_examples: int,
) -> tuple[list[PreparedSample], list[str], dict[str, Any]]:
    root = manifest.parent
    samples: list[PreparedSample] = []
    warnings: list[str] = []
    seen_stems: dict[str, int] = {}
    rule_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    changed_segments = 0

    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"line {lineno}: invalid JSON: {exc}")
            continue

        rel_path = row.get("path")
        if not rel_path:
            warnings.append(f"line {lineno}: missing path")
            continue
        wav_path = (root / str(rel_path)).resolve()
        json_path = wav_path.with_suffix(".json")
        if not wav_path.exists():
            warnings.append(f"line {lineno}: wav not found: {wav_path}")
            continue
        if not json_path.exists():
            warnings.append(f"line {lineno}: sidecar JSON not found: {json_path}")
            continue

        try:
            sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"line {lineno}: invalid sidecar JSON: {json_path}: {exc}")
            continue

        alignments = sidecar.get("alignments", [])
        transcript = []
        for item in alignments:
            converted = alignment_to_segment(item, normalize_text_enabled)
            if converted is None:
                continue
            segment, change = converted
            transcript.append(segment)
            if change is not None:
                changed_segments += 1
                rule_counts.update(change["rules"])
                if len(examples) < normalization_examples:
                    examples.append(
                        {
                            "wav": wav_path.name,
                            "speaker": segment["speaker"],
                            **change,
                        }
                    )
        transcript.sort(key=lambda item: (float(item["start"]), str(item["speaker"])))
        speakers = {str(item["speaker"]) for item in transcript}
        if not {"A", "B"}.issubset(speakers):
            warnings.append(
                f"line {lineno}: skipped {wav_path.name}; need both A(main) and B(user), "
                f"got {sorted(speakers)}"
            )
            continue

        base_stem = wav_path.stem
        count = seen_stems.get(base_stem, 0)
        seen_stems[base_stem] = count + 1
        stem = base_stem if count == 0 else f"{base_stem}_{count + 1:03d}"
        duration = row.get("duration")
        samples.append(
            PreparedSample(
                source_wav=wav_path,
                stem=stem,
                duration=float(duration) if duration is not None else None,
                transcript=transcript,
            )
        )

    normalization_summary = {
        "enabled": normalize_text_enabled,
        "version": 1,
        "changed_segments": changed_segments,
        "rule_counts": dict(sorted(rule_counts.items())),
        "examples": examples,
    }
    return samples, warnings, normalization_summary


def place_wav(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link_mode in {"auto", "hardlink"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def write_split(
    name: str,
    samples: list[PreparedSample],
    output_dir: Path,
    link_mode: str,
) -> dict[str, Any]:
    audio_dir = output_dir / "raw" / name / "audio"
    text_dir = output_dir / "raw" / name / "text"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, int] = {}
    durations = 0.0
    for sample in samples:
        method = place_wav(sample.source_wav, audio_dir / f"{sample.stem}.wav", link_mode)
        methods[method] = methods.get(method, 0) + 1
        (text_dir / f"{sample.stem}.json").write_text(
            json.dumps(sample.transcript, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if sample.duration is not None:
            durations += sample.duration

    return {
        "count": len(samples),
        "duration_sec": round(durations, 3),
        "audio_dir": str(audio_dir),
        "text_dir": str(text_dir),
        "placement": methods,
    }


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1
    if not (0.0 <= args.eval_fraction < 1.0):
        print("ERROR: --eval-fraction must be >= 0 and < 1", file=sys.stderr)
        return 1

    if args.refresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, warnings, normalization = read_samples(
        manifest=manifest,
        normalize_text_enabled=not args.no_normalize_text,
        normalization_examples=max(0, args.normalization_examples),
    )
    if len(samples) < 2:
        print(
            f"ERROR: need at least 2 usable samples for train/eval split, got {len(samples)}",
            file=sys.stderr,
        )
        for warning in warnings[:20]:
            print(f"WARNING: {warning}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.eval_fraction == 0.0:
        n_eval = 0
    else:
        n_eval = min(max(1, int(len(samples) * args.eval_fraction)), len(samples) - 1)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    summary: dict[str, Any] = {
        "source_manifest": str(manifest),
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "usable_count": len(samples),
        "skipped_count": len(warnings),
        "warnings": warnings[:200],
        "text_normalization": normalization,
        "splits": {
            "train": write_split("train", train_samples, output_dir, args.link_mode),
        },
    }
    if eval_samples:
        summary["splits"]["eval"] = write_split("eval", eval_samples, output_dir, args.link_mode)

    summary_path = output_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[nu-data] source_manifest={manifest}")
    print(f"[nu-data] output_dir={output_dir}")
    print(f"[nu-data] train={len(train_samples)} eval={len(eval_samples)} skipped={len(warnings)}")
    if normalization["enabled"]:
        print(
            "[nu-data] text normalization: "
            f"changed_segments={normalization['changed_segments']} "
            f"rules={normalization['rule_counts']}"
        )
        for example in normalization["examples"][:5]:
            print(
                "[nu-data] normalize: "
                f"{example['before']!r} -> {example['after']!r} "
                f"rules={example['rules']}"
            )
    if warnings:
        print(f"[nu-data] warnings written to {summary_path}")
        for warning in warnings[:10]:
            print(f"[nu-data] WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
