#!/usr/bin/env python3
"""Reject cached FDB-JA audio rendered with different benchmark inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def scenarios_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--tts-backend", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--tts-speaker", required=True)
    parser.add_argument("--tts-background-speaker", required=True)
    parser.add_argument("--tts-speed", type=float, required=True)
    parser.add_argument("--whole-utterance", type=int, choices=[0, 1], required=True)
    parser.add_argument("--whole-utterance-max-chars", type=int, required=True)
    parser.add_argument("--opening-greeting", type=int, choices=[0, 1], required=True)
    parser.add_argument("--opening-greeting-gap-sec", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Cached dataset has no manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    config = manifest.get("dataset_config")
    if not isinstance(config, dict):
        raise SystemExit("Cached dataset predates dataset_config; rebuild it once.")
    expected = {
        "tts_backend_requested": args.tts_backend,
        "tts_model": args.tts_model,
        "tts_speaker": args.tts_speaker,
        "tts_background_speaker": args.tts_background_speaker,
        "tts_speed": args.tts_speed,
        "whole_utterance_requested": bool(args.whole_utterance),
        "whole_utterance_max_chars": args.whole_utterance_max_chars,
        "opening_greeting_enabled": bool(args.opening_greeting),
        "opening_greeting_gap_sec": args.opening_greeting_gap_sec,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    scenario_hash = scenarios_sha256(args.scenarios)
    if manifest.get("scenarios_sha256") != scenario_hash:
        mismatches["scenarios_sha256"] = {
            "expected": scenario_hash,
            "actual": manifest.get("scenarios_sha256"),
        }
    if mismatches:
        raise SystemExit("Cached dataset fingerprint mismatch: " + json.dumps(mismatches, ensure_ascii=False))
    print(f"[fdb-data] verified cached dataset: {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
