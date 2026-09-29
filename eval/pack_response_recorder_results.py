#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack response_recorder.py output into compact evaluation JSONL."
    )
    parser.add_argument("--recorder-dir", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_scenarios(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    scenarios = list(iter_jsonl(path))
    by_text = {str(item.get("user_text", "")): item for item in scenarios}
    return scenarios, by_text


def trial_dirs(recorder_dir: Path) -> list[Path]:
    return sorted(path.parent for path in recorder_dir.glob("**/meta.json"))


def compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "input_path",
        "prompt_text",
        "input_duration_sec",
        "silence_sec",
        "seed",
        "model_repo",
        "dtype",
        "device",
        "temp",
        "temp_text",
        "cfg_coef",
        "frame_rate",
        "sample_rate",
        "first_response_latency_sec",
        "audible_response_start_sec",
        "audible_start_after_input_sec",
        "wall_time_sec",
        "output_audio_sec",
    ]
    return {key: meta.get(key) for key in keys if key in meta}


def main() -> int:
    args = parse_args()
    recorder_dir = args.recorder_dir.resolve()
    scenarios, by_text = load_scenarios(args.scenarios)
    rows = []

    for index, directory in enumerate(trial_dirs(recorder_dir), start=1):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8-sig"))
        user_text = str(meta.get("prompt_text") or "")
        scenario = by_text.get(user_text)
        if scenario is None and index <= len(scenarios):
            scenario = scenarios[index - 1]
        scenario = scenario or {}

        assistant_text = read_text(directory / "transcript.txt")
        response_audio = directory / "response.wav"
        seed = meta.get("seed")
        case_id = scenario.get("id") or f"case_{index:04d}"
        row = {
            "id": f"{args.system_id}:{case_id}:seed_{seed}",
            "system_id": args.system_id,
            "case_id": case_id,
            "category": scenario.get("category"),
            "risk_level": scenario.get("risk_level"),
            "user_text": user_text or scenario.get("user_text"),
            "assistant_text": assistant_text,
            "expected_behavior": scenario.get("expected_behavior"),
            "good_signals": scenario.get("good_signals", []),
            "avoid_behavior": scenario.get("avoid_behavior", []),
            "evaluator_notes": scenario.get("evaluator_notes"),
            "response_audio_path": str(response_audio) if response_audio.exists() else None,
            "trial_dir": str(directory),
            "meta": compact_meta(meta),
            "conversation": [
                {"role": "user", "content": user_text or scenario.get("user_text", "")},
                {"role": "assistant", "content": assistant_text},
            ],
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[pack] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
