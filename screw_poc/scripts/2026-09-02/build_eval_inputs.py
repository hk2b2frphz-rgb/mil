#!/usr/bin/env python3
"""Render held-out direct screw questions as Qwen3-TTS user-audio inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Running this file directly puts only its own directory on sys.path, so the
# repository root has to be added before scripts.* can be imported (same
# pattern as scripts/build_content_seeds.py).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_qwen3_tts_data import Qwen3TTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Direct questions and explicit out-of-scope requests can be assessed from
    # one user input. Multi-turn clarification cases remain in the full 200
    # dialogue reference but need conversational context and are excluded here.
    eligible = [
        row for row in source_rows
        if row.get("flow") in {"direct", "out_of_scope"}
    ]
    # Always cover the 30 knowledge rows before spending the remaining budget
    # on domain-boundary behaviour.
    eligible.sort(
        key=lambda row: (
            row.get("flow") == "out_of_scope",
            str(row.get("knowledge_id") or ""),
            str(row.get("id") or ""),
        )
    )
    rows = eligible[: args.limit]
    if len(rows) != args.limit:
        raise SystemExit(f"requested {args.limit} direct queries, found {len(rows)}")

    inputs = args.out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    reference: list[dict[str, object]] = []
    tts = Qwen3TTS(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device=args.device,
        dtype_str=args.dtype,
        attn_impl="default",
        speaker_user="Serena",
        speaker_moshi="Ono_Anna",
        language="Japanese",
        instruct_user=None,
        instruct_moshi=None,
    )
    import sphn  # type: ignore[import]
    for index, row in enumerate(rows):
        turns = list(row["turns"])
        user = next(turn for turn in turns if turn.get("speaker") == "user")
        moshi = next(turn for turn in turns if turn.get("speaker") == "moshi")
        query_id = str(row["id"])
        text = str(user.get("tts_text") or user["text"])
        pcm = np.asarray(
            tts.synthesize(text, "user", speaker_override=("Serena", "Sohee", "Vivian", "Dylan")[index % 4]),
            dtype=np.float32,
        ).reshape(-1)
        wav_path = inputs / f"{query_id}.wav"
        sphn.write_wav(str(wav_path), pcm, tts.sample_rate)
        wav_path.with_suffix(".txt").write_text(str(user["text"]) + "\n", encoding="utf-8")
        reference.append({
            "id": query_id,
            "knowledge_id": row.get("knowledge_id"),
            "flow": row.get("flow"),
            "expected_answer": row.get("expected_answer"),
            "expected_response_patterns": [moshi.get("response_pattern")],
            "turns": [
                {"speaker": "user", "text": user["text"]},
                {"speaker": "moshi", "text": moshi["text"]},
            ],
        })
    (args.out_dir / "reference.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reference),
        encoding="utf-8",
    )
    print(json.dumps({"queries": len(reference), "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
