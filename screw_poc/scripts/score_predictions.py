#!/usr/bin/env python3
"""Score ASR transcripts or text-model predictions against the eval set.

Prediction JSONL format:
  {"id": "evaluation_k01_000", "response_texts": ["...", "..."]}

For a final-response-only evaluation, use ``response_text`` instead.  Optional
``response_patterns`` can be supplied when a separate response-policy
classifier is being evaluated.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


POC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = POC_ROOT / "artifacts" / "evaluation_dialogues.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s、。,.!?！？・:：;；\-ー]", "", text)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def expected_texts(row: dict[str, Any]) -> list[str]:
    return [
        turn["text"] for turn in row["turns"] if turn.get("speaker") == "moshi"
    ]


def predicted_texts(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("response_texts"), list):
        return [str(text) for text in row["response_texts"]]
    if row.get("response_text") is not None:
        return [str(row["response_text"])]
    return []


def score(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_by_id = {row["id"]: row for row in references}
    prediction_by_id = {row["id"]: row for row in predictions}
    unknown_ids = sorted(set(prediction_by_id) - set(reference_by_id))
    matched_ids = sorted(set(prediction_by_id) & set(reference_by_id))

    final_exact = 0
    sequence_exact = 0
    total_edits = 0
    total_reference_chars = 0
    pattern_available = 0
    pattern_exact = 0
    details: list[dict[str, Any]] = []
    for dialogue_id in matched_ids:
        reference = reference_by_id[dialogue_id]
        prediction = prediction_by_id[dialogue_id]
        expected = expected_texts(reference)
        actual = predicted_texts(prediction)
        expected_norm = [normalize(text) for text in expected]
        actual_norm = [normalize(text) for text in actual]

        # response_text means final-only scoring; response_texts means sequence scoring.
        actual_final = actual_norm[-1] if actual_norm else ""
        expected_final = expected_norm[-1] if expected_norm else ""
        is_final_exact = bool(actual_final) and actual_final == expected_final
        is_sequence_exact = "response_texts" in prediction and actual_norm == expected_norm
        final_exact += is_final_exact
        sequence_exact += is_sequence_exact
        total_edits += edit_distance(expected_final, actual_final)
        total_reference_chars += max(1, len(expected_final))

        if isinstance(prediction.get("response_patterns"), list):
            pattern_available += 1
            pattern_exact += prediction["response_patterns"] == reference[
                "expected_response_patterns"
            ]
        if not is_final_exact:
            details.append(
                {
                    "id": dialogue_id,
                    "expected_final": expected[-1] if expected else "",
                    "actual_final": actual[-1] if actual else "",
                }
            )

    matched = len(matched_ids)
    return {
        "reference_dialogues": len(references),
        "scored_dialogues": matched,
        "coverage": matched / len(references) if references else 0.0,
        "final_response_exact_accuracy": final_exact / matched if matched else 0.0,
        "full_response_sequence_exact_accuracy": sequence_exact / matched if matched else 0.0,
        "final_response_character_error_rate": (
            total_edits / total_reference_chars if matched else 0.0
        ),
        "response_pattern_sequence_accuracy": (
            pattern_exact / pattern_available if pattern_available else None
        ),
        "pattern_scored_dialogues": pattern_available,
        "missing_prediction_ids": sorted(set(reference_by_id) - set(prediction_by_id)),
        "unknown_prediction_ids": unknown_ids,
        "mismatches": details,
    }


def main() -> None:
    args = parse_args()
    result = score(read_jsonl(args.reference), read_jsonl(args.predictions))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["unknown_prediction_ids"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
