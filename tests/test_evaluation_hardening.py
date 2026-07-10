from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
for path in (ROOT, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.evaluate_full_duplex_adaptation import holm_adjust, paired_summary
from eval.evaluate_full_duplex_ja import aggregate, occurrence_distribution
from eval.full_duplex_judge_input import compact_event_segments
from eval.judge_full_duplex_azure import validate as validate_full_duplex
from eval.judge_llmjp_style import validate as validate_llmjp
from eval.judge_openai import validate_judge
from eval.pairwise_openai import load_by_key


def test_event_segments_keep_straddling_chunk_in_during() -> None:
    chunks = [
        {"text": "before", "timestamp": [0.0, 0.5]},
        {"text": "during", "timestamp": [0.4, 0.8]},
        {"text": "after", "timestamp": [0.8, 1.0]},
    ]
    assert compact_event_segments(chunks, (0.5, 0.8)) == {
        "before": "before",
        "during": "during",
        "after": "after",
    }


def test_full_duplex_judge_contract_requires_overlap_action() -> None:
    raw = {
        "scores": {
            "contextual_relevance": 5,
            "interruption_handling": 5,
            "topic_stability": 5,
            "empathy_acknowledgement": 5,
            "safety_boundary": 5,
            "conversation_naturalness": 5,
            "backchannel_naturalness": 5,
            "overall": 5,
        },
        "flags": {
            "ignored_interruption": False,
            "topic_drift": False,
            "unsafe": False,
            "empty_or_failed": False,
        },
        "overlap_action": "C_RESPOND",
    }
    validate_full_duplex(raw, "user_backchannel")
    raw["overlap_action"] = None
    with pytest.raises(ValueError, match="overlap_action"):
        validate_full_duplex(raw, "user_backchannel")


def test_general_judge_contract_rejects_missing_axes() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_judge({"scores": {}, "flags": {}})
    with pytest.raises(ValueError, match="missing required"):
        validate_llmjp({"scores": {}, "flags": {}})


def test_pairwise_rejects_duplicate_case_seed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    row = {"case_id": "case", "meta": {"seed": 0}}
    path.write_text("\n".join(json.dumps(row) for _ in range(2)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_by_key(path)


def test_summary_includes_numeric_metric_denominators() -> None:
    rows = [
        {"task": "x", "metrics": {"TOR": 1, "response_latency_sec": 0.5, "vad_backend": "silero"}},
        {"task": "x", "metrics": {"TOR": 0, "response_latency_sec": None, "vad_backend": "energy_fallback"}},
    ]
    summary = aggregate(rows)["x"]
    assert summary["metric_counts"] == {"TOR": 2, "response_latency_sec": 1}
    assert summary["observed_values"]["vad_backend"] == ["energy_fallback", "silero"]


def test_jsd_distribution_matches_upstream_boundary_bucket() -> None:
    # Exact 1.0 seconds yields six 0.2-second buckets in upstream v1/v1.5.
    assert len(occurrence_distribution([], 1.0)) == 6


def test_adaptation_holm_and_paired_summary() -> None:
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    rows = [
        {"pre": {"wpm": 1.0}, "post": {"wpm": 2.0}, "clean": {"wpm": 1.5}},
        {"pre": {"wpm": 2.0}, "post": {"wpm": 3.2}, "clean": {"wpm": 2.5}},
    ]
    assert paired_summary(rows, "pre", "post")["wpm"]["n"] == 2


def test_expanded_full_duplex_set_has_50_cases_per_task() -> None:
    source = ROOT / "eval_sets" / "full_duplex_ja" / "scenarios_expanded.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
        assert row["timeline"]
        assert row["expected_behavior"]
        assert row["avoid_behavior"]
    assert len(rows) == 350
    assert set(counts.values()) == {50}
    assert len({row["id"] for row in rows}) == len(rows)


def test_expanded_overlap_cases_have_broad_event_and_timing_variation() -> None:
    source = ROOT / "eval_sets" / "full_duplex_ja" / "scenarios_expanded.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line]
    for task in ("user_backchannel", "talking_to_other", "background_speech"):
        generated = [row for row in rows if row["task"] == task and row.get("scenario_set") == "expanded_v1"]
        event_texts = {
            next(item["text"] for item in row["timeline"] if item["type"] == "overlap_speech")
            for row in generated
        }
        response_windows = {
            next(item["duration_sec"] for item in row["timeline"] if item.get("event") == "response_window")
            for row in generated
        }
        assert len(event_texts) >= 12
        assert len(response_windows) >= 10
