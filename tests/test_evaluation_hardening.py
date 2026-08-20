from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
for path in (ROOT, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.evaluate_full_duplex_adaptation import holm_adjust, paired_summary
from eval.evaluate_full_duplex_ja import aggregate, occurrence_distribution
import eval.evaluate_full_duplex_ja as full_duplex_eval
import eval.combine_full_duplex_summaries as combine_summaries
from eval.full_duplex_judge_input import compact_event_segments
from eval.full_duplex_sample_selection import select_samples
from eval.judge_full_duplex_azure import (
    RUBRIC as FULL_DUPLEX_RUBRIC,
    SCORE_KEYS as FULL_DUPLEX_SCORE_KEYS,
    SYSTEM_PROMPT as FULL_DUPLEX_SYSTEM_PROMPT,
    prompt as full_duplex_prompt,
    validate as validate_full_duplex,
)
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


def test_full_duplex_rubric_gives_every_axis_scale_anchors() -> None:
    # A bare "1-5" range lets the same axis drift between judge models and runs.
    assert tuple(FULL_DUPLEX_RUBRIC) == FULL_DUPLEX_SCORE_KEYS
    for axis, anchors in FULL_DUPLEX_RUBRIC.items():
        if axis == "overall":
            # A roll-up of the other axes, worded as in eval/judge_openai.py.
            assert "総合" in anchors
            continue
        assert "1=" in anchors, axis
        assert "5=" in anchors, axis


def test_full_duplex_user_prompt_carries_only_the_row() -> None:
    # Rubric, labels and schema belong to the cacheable system-prompt prefix;
    # the per-row message must not repeat them.
    for fragment in ("評価ルーブリック", "フラグの定義", "返すJSON形式", "C_UNCERTAIN_HANDLING"):
        assert fragment in FULL_DUPLEX_SYSTEM_PROMPT

    rendered = full_duplex_prompt({"case_id": "case", "task": "user_interruption"})
    for fragment in ("1-5", "true/false", "C_UNCERTAIN_HANDLING"):
        assert fragment not in rendered
    # The branch the model needs now travels with the data instead.
    assert json.loads(rendered.split("\n", 1)[1])["overlap_action_required"] is True
    assert json.loads(full_duplex_prompt({"case_id": "case", "task": "open_ended"}).split("\n", 1)[1])[
        "overlap_action_required"
    ] is False


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


def test_strict_overlap_writes_partial_summary_from_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "inference"
    out_dir = tmp_path / "benchmark_results"
    successful_trial = run_dir / "background_speech" / "ok" / "seed_0"
    failed_trial = run_dir / "background_speech" / "miss" / "seed_0"
    for trial in (successful_trial, failed_trial):
        trial.mkdir(parents=True)
        (trial / "output.meta.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_id": "test-model",
                "system": "moshi",
                "protocol": {"name": "Full-Duplex-Bench v1.5 static overlap protocol"},
            }
        ),
        encoding="utf-8",
    )

    def fake_evaluate_case(trial: Path, _backchannel_gt=None):
        achieved = trial.name == "seed_0" and trial.parent.name == "ok"
        return {
            "model_id": "test-model",
            "task": "background_speech",
            "case_id": trial.parent.name,
            "seed": 0,
            "metrics": {"overlap_achieved": achieved, "TOR": 1 if achieved else 0},
        }

    monkeypatch.setattr(
        full_duplex_eval,
        "parse_args",
        lambda: Namespace(
            run_dir=run_dir,
            out_dir=out_dir,
            backchannel_gt=None,
            require_overlap=True,
        ),
    )
    monkeypatch.setattr(full_duplex_eval, "evaluate_case", fake_evaluate_case)

    assert full_duplex_eval.main() == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 1
    assert summary["n_total"] == 2
    assert summary["tasks"]["background_speech"]["means"]["TOR"] == 1.0
    assert summary["evaluation"]["successful_trials"] == 1
    assert summary["evaluation"]["failed_trials"] == 1
    assert summary["evaluation"]["failures_by_reason"] == {"overlap_not_achieved": 1}
    assert summary["evaluation"]["failures"][0]["trial_id"] == "background_speech/miss/seed_0"
    per_case = [
        json.loads(line)
        for line in (out_dir / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in per_case] == ["ok"]


def test_batch_combiner_compares_successes_and_counts_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_dir = tmp_path / "batch"
    out = batch_dir / "combined_summary.json"
    summaries = {
        "ok": {"tasks": {"x": {"means": {"TOR": 1.0}, "rates": {}}}},
        "partial": {"tasks": {"x": {"means": {"TOR": 0.5}, "rates": {}}}},
    }
    for name, summary in summaries.items():
        path = batch_dir / name / "benchmark_results" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    status_file = batch_dir / "batch_status.jsonl"
    status_file.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"model_id": "ok", "output_name": "ok", "status": "ok", "elapsed_sec": "1"},
                {
                    "model_id": "partial",
                    "output_name": "partial",
                    "status": "FAILED",
                    "elapsed_sec": "2",
                },
                {
                    "model_id": "missing",
                    "output_name": "missing",
                    "status": "FAILED",
                    "elapsed_sec": "3",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        combine_summaries,
        "parse_args",
        lambda: Namespace(batch_dir=batch_dir, status_file=status_file, out=out),
    )

    assert combine_summaries.main() == 0
    combined = json.loads(out.read_text(encoding="utf-8"))
    assert combined["n_models_requested"] == 3
    assert combined["n_models"] == 1
    assert combined["n_models_failed"] == 2
    assert list(combined["models"]) == ["ok"]
    assert list(combined["partial_models"]) == ["partial"]
    assert combined["comparison"]["x"]["means"]["TOR"] == {"ok": 1.0}
    assert combined["failures"]["missing_summary_output_names"] == ["missing"]


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


def test_cases_per_task_selects_deterministic_smoke_subset() -> None:
    samples = [
        {"id": f"{task}_{index}", "task": task}
        for index in range(3)
        for task in ("a", "b")
    ]
    selected = select_samples(samples, None, 1)
    assert [item["id"] for item in selected] == ["a_0", "b_0"]
    assert [item["id"] for item in select_samples(samples, {"b"}, 2)] == ["b_0", "b_1"]
    with pytest.raises(ValueError, match=">= 1"):
        select_samples(samples, None, 0)


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
