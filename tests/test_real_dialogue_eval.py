"""実対話リプレイ評価パイプラインのテスト。

合成のタイムラインと WAV でデータセット構築から相槌一致度・judge 入力までを
通す。実データが無い環境でも回るようにしてある(実データはこのリポジトリに
置かない方針のため、CI で踏めるのはこの経路だけになる)。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
for path in (ROOT, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.build_real_dialogue_dataset import (  # noqa: E402
    Turn,
    detect_cases,
    merge_turns,
    resolve_counselor,
)
from eval.evaluate_real_backchannel import match_pairs, prf  # noqa: E402


def _args(**overrides):
    from argparse import Namespace

    base = dict(
        pause_min_sec=0.8,
        backchannel_min_turn_sec=6.0,
        turn_merge_gap_sec=0.4,
        context_sec=30.0,
        tail_sec=0.0,
    )
    base.update(overrides)
    return Namespace(**base)


def _turn(speaker, start, end, text="テキスト", aizuchi=False, labels=(), index=0):
    return Turn(
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        is_aizuchi=aizuchi,
        aizuchi_labels=list(labels),
        index=index,
    )


def _write_analysis_dir(root: Path, counselor: str = "B") -> Path:
    """分離ステージが出したのと同じ形のディレクトリを合成で作る。"""
    analysis = root / "dialogue01"
    (analysis / "separated").mkdir(parents=True, exist_ok=True)

    # A(ユーザー)が 0-10 秒を話し、その間に B(相談員)が 2 回相槌を打つ。
    turns = [
        {"speaker": "A", "start": 0.0, "end": 10.0, "text": "最近ずっと眠れなくて",
         "overlap_sec": 0.6, "is_aizuchi": False, "aizuchi_labels": []},
        {"speaker": "B", "start": 3.0, "end": 3.4, "text": "ええ",
         "overlap_sec": 0.4, "is_aizuchi": True, "aizuchi_labels": ["ee"]},
        {"speaker": "B", "start": 7.0, "end": 7.3, "text": "そうなんですね",
         "overlap_sec": 0.3, "is_aizuchi": True, "aizuchi_labels": ["sou"]},
        {"speaker": "B", "start": 10.5, "end": 13.0, "text": "それはおつらいですね",
         "overlap_sec": 0.0, "is_aizuchi": False, "aizuchi_labels": []},
    ]
    with (analysis / "timeline_separated.jsonl").open("w", encoding="utf-8") as handle:
        for row in turns:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sr = 16000
    for speaker in ("A", "B"):
        sf.write(
            analysis / "separated" / f"{speaker}.wav",
            np.zeros(sr * 15, dtype=np.float32),
            sr,
        )
    (analysis / f"counselor_{counselor}.txt").write_text("", encoding="utf-8")
    return analysis


def test_resolve_counselor_requires_exactly_one_marker(tmp_path: Path) -> None:
    analysis = _write_analysis_dir(tmp_path)
    assert resolve_counselor(analysis) == "B"

    # 0 個: どちらが相談員か決まらないので落ちること。
    (analysis / "counselor_B.txt").unlink()
    with pytest.raises(SystemExit):
        resolve_counselor(analysis)

    # 2 個: 矛盾しているので落ちること。取り違えは評価を静かに無意味にする。
    (analysis / "counselor_A.txt").write_text("", encoding="utf-8")
    (analysis / "counselor_B.txt").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        resolve_counselor(analysis)


def test_merge_turns_joins_fragments_but_keeps_aizuchi() -> None:
    turns = [
        _turn("A", 0.0, 1.0, "最近", index=0),
        _turn("A", 1.2, 2.0, "眠れなくて", index=1),   # gap 0.2 < 0.4 -> 結合
        _turn("B", 1.5, 1.8, "ええ", aizuchi=True, labels=["ee"], index=2),
        _turn("A", 3.0, 4.0, "それで", index=3),        # gap 1.0 -> 結合しない
    ]
    merged = merge_turns(turns, gap=0.4)
    speakers_a = [t for t in merged if t.speaker == "A"]
    assert len(speakers_a) == 2
    assert speakers_a[0].start == 0.0 and speakers_a[0].end == 2.0
    assert speakers_a[0].text == "最近眠れなくて"
    # 相槌は結合されず単独で残る。
    assert any(t.is_aizuchi and t.speaker == "B" for t in merged)


def test_detect_cases_backchannel_collects_counselor_aizuchi() -> None:
    turns = [
        _turn("A", 0.0, 10.0, "長い相談の発話", index=0),
        _turn("B", 3.0, 3.4, "ええ", aizuchi=True, labels=["ee"], index=1),
        _turn("B", 7.0, 7.3, "そう", aizuchi=True, labels=["sou"], index=2),
    ]
    cases = detect_cases(turns, user="A", counselor="B", keep=None, args=_args())
    backchannel = [c for c in cases if c["task"] == "backchannel"]
    assert len(backchannel) == 1
    assert len(backchannel[0]["backchannel_gt"]) == 2


def test_detect_cases_respects_review_keep_filter() -> None:
    turns = [
        _turn("A", 0.0, 10.0, "長い相談の発話", index=0),
        _turn("B", 3.0, 3.4, "ええ", aizuchi=True, labels=["ee"], index=1),
    ]
    # index 0 を落とすと、そのユーザー発話由来のケースは一切作られないこと。
    cases = detect_cases(turns, user="A", counselor="B", keep=set(), args=_args())
    assert cases == []


def test_detect_cases_pause_requires_silent_counselor() -> None:
    # ユーザーの無音の間に相談員が喋っていれば、それは「間」ではなくターン交替。
    turns = [
        _turn("A", 0.0, 2.0, "前半", index=0),
        _turn("B", 2.2, 3.0, "相談員の発話", index=1),
        _turn("A", 3.5, 5.0, "後半", index=2),
    ]
    cases = detect_cases(turns, user="A", counselor="B", keep=None, args=_args())
    assert not [c for c in cases if c["task"] == "pause_handling"]

    turns_quiet = [
        _turn("A", 0.0, 2.0, "前半", index=0),
        _turn("A", 3.5, 5.0, "後半", index=1),
    ]
    cases_quiet = detect_cases(
        turns_quiet, user="A", counselor="B", keep=None, args=_args()
    )
    pause = [c for c in cases_quiet if c["task"] == "pause_handling"]
    assert len(pause) == 1
    assert pause[0]["events"]["pause"] == [[2.0, 3.5]]


def test_detect_cases_separates_backchannel_from_interruption() -> None:
    turns = [
        _turn("B", 0.0, 5.0, "相談員が話している", index=0),
        _turn("A", 1.0, 1.3, "うん", aizuchi=True, labels=["un"], index=1),
        _turn("A", 3.0, 4.0, "でもそれは違って", index=2),
    ]
    cases = detect_cases(turns, user="A", counselor="B", keep=None, args=_args())
    tasks = {c["task"] for c in cases}
    assert "user_backchannel" in tasks
    assert "user_interruption" in tasks
    interruption = next(c for c in cases if c["task"] == "user_interruption")
    assert interruption["events"]["interruption"] == [[3.0, 4.0]]


def test_match_pairs_is_one_to_one() -> None:
    gt = [{"start_sec": 3.0, "labels": ["ee"]}]
    pred = [
        {"start_sec": 3.1, "labels": ["ee"]},
        {"start_sec": 3.2, "labels": ["ee"]},
    ]
    # 1 つの正解に 2 つの予測を当てて再現率を水増しできないこと。
    pairs = match_pairs(gt, pred, tolerance=1.5, require_type=False)
    assert len(pairs) == 1
    assert pairs[0] == (0, 0)  # より近い方が選ばれる


def test_match_pairs_type_requirement() -> None:
    gt = [{"start_sec": 3.0, "labels": ["naruhodo"]}]
    pred = [{"start_sec": 3.1, "labels": ["un"]}]
    assert len(match_pairs(gt, pred, 1.5, require_type=False)) == 1
    # 種類が違えば typed 一致にはならない。
    assert match_pairs(gt, pred, 1.5, require_type=True) == []


def test_match_pairs_respects_tolerance() -> None:
    gt = [{"start_sec": 3.0, "labels": ["ee"]}]
    pred = [{"start_sec": 6.0, "labels": ["ee"]}]
    assert match_pairs(gt, pred, 1.5, require_type=False) == []


def test_prf_handles_empty_predictions() -> None:
    assert prf(0, 0, 3) == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 3,
    }


def test_build_dataset_end_to_end(tmp_path: Path) -> None:
    analysis = _write_analysis_dir(tmp_path)
    out_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "build_real_dialogue_dataset.py"),
            "--analysis-dir", str(analysis),
            "--out-dir", str(out_dir),
            "--context-sec", "5",
            "--no-review-filter",
        ],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] > 0
    # 合成 FDB-JA と取り違えられないよう、v1.5 を名乗っていないこと。
    assert "v1.5" not in manifest["protocol"]["name"]
    assert manifest["protocol"]["open_loop"] is True

    sample = manifest["samples"][0]
    metadata = json.loads(
        (out_dir / sample["path"] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source"]["counselor_speaker"] == "B"
    assert metadata["source"]["user_speaker"] == "A"
    assert (out_dir / sample["path"] / "input.wav").is_file()

    backchannel_cases = [s for s in manifest["samples"] if s["task"] == "backchannel"]
    assert backchannel_cases, "10 秒のユーザー発話から backchannel ケースが出るはず"
    bc_metadata = json.loads(
        (out_dir / backchannel_cases[0]["path"] / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    # 相談員が実際に打った 2 回の相槌が正解として入っていること。
    assert len(bc_metadata["backchannel_gt"]) == 2
    labels = {tuple(item["labels"]) for item in bc_metadata["backchannel_gt"]}
    assert ("ee",) in labels
    # 時刻は窓の先頭からの相対秒であること(絶対時刻のままだと評価器がずれる)。
    window_start = bc_metadata["source"]["window_start_sec"]
    assert bc_metadata["backchannel_gt"][0]["start_sec"] == pytest.approx(
        3.0 - window_start, abs=1e-3
    )
    # 人間参照が入っていること(judge の上限として使う)。
    assert bc_metadata["human_reference"]["segments"]
