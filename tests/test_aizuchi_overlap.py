from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_qwen3_tts_data import apply_aizuchi_overlap


def test_aizuchi_turn_overlaps_previous_user() -> None:
    turns = [
        {"speaker": "user", "text": "今日は少し疲れました。"},
        {
            "speaker": "moshi",
            "text": "はい。",
            "truncate_previous_after_sec": 0.2,
        },
    ]

    updated = apply_aizuchi_overlap(turns)

    assert updated[1]["timing"] == "overlap_previous"
    assert updated[1]["start_after_previous_start_sec"] == 999
    assert "truncate_previous_after_sec" not in updated[1]


def test_aizuchi_without_previous_user_is_unchanged() -> None:
    turns = [
        {"speaker": "moshi", "text": "ええ。"},
        {"speaker": "moshi", "text": "なるほど。"},
    ]

    updated = apply_aizuchi_overlap(turns)

    assert updated == turns


def test_regular_moshi_response_is_unchanged() -> None:
    turns = [
        {"speaker": "user", "text": "相談してもいいですか。"},
        {"speaker": "moshi", "text": "もちろんです。ゆっくり話してください。"},
    ]

    updated = apply_aizuchi_overlap(turns)

    assert updated == turns


def test_apply_aizuchi_overlap_returns_copy_without_mutating_input() -> None:
    turns = [
        {"speaker": "user", "text": "ちょっと聞いてください。"},
        {"speaker": "moshi", "text": "そうなんですね。"},
    ]
    original_second = dict(turns[1])

    updated = apply_aizuchi_overlap(turns)

    assert updated is not turns
    assert updated[0] is not turns[0]
    assert updated[1] is not turns[1]
    assert turns[1] == original_second
    assert "timing" not in turns[1]
    assert updated[1]["timing"] == "overlap_previous"
