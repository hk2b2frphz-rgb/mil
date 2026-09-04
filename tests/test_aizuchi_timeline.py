from __future__ import annotations

import random
import unittest

import numpy as np

from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueTurn,
    build_segments_whole_utterance,
)
from scripts.generate_synthetic_moshi_training_data import pick_reaction_points


class _FakeTTS:
    sample_rate = 100

    def synthesize(
        self,
        text: str,
        speaker: str,
        *,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        del text, instruct, speaker_override
        duration = 6.0 if speaker == "user" else 0.8
        return np.ones(round(duration * self.sample_rate), dtype=np.float32)


class _FakeAligner:
    def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        texts: list[str],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        del audio, sample_rate
        cursor = 0.0
        spans: list[tuple[float, float]] = []
        for text in texts:
            duration = 0.4 if text in {"はい。", "ええ。"} else 2.0
            spans.append((cursor, cursor + duration))
            cursor += duration
        return list(spans), list(spans)


class AizuchiTimelineTest(unittest.TestCase):
    def test_multiple_aizuchi_stay_anchored_to_the_merged_user(self) -> None:
        dialogue = Dialogue(
            id="multiple-aizuchi",
            category="test",
            risk_level="low",
            title="test",
            turns=[
                DialogueTurn("user", "第一句。"),
                DialogueTurn("moshi", "はい。", timing="overlap_previous"),
                DialogueTurn("user", "第二句。"),
                DialogueTurn("moshi", "ええ。", timing="overlap_previous"),
                DialogueTurn("user", "第三句。"),
            ],
        )

        segments, _ = build_segments_whole_utterance(
            dialogue,
            _FakeTTS(),
            _FakeAligner(),
            lead_in_sec=0.0,
            gap_sec=0.0,
        )

        user_segment = next(segment for segment in segments if segment.speaker == "user")
        aizuchi = [segment for segment in segments if segment.speaker == "moshi"]
        self.assertEqual([segment.text for segment in aizuchi], ["はい。", "ええ。"])
        self.assertGreater(aizuchi[1].start_sec - aizuchi[0].start_sec, 1.5)
        self.assertLess(aizuchi[1].start_sec, user_segment.end_sec)

    def test_frequency_limits_are_applied(self) -> None:
        frequency = {
            "rates": {"end": 1.0, "cont": 1.0, "weak": 1.0},
            "max_per_turn": 2,
            "min_chars": 0,
            "min_gap": 1,
            "min_chunk_chars": 0,
        }
        points = pick_reaction_points(
            ["一つ目、", "二つ目、", "三つ目、", "四つ目。"],
            frequency,
            random.Random(0),
        )
        self.assertEqual([point["after_clause"] for point in points], [1, 3])

    def test_short_turn_is_silent_when_below_min_chars(self) -> None:
        frequency = {
            "rates": {"end": 1.0, "cont": 1.0, "weak": 1.0},
            "max_per_turn": 2,
            "min_chars": 20,
            "min_gap": 0,
            "min_chunk_chars": 0,
        }
        self.assertEqual(
            pick_reaction_points(["短いです。"], frequency, random.Random(0)),
            [],
        )


if __name__ == "__main__":
    unittest.main()
