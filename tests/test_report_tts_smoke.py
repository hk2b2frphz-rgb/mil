from __future__ import annotations

import unittest

import numpy as np

from scripts.report_tts_smoke import FPS, turn_taking_stats, voice_activity


def _activity(left: list[int], right: list[int]) -> np.ndarray:
    return np.array([left, right], dtype=bool)


class TurnTakingStatsTest(unittest.TestCase):
    def test_strict_alternation_counts_one_switch(self) -> None:
        stats = turn_taking_stats(_activity([1] * 10 + [0] * 10, [0] * 10 + [1] * 10))
        self.assertEqual(stats["overlap"], 0.0)
        self.assertEqual(stats["silence"], 0.0)
        # 20 frames at 25fps = 0.8s, so one switch is 75 per minute.
        self.assertAlmostEqual(stats["switches_per_min"], 75.0)

    def test_overlap_is_the_fraction_of_both_speaking(self) -> None:
        stats = turn_taking_stats(
            _activity([1] * 10 + [0] * 5, [0] * 5 + [1] * 10)
        )
        self.assertAlmostEqual(stats["overlap"], 5 / 15)
        self.assertEqual(stats["silence"], 0.0)

    def test_silence_is_the_fraction_of_neither_speaking(self) -> None:
        stats = turn_taking_stats(_activity([1] * 5 + [0] * 15, [0] * 20))
        self.assertAlmostEqual(stats["silence"], 0.75)
        self.assertEqual(stats["overlap"], 0.0)
        self.assertEqual(stats["switches_per_min"], 0.0)

    def test_overlap_does_not_break_a_turn_in_two(self) -> None:
        # A talks, both talk, A talks again: the floor never changed hands.
        stats = turn_taking_stats(
            _activity([1] * 15, [0] * 5 + [1] * 5 + [0] * 5)
        )
        self.assertEqual(stats["switches_per_min"], 0.0)

    def test_a_backchannel_inside_a_turn_counts_no_switch(self) -> None:
        # The listener's aizuchi overlaps the speaker entirely, so nobody ever
        # holds the floor alone on the listener's side.
        stats = turn_taking_stats(
            _activity([1] * 20, [0] * 8 + [1] * 3 + [0] * 9)
        )
        self.assertEqual(stats["switches_per_min"], 0.0)
        self.assertAlmostEqual(stats["overlap"], 3 / 20)

    def test_empty_input_is_all_silence(self) -> None:
        stats = turn_taking_stats(np.zeros((2, 0), dtype=bool))
        self.assertEqual(stats["silence"], 1.0)
        self.assertEqual(stats["overlap"], 0.0)


class VoiceActivityTest(unittest.TestCase):
    def test_detects_the_loud_half_of_each_channel(self) -> None:
        sample_rate = 1000
        seconds = 2
        n = sample_rate * seconds
        t = np.arange(n) / sample_rate
        tone = 0.5 * np.sin(2 * np.pi * 220 * t)
        left = tone.copy()
        left[n // 2 :] = 0.0   # left speaks in the first half
        right = tone.copy()
        right[: n // 2] = 0.0  # right speaks in the second half

        activity = voice_activity(np.stack([left, right]), sample_rate)
        frames = activity.shape[-1]
        self.assertGreater(frames, 0)
        quarter = frames // 4
        self.assertTrue(activity[0, :quarter].all())
        self.assertFalse(activity[0, -quarter:].any())
        self.assertFalse(activity[1, :quarter].any())
        self.assertTrue(activity[1, -quarter:].all())

    def test_frame_rate_matches_the_paper_definition(self) -> None:
        sample_rate = 2000
        wav = np.ones((2, sample_rate * 3))
        activity = voice_activity(wav, sample_rate)
        # 3 seconds at 25 fps, minus the tail that cannot fit a full window.
        self.assertEqual(activity.shape[-1], 3 * FPS)

    def test_a_silent_channel_is_never_active(self) -> None:
        sample_rate = 1000
        wav = np.zeros((2, sample_rate))
        activity = voice_activity(wav, sample_rate)
        self.assertFalse(activity.any())


if __name__ == "__main__":
    unittest.main()
