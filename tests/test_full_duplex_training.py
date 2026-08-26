from __future__ import annotations

import unittest
from dataclasses import asdict
from random import Random

import numpy as np

from scripts.build_full_duplex_training_use_cases import TASKS
from scripts.generate_qwen3_tts_data import (
    CHANNEL_GAIN_MAX,
    AudioSegment,
    Dialogue,
    DialogueTurn,
    build_segments,
    channel_rms,
    render_stereo,
    validate_duplex_dialogue,
)
from scripts.generate_synthetic_moshi_training_data import (
    build_llm_prompt,
    template_full_duplex_dialogue,
)


class FakeTTS:
    sample_rate = 100

    def __init__(self, durations: dict[str, float]):
        self.durations = durations
        self.calls: list[dict[str, str | None]] = []

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "text": text,
                "speaker_role": speaker_role,
                "speaker_override": speaker_override,
            }
        )
        return np.ones(
            int(round(self.durations[text] * self.sample_rate)),
            dtype=np.float32,
        )


class FullDuplexTimingTests(unittest.TestCase):
    def test_pause_prompt_requires_user_continuation_after_silence(self) -> None:
        prompt = build_llm_prompt(
            {
                "id": "pause",
                "duplex_task": "pause_handling",
                "silence_pattern": "occasional",
            },
            Random(0),
        )
        self.assertIn("その直後はuserが同じ発話を続ける", prompt)
        self.assertNotIn("沈黙の直後は moshi が穏やかに声をかける", prompt)

    def test_all_fallback_templates_pass_duplex_validation(self) -> None:
        for task in TASKS:
            with self.subTest(task=task):
                dialogue = template_full_duplex_dialogue(
                    {
                        "id": f"test_{task}",
                        "duplex_task": task,
                        "opening": "少し話してもいいですか。",
                    }
                )
                self.assertEqual(validate_duplex_dialogue(asdict(dialogue)), [])

    def test_validator_rejects_untruncated_interruption(self) -> None:
        errors = validate_duplex_dialogue(
            {
                "id": "bad",
                "duplex_task": "user_interruption",
                "turns": [
                    {"speaker": "moshi", "text": "説明"},
                    {
                        "speaker": "user",
                        "text": "訂正",
                        "timing": "overlap_previous",
                        "event": "user_interruption",
                    },
                ],
            }
        )
        self.assertIn(
            "user_interruption requires truncate_previous_after_sec",
            errors,
        )

    def test_user_interruption_overlaps_and_truncates_model(self) -> None:
        tts = FakeTTS({"model": 4.0, "interrupt": 1.0, "reply": 1.0})
        dialogue = Dialogue(
            id="interrupt",
            category="test",
            risk_level="low",
            title="test",
            duplex_task="user_interruption",
            turns=[
                DialogueTurn("moshi", "model"),
                DialogueTurn(
                    "user",
                    "interrupt",
                    timing="overlap_previous",
                    start_after_previous_start_sec=1.0,
                    truncate_previous_after_sec=0.2,
                    event="user_interruption",
                ),
                DialogueTurn("moshi", "reply"),
            ],
        )

        segments, _ = build_segments(dialogue, tts, 0.3, 0.4)

        self.assertAlmostEqual(segments[1].start_sec, 1.3)
        self.assertAlmostEqual(segments[0].end_sec, 1.5)
        self.assertNotEqual(segments[0].text, "model")
        self.assertTrue(segments[0].text.endswith("…"))
        self.assertLess(segments[1].start_sec, segments[0].end_sec)
        self.assertGreaterEqual(segments[2].start_sec, segments[1].end_sec + 0.4)

    def test_background_uses_distinct_voice_and_gain(self) -> None:
        tts = FakeTTS({"model": 3.0, "announcement": 1.0})
        dialogue = Dialogue(
            id="background",
            category="test",
            risk_level="low",
            title="test",
            duplex_task="background_speech",
            turns=[
                DialogueTurn("moshi", "model"),
                DialogueTurn(
                    "user",
                    "announcement",
                    timing="overlap_previous",
                    start_after_previous_start_sec=0.8,
                    gain=0.3,
                    voice_role="background",
                    event="background_speech",
                ),
            ],
        )

        segments, _ = build_segments(
            dialogue,
            tts,
            0.0,
            0.4,
            user_speaker_override="user_voice",
            other_speaker_override="other_voice",
            background_speaker_override="background_voice",
        )

        self.assertAlmostEqual(float(segments[1].pcm.max()), 0.3)
        self.assertEqual(tts.calls[1]["speaker_override"], "background_voice")
        self.assertLess(segments[1].start_sec, segments[0].end_sec)
        stereo = render_stereo(segments, tts.sample_rate, tail_sec=0.0)
        overlap_sample = int(round(0.9 * tts.sample_rate))
        self.assertGreater(float(stereo[0, overlap_sample]), 0.0)
        self.assertGreater(float(stereo[1, overlap_sample]), 0.0)


class ChannelLevelMatchingTest(unittest.TestCase):
    SAMPLE_RATE = 100

    def _segment(self, speaker: str, start_sec: float, amplitude: float) -> AudioSegment:
        pcm = np.full(self.SAMPLE_RATE, amplitude, dtype=np.float32)
        return AudioSegment(
            speaker=speaker,
            label=speaker,
            text="t",
            start_sec=start_sec,
            end_sec=start_sec + 1.0,
            pcm=pcm,
        )

    def test_quiet_moshi_is_raised_to_the_user_channel(self) -> None:
        stereo = render_stereo(
            [
                self._segment("moshi", 0.0, 0.05),
                self._segment("user", 1.0, 0.20),
            ],
            self.SAMPLE_RATE,
            tail_sec=0.0,
        )

        self.assertAlmostEqual(
            channel_rms(stereo[0]), channel_rms(stereo[1]), places=5
        )
        # The user channel is the reference and stays where it was.
        self.assertAlmostEqual(channel_rms(stereo[1]), 0.20, places=5)

    def test_gain_is_clamped_so_a_broken_channel_is_not_amplified(self) -> None:
        quiet = 1e-3
        stereo = render_stereo(
            [
                self._segment("moshi", 0.0, quiet),
                self._segment("user", 1.0, 0.50),
            ],
            self.SAMPLE_RATE,
            tail_sec=0.0,
        )

        self.assertAlmostEqual(
            channel_rms(stereo[0]), quiet * CHANNEL_GAIN_MAX, places=5
        )
        self.assertLess(channel_rms(stereo[0]), channel_rms(stereo[1]))

    def test_a_silent_moshi_channel_is_left_alone(self) -> None:
        stereo = render_stereo(
            [self._segment("user", 0.0, 0.20)],
            self.SAMPLE_RATE,
            tail_sec=0.0,
        )

        self.assertEqual(channel_rms(stereo[0]), 0.0)
        self.assertAlmostEqual(channel_rms(stereo[1]), 0.20, places=5)


if __name__ == "__main__":
    unittest.main()
