from __future__ import annotations

import unittest
from dataclasses import asdict
from random import Random

import numpy as np

from scripts.build_full_duplex_training_use_cases import TASKS
from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueTurn,
    build_segments,
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


if __name__ == "__main__":
    unittest.main()
