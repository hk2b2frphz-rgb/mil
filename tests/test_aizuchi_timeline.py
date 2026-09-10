from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueTurn,
    build_segments_whole_utterance,
)
from scripts.generate_synthetic_moshi_training_data import (
    AIZUCHI_FREQUENCY_PRESETS,
    AIZUCHI_MIXED_FREQUENCY_POOL,
    AIZUCHI_ONLY_GREETING,
    DialogueTurn as SyntheticDialogueTurn,
    build_aizuchi_only_user_prompt,
    complete_aizuchi_reactions,
    plan_aizuchi_only_blocks,
    pick_reaction_points,
    sanitize_aizuchi_only_turns,
    split_user_text_on_pauses,
)


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

    def test_normal_guarantees_one_reaction_when_probability_misses(self) -> None:
        frequency = dict(AIZUCHI_FREQUENCY_PRESETS["normal"])
        frequency["rates"] = {"end": 0.0, "cont": 0.0, "weak": 0.0}

        points = pick_reaction_points(
            ["今日は少しつらかったです。"], frequency, random.Random(0)
        )

        self.assertEqual(points, [{"after_clause": 1, "kind": "end"}])

    def test_normal_is_configured_for_dense_reactions(self) -> None:
        frequency = AIZUCHI_FREQUENCY_PRESETS["normal"]

        self.assertEqual(frequency["rates"]["end"], 1.0)
        self.assertGreaterEqual(frequency["rates"]["cont"], 0.75)
        self.assertEqual(frequency["max_per_turn"], 3)
        self.assertEqual(frequency["min_per_turn"], 1)

    def test_flood_is_denser_than_normal(self) -> None:
        flood = AIZUCHI_FREQUENCY_PRESETS["flood"]
        normal = AIZUCHI_FREQUENCY_PRESETS["normal"]

        for kind in ("end", "cont", "weak"):
            self.assertGreaterEqual(flood["rates"][kind], normal["rates"][kind])
        self.assertGreater(flood["max_per_turn"], normal["max_per_turn"])
        self.assertLess(flood["min_gap"], normal["min_gap"])
        self.assertEqual(flood["min_per_turn"], 1)

    def test_flood_reacts_at_every_clause_boundary(self) -> None:
        clauses = ["一つ目、", "二つ目、", "三つ目、", "四つ目。"]

        points = pick_reaction_points(
            clauses, AIZUCHI_FREQUENCY_PRESETS["flood"], random.Random(0)
        )

        self.assertEqual([point["after_clause"] for point in points], [1, 2, 3, 4])

    def test_flood_is_excluded_from_the_mixed_pool(self) -> None:
        self.assertNotIn("flood", AIZUCHI_MIXED_FREQUENCY_POOL)
        self.assertEqual(
            set(AIZUCHI_MIXED_FREQUENCY_POOL),
            set(AIZUCHI_FREQUENCY_PRESETS) - {"flood"},
        )

    def test_zero_silence_budget_disables_planned_and_inline_pauses(self) -> None:
        args = SimpleNamespace(
            aizuchi_only_probe_rate=1.0,
            aizuchi_only_silence_rate=1.0,
            aizuchi_only_silence_min_sec=2.0,
            aizuchi_only_silence_max_sec=4.0,
            aizuchi_only_max_silences=0,
        )

        plans = plan_aizuchi_only_blocks(random.Random(0), 6, args)
        prompt = build_aizuchi_only_user_prompt(
            {},
            {
                "opening_style": "率直",
                "disclosure_pace": "普通",
                "speech_texture": "短文",
                "emotional_arc": "一定",
                "topic_order": "時系列",
            },
            [],
            0,
            6,
            plans[0],
        )
        split = split_user_text_on_pauses(
            "言葉が出ません。<<pause:3.5>>でも続けます。",
            random.Random(0),
            max_pauses=0,
        )

        self.assertTrue(all(plan["trailing_silence"] is None for plan in plans))
        self.assertTrue(all(not plan["allow_pause"] for plan in plans))
        self.assertNotIn("<<pause", prompt.user)
        self.assertEqual([turn.speaker for turn in split], ["user"])

    def test_missing_llm_reaction_is_filled_with_acknowledgement(self) -> None:
        completed = complete_aizuchi_reactions(
            [], [{"after_clause": 1, "kind": "end"}], []
        )

        self.assertEqual(
            completed,
            [{"after_clause": 1, "text": "そうなんですね。"}],
        )

    def test_closing_phrases_are_removed_from_aizuchi_training_data(self) -> None:
        turns = [
            SyntheticDialogueTurn("moshi", AIZUCHI_ONLY_GREETING),
            SyntheticDialogueTurn("user", "今日はもう休みます。"),
            SyntheticDialogueTurn("moshi", "はい。おやすみなさい。"),
            SyntheticDialogueTurn("user", "聞いてくれてありがとう。"),
            SyntheticDialogueTurn("moshi", "はい。失礼いたします。"),
        ]

        clean_turns = sanitize_aizuchi_only_turns(turns)

        self.assertEqual(
            [(turn.speaker, turn.text) for turn in clean_turns],
            [
                ("moshi", AIZUCHI_ONLY_GREETING),
                ("user", "今日はもう休みます。"),
                ("user", "聞いてくれてありがとう。"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
