from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts import build_use_cases as cases
from scripts import generate_synthetic_moshi_training_data as gen
from scripts import use_case_interactions as axes


class InteractionAxesTest(unittest.TestCase):
    def case(self, **overrides):
        card = asdict(cases.build_one(random.Random(0), 1))
        card["interaction_axes"].update(overrides)
        return card

    def test_sampling_covers_the_requested_axes_without_mixing_end_requests(self):
        rng = random.Random(0)
        rows = [axes.sample_interaction_axes(rng, "distress") for _ in range(10000)]
        self.assertEqual({r["purpose"] for r in rows}, {"be_heard", "organize", "find_support", "unsure"})
        self.assertEqual({r["speaking_style"] for r in rows}, set(axes.SPEAKING_STYLES))
        self.assertEqual({r["disclosure_style"] for r in rows}, set(axes.DISCLOSURE_STYLES))
        self.assertEqual({b for r in rows for b in r["support_barriers"]}, set(axes.BARRIERS))
        self.assertEqual({r["reaction"] for r in rows}, set(axes.REACTIONS) - {"end"})
        self.assertTrue(any(not r["support_barriers"] for r in rows))
        for row in rows:
            self.assertEqual(len(row["support_barriers"]), len(set(row["support_barriers"])))
            self.assertEqual(row["track"], "ongoing")
        self.assertEqual(axes.sample_interaction_axes(random.Random(2), "distress"),
                         axes.sample_interaction_axes(random.Random(2), "distress"))

    def test_same_background_can_have_different_interaction_axes(self):
        with patch.object(cases, "SITUATIONS", [cases.SITUATIONS[0]]):
            cards = [cases.build_one(random.Random(seed), 1) for seed in range(50)]
        self.assertEqual(len({c.situation_id for c in cards}), 1)
        self.assertGreater(len({json.dumps(c.interaction_axes, sort_keys=True) for c in cards}), 40)

    def test_positive_and_curiosity_motivations_are_preserved(self):
        for seed in range(50):
            positive = axes.sample_interaction_axes(random.Random(seed), "positive")
            self.assertNotIn(positive["purpose"], {"find_support", "unsure"})
            self.assertNotEqual(positive["speaking_style"], "anger")
            self.assertFalse(positive["support_barriers"])
        with patch.object(cases, "SITUATIONS", [next(r for r in cases.SITUATIONS if r[0] == "curious_about_line")]):
            cards = [cases.build_one(random.Random(seed), 1) for seed in range(50)]
        self.assertIn("explore", {c.interaction_axes["purpose"] for c in cards})

    def test_brief_and_upfront_override_conflicting_random_profile_and_length(self):
        case = self.case(speaking_style="brief", disclosure_style="upfront")
        profile = gen.build_user_agent_profile(case, random.Random(5), listener_only=True)
        self.assertIn("最初から", profile["opening_style"])
        self.assertIn("短い一言", profile["speech_texture"])
        prompt = gen.build_aizuchi_only_user_prompt(case, profile, [], 0, 12,
            dict(probe=False, allow_pause=False, utterance_length="long"))
        self.assertIn("短い一言だけ", prompt.user)
        self.assertNotIn("今回は4〜6文", prompt.user)

    def test_anger_is_also_reflected_in_audio_emotion_metadata(self):
        card = next(cases.build_one(random.Random(seed), 1) for seed in range(100)
                    if cases.build_one(random.Random(seed), 1).interaction_axes["speaking_style"] == "anger")
        self.assertEqual(card.emotional_state, "気持ちが高ぶる")
        self.assertIn("_agitated_", card.id)

    def test_reactions_depend_on_actual_last_listener_turn(self):
        for reaction in ("answer", "difficult", "decline", "correct"):
            case = self.case(reaction=reaction)
            for history in ([], [gen.DialogueTurn("moshi", "はい。")]):
                guidance = axes.user_interaction_guidance(case, history, listener_only=True)
                self.assertIn("あったことにして反応を作らない", guidance)
                self.assertNotIn("- " + axes.REACTIONS[reaction][1], guidance)
        case = self.case(reaction="difficult")
        self.assertIn(axes.REACTIONS["difficult"][1], axes.user_interaction_guidance(
            case, [gen.DialogueTurn("moshi", "どんな気持ちですか？")]))
        case = self.case(reaction="decline")
        self.assertIn(axes.REACTIONS["decline"][1], axes.user_interaction_guidance(
            case, [gen.DialogueTurn("moshi", "よければ別の方法を考えてみませんか？")]))
        self.assertNotIn("- " + axes.REACTIONS["decline"][1], axes.user_interaction_guidance(
            case, [gen.DialogueTurn("moshi", "はい。")]))
        # A question before a silence is not the current listener turn.
        self.assertNotIn("- " + axes.REACTIONS["difficult"][1], axes.user_interaction_guidance(
            self.case(reaction="difficult"), [gen.DialogueTurn("moshi", "どんな気持ちですか？"),
                                               gen.DialogueTurn("silence", "", duration_sec=3)]))

    def test_withholding_privacy_and_support_purpose_reach_user_prompt(self):
        case = self.case(purpose="find_support", disclosure_style="withhold", support_barriers=["privacy", "cost"])
        text = axes.user_interaction_guidance(case, [], listener_only=True)
        self.assertIn("名前や言いたくない事情は明かさない", text)
        self.assertIn("金額や無料かどうかを事実として作らない", text)
        self.assertIn("聞き手が制度を紹介したことにしない", text)
        self.assertIn("個人情報への不安", gen.case_brief(case))

    def test_first_turn_cannot_correct_an_invented_previous_statement(self):
        case = self.case(disclosure_style="correct")
        self.assertIn("まだ自分の前の発話はない", axes.user_interaction_guidance(case, []))
        self.assertNotIn("まだ自分の前の発話はない", axes.user_interaction_guidance(
            case, [gen.DialogueTurn("user", "昨日のことなんです。")]))

    def test_end_request_entrypoint_keeps_the_track_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "end.jsonl"
            result = subprocess.run([sys.executable, "scripts/build_full_duplex_training_use_cases.py",
                "--out-path", str(output), "--num", "100", "--interaction-track", "end-request"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["interaction_axes"]["reaction"] == "end" for row in rows))
            self.assertTrue(all(row["interaction_axes"]["track"] == "end-request" for row in rows))

    def make_generator(self, directory):
        with patch.object(sys, "argv", ["generator", "--out-dir", str(directory)]):
            args = gen.parse_args()
        args.llm_backend = "openai-compatible"
        args.dialogue_generation_mode = "multi-agent"
        args.multi_agent_min_pairs = args.multi_agent_max_pairs = 5
        args.no_multi_agent_aizuchi = True
        generator = gen.LLMDialogueGenerator(args)
        generator.trace_path = None
        return generator

    def test_end_requests_are_rejected_before_calling_listener_only_llm(self):
        case = self.case(track="end-request", reaction="end")
        with tempfile.TemporaryDirectory() as directory:
            generator = self.make_generator(directory)
            with patch.object(generator, "call_agent") as caller:
                with self.assertRaisesRegex(ValueError, "require multi-agent"):
                    generator.generate_aizuchi_only(case, random.Random(0))
                caller.assert_not_called()

    def test_direct_end_request_stops_even_before_minimum_pair_count(self):
        case = self.case(track="end-request", reaction="end", speaking_style="brief")
        replies = ["気になることがあってかけたんです。", "気になったんですね。",
                   "今日はここまでにしたいです。", "はい、ここまでにしましょう。"]
        with tempfile.TemporaryDirectory() as directory:
            generator = self.make_generator(directory)
            with patch.object(generator, "call_agent", side_effect=replies) as caller, patch.object(gen.logger, "info"):
                dialogue = generator.generate_multi_agent(case, random.Random(0))
            self.assertEqual(caller.call_count, 4)
            self.assertEqual(len(dialogue.turns), 4)
            self.assertEqual(dialogue.interaction_axes["track"], "end-request")
            self.assertIn("引き留めません", caller.call_args_list[-1].args[0].user)

    def test_missing_end_request_is_retried_and_fails_instead_of_labeling_ongoing_talk_as_end(self):
        case = self.case(track="end-request", reaction="end")
        with tempfile.TemporaryDirectory() as directory:
            generator = self.make_generator(directory)
            replies = ["少し聞いてほしくなったんです。", "聞いてほしかったんですね。",
                       "そのことをまだ話しているんです。", "まだ続きを話します。"]
            with patch.object(generator, "call_agent", side_effect=replies) as caller, patch.object(gen.logger, "info"), patch.object(gen.logger, "warning"):
                with self.assertRaises(ValueError):
                    generator.generate_multi_agent(case, random.Random(0))
            self.assertEqual(caller.call_count, 4)

    def test_explicit_end_requests_cannot_leak_into_continuous_listener_data(self):
        issue = gen.ongoing_user_utterance("今日はここまでにしたいです。")
        self.assertIsNotNone(issue)
        self.assertTrue(issue[1])

    def test_end_detector_does_not_confuse_quoted_or_negated_endings(self):
        self.assertTrue(axes.user_requests_end("今日はここまでにしたいです。"))
        self.assertTrue(axes.user_requests_end("もう話したくないです。"))
        for text in ("今日はここまでにしたくないです。", "今日はまだ話したいです。",
                     "昨日『今日はここまでにしたいです』と言われたんです。"):
            self.assertFalse(axes.user_requests_end(text), text)

    def test_resume_refuses_data_from_before_the_interaction_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = self.make_generator(directory)
            args = generator.args
            args.dialogue_generation_mode = "aizuchi-only"
            args.resume = True
            protocol = gen.continuous_listener_protocol(args)
            protocol.pop("interaction_axes_version")
            output = Path(directory) / "dialogues.jsonl"
            before = json.dumps({"id": "old", "turns": [], "listener_protocol": protocol}) + "\n"
            output.write_text(before, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                gen.write_dialogues_only(args)
            self.assertEqual(output.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
