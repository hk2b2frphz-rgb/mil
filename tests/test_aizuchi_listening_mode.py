from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_synthetic_moshi_training_data as gen  # noqa: E402

REAL_DIST = REPO_ROOT / "scripts/2026-09-15/real_backchannel_dist.tsv"


class VocabSwapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = set(gen.AIZUCHI_ACTIVE_VOCAB)
        self.window = gen.AIZUCHI_NO_REPEAT_WINDOW

    def tearDown(self) -> None:
        gen.AIZUCHI_ACTIVE_VOCAB = self.saved
        gen.AIZUCHI_NO_REPEAT_WINDOW = self.window

    def test_the_real_list_loads_from_the_weighted_file(self) -> None:
        words = gen.load_aizuchi_vocab_file(REAL_DIST)
        self.assertEqual(words[:4], ["うん", "うーん", "そっか", "うんうん"])

    def test_none_of_the_real_top_four_is_in_the_written_vocabulary(self) -> None:
        # The reason this mode exists: a corpus built on the written list
        # cannot be used to study the words people actually say.
        written = {w.strip("。、…") for w in gen.AIZUCHI_ONLY_VOCAB}
        for word in ("うーん", "そっか", "うんうん"):
            self.assertNotIn(word, written)

    def test_swapping_changes_what_the_parser_accepts(self) -> None:
        gen.set_aizuchi_vocab(gen.load_aizuchi_vocab_file(REAL_DIST), 0)
        clauses = ["今日は疲れて", "何もできなくて"]
        raw = '{"reactions":[{"after_clause":1,"text":"そっか"}]}'
        self.assertEqual(
            gen.parse_aizuchi_listening_reactions(raw, clauses),
            [{"after_clause": 1, "text": "そっか"}],
        )
        # A word from the written list is now out of vocabulary.
        raw = '{"reactions":[{"after_clause":1,"text":"そうなんですね。"}]}'
        self.assertEqual(gen.parse_aizuchi_listening_reactions(raw, clauses), [])

    def test_a_missing_or_empty_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v.txt"
            path.write_text("# only a comment\n\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                gen.load_aizuchi_vocab_file(path)


class ListeningParseTest(unittest.TestCase):
    def clauses(self) -> list[str]:
        return gen.split_text_into_clauses(self.TEXT)

    TEXT = "今日は疲れて、何もできなくて、ずっと横になっていました。"

    def test_the_model_may_choose_any_clause_including_the_last(self) -> None:
        raw = (
            '{"reactions":[{"after_clause":1,"text":"うん。"},'
            '{"after_clause":3,"text":"はい。"}]}'
        )
        parsed = gen.parse_aizuchi_listening_reactions(raw, self.clauses())
        self.assertEqual([r["after_clause"] for r in parsed], [1, 3])

    def test_there_is_no_cap_on_how_many(self) -> None:
        # The rule-based route capped this per utterance; listening does not.
        raw = (
            '{"reactions":[{"after_clause":1,"text":"うん。"},'
            '{"after_clause":2,"text":"はい。"},'
            '{"after_clause":3,"text":"ええ。"}]}'
        )
        self.assertEqual(len(gen.parse_aizuchi_listening_reactions(raw, self.clauses())), 3)

    def test_out_of_range_and_duplicate_positions_are_dropped(self) -> None:
        raw = (
            '{"reactions":[{"after_clause":0,"text":"うん。"},'
            '{"after_clause":9,"text":"うん。"},'
            '{"after_clause":2,"text":"はい。"},'
            '{"after_clause":2,"text":"ええ。"}]}'
        )
        parsed = gen.parse_aizuchi_listening_reactions(raw, self.clauses())
        self.assertEqual(parsed, [{"after_clause": 2, "text": "はい。"}])

    def test_junk_returns_nothing_rather_than_raising(self) -> None:
        self.assertEqual(gen.parse_aizuchi_listening_reactions("no json here", self.clauses()), [])

    def test_bare_single_words_are_no_longer_forbidden(self) -> None:
        # The real recording showed "un" repeated bare, back to back. The
        # earlier prompt told the model never to do that; it was wrong. The
        # instruction is worded generically (no hardcoded word) so it stays
        # correct across a vocabulary swap.
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertNotIn("並べないでください", prompt.user)
        self.assertIn("語彙にある形をそのまま使います", prompt.user)

    def test_the_example_is_omitted_by_default(self) -> None:
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertNotIn("娘が一人いるんですけど", prompt.user)

    def test_the_real_example_appears_when_requested(self) -> None:
        # Drawn from actual transcript excerpts rather than invented, with the
        # non-backchannel half of each fused turn ("sokka-. nan'nensei kana?")
        # stripped out -- this mode only reproduces the backchannel. Multiple
        # examples, not one, since a single excerpt only ever shows bare
        # words and one elongation, not the other real patterns (a repeated
        # run, two backchannels fused into one word).
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"], with_example=True
        )
        self.assertIn("娘が一人いるんですけど", prompt.user)
        self.assertIn("例1（", prompt.user)
        self.assertIn("例2（", prompt.user)
        self.assertIn("例3（", prompt.user)
        self.assertNotIn("何年生かな", prompt.user)

    def test_the_prompt_names_the_last_clause_and_the_vocabulary(self) -> None:
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertIn("（1〜3）", prompt.user)
        self.assertIn("うん / そっか", prompt.user)
        # The end of the utterance must be asked for explicitly.
        self.assertIn("言い切った所）には必ず置いてください", prompt.user)
        self.assertIn("上限はありません", prompt.user)


class RepeatWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = set(gen.AIZUCHI_ACTIVE_VOCAB)
        self.window = gen.AIZUCHI_NO_REPEAT_WINDOW

    def tearDown(self) -> None:
        gen.AIZUCHI_ACTIVE_VOCAB = self.saved
        gen.AIZUCHI_NO_REPEAT_WINDOW = self.window

    def turns(self) -> list:
        # "un" three times over, each answering a user turn.
        out = [gen.DialogueTurn("moshi", gen.AIZUCHI_ONLY_GREETING)]
        for _ in range(3):
            out.append(gen.DialogueTurn("user", "今日は疲れました"))
            out.append(gen.DialogueTurn("moshi", "うん"))
        return out

    def test_the_default_window_blocks_the_repeat(self) -> None:
        gen.set_aizuchi_vocab(["うん"], 3)
        kept = gen.sanitize_aizuchi_only_turns(self.turns())
        self.assertEqual(sum(1 for t in kept if t.text == "うん"), 1)

    def test_a_zero_window_lets_a_small_vocabulary_repeat(self) -> None:
        # With seven words, forbidding a word for three turns would make the
        # commonest backchannel unusable; the recording had "un" as one in four.
        gen.set_aizuchi_vocab(["うん"], 0)
        kept = gen.sanitize_aizuchi_only_turns(self.turns())
        self.assertEqual(sum(1 for t in kept if t.text == "うん"), 3)


class ThinkingPromptTest(unittest.TestCase):
    def test_force_think_appends_the_slash_command(self) -> None:
        prompt = gen.messages_to_completion_prompt(
            [{"role": "user", "content": "hello"}], force_think=True
        )
        self.assertIn("/think", prompt)
        self.assertNotIn("/no_think", prompt)

    def test_no_think_still_wins_when_force_think_is_false(self) -> None:
        prompt = gen.messages_to_completion_prompt(
            [{"role": "user", "content": "hello"}], no_think=True, force_think=False
        )
        self.assertIn("/no_think", prompt)

    def test_a_leaked_think_block_does_not_break_json_extraction(self) -> None:
        # A reasoning model with no server-side parser configured leaks
        # <think>...</think> straight into the content channel; the answer
        # that follows must still parse.
        gen.set_aizuchi_vocab(["うん"], 0)
        raw = (
            "<think>まだ続きそうなので軽く受ける。</think>\n"
            '{"reactions":[{"after_clause":1,"text":"うん"}]}'
        )
        self.assertEqual(
            gen.parse_aizuchi_listening_reactions(raw, ["今日は疲れて"]),
            [{"after_clause": 1, "text": "うん"}],
        )

    def test_an_unclosed_think_block_yields_no_answer(self) -> None:
        # The whole token budget went to reasoning; there is nothing to parse,
        # and this must not be mistaken for valid JSON.
        raw = "<think>ここでずっと考え続けて token 切れ"
        self.assertEqual(gen.parse_aizuchi_listening_reactions(raw, ["a"]), [])


class ThinkingTruncationTest(unittest.TestCase):
    def test_an_unclosed_think_block_is_truncated(self) -> None:
        self.assertTrue(
            gen.aizuchi_thinking_truncated("<think>まだ考えている途中で token が切れて")
        )

    def test_a_deliberately_empty_reactions_list_is_not_truncated(self) -> None:
        # The model finished reasoning and correctly decided nothing was
        # worth responding to; this must not be treated as a failure that
        # deserves a retry.
        self.assertFalse(
            gen.aizuchi_thinking_truncated(
                "<think>短いので反応不要。</think>\n" '{"reactions":[]}'
            )
        )

    def test_a_normal_closed_response_is_not_truncated(self) -> None:
        self.assertFalse(
            gen.aizuchi_thinking_truncated(
                "<think>ここで反応。</think>\n"
                '{"reactions":[{"after_clause":1,"text":"うん"}]}'
            )
        )

    def test_plain_json_with_no_thinking_at_all_is_not_truncated(self) -> None:
        # enable_thinking=False still runs through this check on the retry
        # path; ordinary output must not be flagged.
        self.assertFalse(
            gen.aizuchi_thinking_truncated('{"reactions":[{"after_clause":1,"text":"うん"}]}')
        )


class CompletionsThinkingOverrideTest(unittest.TestCase):
    """enable_thinking=True/False must win outright on the completions
    endpoint, not just enable_thinking=True. A previous version of the branch
    only special-cased True, so the aizuchi thinking-timeout retry's explicit
    False fell through to the job-wide default and did nothing -- on a job
    that had not set --llm-completions-no-think, the retry kept thinking on,
    now with a smaller max_tokens than the call it was retrying."""

    def make_generator(self, *, llm_completions_no_think: bool) -> gen.LLMDialogueGenerator:
        args = argparse.Namespace(
            llm_backend="openai-compatible",
            llm_openai_endpoint="completions",
            llm_completions_no_think=llm_completions_no_think,
            llm_api_base="http://localhost:9",
            llm_api_key="",
            llm_temperature=0.7,
            llm_frequency_penalty=0.0,
            llm_presence_penalty=0.0,
            llm_max_new_tokens=64,
            llm_repetition_penalty=0.0,
            llm_reasoning_effort="",
            llm_timeout_sec=5.0,
            llm_model="x",
            llm_served_model_name="",
            dialogue_generation_mode="multi-agent",
            out_dir=Path("/tmp"),
            llm_task="dialogue",
        )
        generator = gen.LLMDialogueGenerator(args)
        # Bypass the /v1/models discovery call resolve_served_model() makes
        # on first use -- irrelevant to what this test checks, and would
        # otherwise need its own network stub.
        generator._served_model = "x"
        return generator

    def captured_prompt(self, generator, **kwargs) -> str:
        # generate_openai_compatible_messages's own try/except only wraps the
        # urlopen() call, so the payload is captured at Request() construction
        # (unpatched, so it behaves normally) and the network failure is
        # injected at urlopen() instead, where the code actually expects and
        # handles one.
        captured = {}
        real_request = urllib.request.Request

        def spying_request(url, data=None, headers=None, method=None):
            captured["prompt"] = json.loads(data.decode("utf-8"))["prompt"]
            return real_request(url, data=data, headers=headers, method=method)

        with unittest.mock.patch("urllib.request.Request", side_effect=spying_request), \
             unittest.mock.patch(
                 "urllib.request.urlopen",
                 side_effect=urllib.error.URLError("no network in this test"),
             ):
            try:
                generator.generate_openai_compatible_messages(
                    [{"role": "user", "content": "hello"}], role_name="test", **kwargs
                )
            except RuntimeError:
                pass  # expected: the stubbed network call always fails
        return captured["prompt"]

    def test_explicit_false_forces_no_think_even_without_the_job_default(self) -> None:
        generator = self.make_generator(llm_completions_no_think=False)
        prompt = self.captured_prompt(generator, enable_thinking=False)
        self.assertIn("/no_think", prompt)

    def test_explicit_true_forces_think_even_with_the_job_default_on(self) -> None:
        generator = self.make_generator(llm_completions_no_think=True)
        prompt = self.captured_prompt(generator, enable_thinking=True)
        self.assertIn("/think", prompt)
        self.assertNotIn("/no_think", prompt)

    def test_none_falls_back_to_the_job_wide_default(self) -> None:
        generator = self.make_generator(llm_completions_no_think=True)
        prompt = self.captured_prompt(generator, enable_thinking=None)
        self.assertIn("/no_think", prompt)


class DensityInterpolationTest(unittest.TestCase):
    def test_zero_disables_backchannels_entirely(self) -> None:
        self.assertIsNone(gen.resolve_aizuchi_density(0.0))

    def test_the_named_anchors_match_the_existing_presets(self) -> None:
        reserved = gen.resolve_aizuchi_density(0.25)
        normal = gen.resolve_aizuchi_density(0.5)
        eager = gen.resolve_aizuchi_density(0.75)
        flood = gen.resolve_aizuchi_density(1.0)
        # "end" is deliberately not part of resolve_aizuchi_density's output --
        # the end position is guaranteed unconditionally elsewhere, not by
        # interpolating the presets' end rate -- so compare cont/weak only.
        for name, resolved in (
            ("reserved", reserved), ("normal", normal), ("eager", eager), ("flood", flood)
        ):
            preset_rates = gen.AIZUCHI_FREQUENCY_PRESETS[name]["rates"]
            self.assertEqual(resolved["rates"]["cont"], preset_rates["cont"], name)
            self.assertEqual(resolved["rates"]["weak"], preset_rates["weak"], name)

    def test_a_value_between_two_anchors_is_interpolated(self) -> None:
        low = gen.resolve_aizuchi_density(0.25)["rates"]["cont"]
        high = gen.resolve_aizuchi_density(0.5)["rates"]["cont"]
        mid = gen.resolve_aizuchi_density(0.375)["rates"]["cont"]
        self.assertAlmostEqual(mid, (low + high) / 2, places=6)

    def test_values_are_clamped_to_the_valid_range(self) -> None:
        self.assertEqual(gen.resolve_aizuchi_density(-1.0), gen.resolve_aizuchi_density(0.0))
        self.assertEqual(gen.resolve_aizuchi_density(5.0), gen.resolve_aizuchi_density(1.0))


class DensityPlacementTest(unittest.TestCase):
    CLAUSES = ["今日は疲れて、", "何もできなくて、", "ずっと横になっていました。"]

    def test_the_end_of_the_utterance_always_gets_a_point(self) -> None:
        # Even at the lowest nonzero density, over many seeds.
        frequency = gen.resolve_aizuchi_density(0.01)
        for seed in range(30):
            points = gen.pick_density_points(self.CLAUSES, frequency, random.Random(seed))
            self.assertEqual(points[-1]["after_clause"], len(self.CLAUSES))
            self.assertEqual(points[-1]["kind"], "end")

    def test_no_clauses_yields_no_points(self) -> None:
        # The density==0 case is handled upstream in _react_by_density
        # (resolve_aizuchi_density(0.0) returns None, short-circuiting before
        # this is ever called); this function's own empty-input guard is
        # simpler: no clauses, no points, regardless of frequency.
        self.assertEqual(gen.pick_density_points([], {}, random.Random(0)), [])

    def test_a_single_clause_utterance_still_gets_its_end_point(self) -> None:
        frequency = gen.resolve_aizuchi_density(0.5)
        points = gen.pick_density_points(["少し疲れました。"], frequency, random.Random(0))
        self.assertEqual(points, [{"after_clause": 1, "kind": "end"}])

    def test_max_per_turn_reserves_a_slot_for_the_end_point(self) -> None:
        # flood's max_per_turn is 6, so a long utterance must not exceed it
        # even counting the guaranteed end point.
        frequency = gen.resolve_aizuchi_density(1.0)
        clauses = [f"それで{i}、" for i in range(10)] + ["終わりです。"]
        points = gen.pick_density_points(clauses, frequency, random.Random(0))
        self.assertLessEqual(len(points), frequency["max_per_turn"])
        self.assertEqual(points[-1]["after_clause"], len(clauses))


class DensityDispatchTest(unittest.TestCase):
    """react_to_user_turn end to end in density mode: position is picked by
    probability (end guaranteed), but word choice still goes through the
    same LLM call as rule mode -- pick_aizuchi_words (pure random) was
    removed after the user pointed out word choice was never meant to be
    random, only WHERE to place a backchannel was."""

    def make_generator(self, density: float) -> gen.LLMDialogueGenerator:
        args = argparse.Namespace(
            llm_backend="template",
            dialogue_generation_mode="aizuchi-only",
            out_dir=Path("/tmp"),
            aizuchi_only_placement="density",
            aizuchi_density=density,
            aizuchi_only_example=0,
            multi_agent_aizuchi_temperature=0.3,
            llm_task="dialogue",
        )
        return gen.LLMDialogueGenerator(args)

    def setUp(self) -> None:
        self.saved = set(gen.AIZUCHI_ACTIVE_VOCAB)
        self.window = gen.AIZUCHI_NO_REPEAT_WINDOW
        gen.set_aizuchi_vocab(["うん", "そっか", "はい"], 0)

    def tearDown(self) -> None:
        gen.AIZUCHI_ACTIVE_VOCAB = self.saved
        gen.AIZUCHI_NO_REPEAT_WINDOW = self.window

    def test_a_multi_clause_utterance_ends_with_a_backchannel(self) -> None:
        generator = self.make_generator(0.5)
        user_turn = gen.DialogueTurn(
            "user", "今日は疲れて、何もできなくて、ずっと横になっていました。"
        )
        with unittest.mock.patch.object(
            generator, "call_agent", return_value='{"reactions":[{"after_clause": 3, "text": "はい"}]}'
        ) as mock_call:
            turns = generator.react_to_user_turn(
                use_case={"id": "x"},
                visible_turns=[],
                user_turn=user_turn,
                case_id="c1",
                block_index=0,
                frequency={},
                rng=random.Random(0),
            )
        mock_call.assert_called_once()
        self.assertEqual(turns[-1].speaker, "moshi")

    def test_density_zero_returns_the_utterance_untouched(self) -> None:
        generator = self.make_generator(0.0)
        user_turn = gen.DialogueTurn("user", "今日は疲れて、何もできなくて。")
        with unittest.mock.patch.object(generator, "call_agent") as mock_call:
            turns = generator.react_to_user_turn(
                use_case={"id": "x"},
                visible_turns=[],
                user_turn=user_turn,
                case_id="c1",
                block_index=0,
                frequency={},
                rng=random.Random(0),
            )
        mock_call.assert_not_called()
        self.assertEqual(turns, [user_turn])


class AizuchiFrequencyConditioningLabelTest(unittest.TestCase):
    """Dialogue.aizuchi_frequency_label -- the value inject_inner_thoughts.py
    --from-aizuchi-density later embeds as an unspoken tag so the model can be
    conditioned on how talkative the listener is, instead of the frequency
    only ever showing up baked into the generated text itself."""

    def test_non_density_placements_have_no_density_concept(self) -> None:
        self.assertIsNone(
            gen.resolve_aizuchi_only_density("rule", 0.5, False, random.Random(0))
        )
        self.assertIsNone(
            gen.resolve_aizuchi_only_density("llm", 0.5, True, random.Random(0))
        )

    def test_fixed_density_is_used_as_is(self) -> None:
        density = gen.resolve_aizuchi_only_density("density", 0.75, False, random.Random(0))
        self.assertEqual(density, 0.75)

    def test_mixed_density_draws_a_fresh_value_per_call(self) -> None:
        rng = random.Random(0)
        seen = {
            gen.resolve_aizuchi_only_density("density", 0.5, True, rng) for _ in range(20)
        }
        self.assertTrue(all(0.0 <= d < 1.0 for d in seen))
        self.assertGreater(len(seen), 1)

    def test_density_label_reports_the_actual_resolved_value(self) -> None:
        self.assertEqual(
            gen.aizuchi_only_frequency_label("density", "normal", 0.75), "density=0.75"
        )

    def test_rule_label_is_the_preset_name(self) -> None:
        self.assertEqual(
            gen.aizuchi_only_frequency_label("rule", "eager", None), "eager"
        )

    def test_llm_label_marks_frequency_as_uncontrolled(self) -> None:
        self.assertEqual(
            gen.aizuchi_only_frequency_label("llm", "normal", None), "llm"
        )


if __name__ == "__main__":
    unittest.main()
