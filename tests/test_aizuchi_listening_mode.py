from __future__ import annotations

import sys
import tempfile
import unittest
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
        # earlier prompt told the model never to do that; it was wrong.
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertNotIn("並べないでください", prompt.user)
        self.assertIn("裸の一語（「うん」「そっか」）は普通に使ってください", prompt.user)

    def test_the_example_is_omitted_by_default(self) -> None:
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertNotIn("娘が一人いるんですけど", prompt.user)

    def test_the_real_example_appears_when_requested(self) -> None:
        # Drawn from an actual transcript rather than invented, with the
        # non-backchannel half of a fused turn ("sokka-. nan'nensei kana?")
        # stripped out -- this mode only reproduces the backchannel.
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"], with_example=True
        )
        self.assertIn("娘が一人いるんですけど", prompt.user)
        self.assertIn("実際の相談ダイヤルの書き起こしから", prompt.user)
        self.assertNotIn("何年生かな", prompt.user)

    def test_the_prompt_names_the_last_clause_and_the_vocabulary(self) -> None:
        prompt = gen.build_aizuchi_listening_prompt(
            {"id": "x"}, [], self.TEXT, ["うん", "そっか"]
        )
        self.assertIn("1 から 3 まで", prompt.user)
        self.assertIn("うん / そっか", prompt.user)
        # The end of the utterance must be asked for explicitly.
        self.assertIn("言い切った所には受け止めを置いてください", prompt.user)
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


if __name__ == "__main__":
    unittest.main()
