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


if __name__ == "__main__":
    unittest.main()
