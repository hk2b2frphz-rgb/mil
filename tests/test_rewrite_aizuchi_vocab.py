from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rewrite_aizuchi_vocab as rewriter  # noqa: E402


def dialogue(*turns: tuple[str, str]) -> dict:
    return {
        "id": "d-1",
        "turns": [{"speaker": speaker, "text": text} for speaker, text in turns],
    }


class DistributionTest(unittest.TestCase):
    def test_the_bundled_distribution_is_the_measured_one(self) -> None:
        words, weights = rewriter.load_distribution(rewriter.DEFAULT_DIST)
        self.assertEqual(words[:4], ["うん", "うーん", "そっか", "うんうん"])
        # The weights are the raw counts, so they stay checkable against the
        # recording: the top four were 88 of the 116 backchannels measured.
        self.assertEqual(sum(weights[:4]), 88.0)
        # The file keeps only the seven words with a usable n; the rest of the
        # 116 were singletons, which is why the total falls short of it.
        self.assertEqual(sum(weights), 103.0)

    def test_comments_and_blanks_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.tsv"
            path.write_text("# note\n\nうん\t2\nはい\t1\n", encoding="utf-8")
            self.assertEqual(rewriter.load_distribution(path), (["うん", "はい"], [2.0, 1.0]))

    def test_a_weightless_entry_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.tsv"
            path.write_text("うん\nはい\n", encoding="utf-8")
            self.assertEqual(rewriter.load_distribution(path)[1], [1.0, 1.0])


class RewriteTest(unittest.TestCase):
    def test_only_the_listener_is_rewritten(self) -> None:
        source = [dialogue(("user", "今日は疲れました"), ("moshi", "はい。"))]
        out, replaced, _kept = rewriter.rewrite(
            source, ["うん"], [1.0], 12, False, random.Random(0)
        )
        turns = out[0]["turns"]
        self.assertEqual(turns[0]["text"], "今日は疲れました")
        self.assertEqual(turns[1]["text"], "うん")
        self.assertEqual(replaced, {"うん": 1})

    def test_the_reading_follows_the_new_word(self) -> None:
        # tts_text left pointing at the old word would synthesize "hai" while
        # the transcript said "un".
        source = [dialogue(("moshi", "はい。"))]
        out, _replaced, _kept = rewriter.rewrite(
            source, ["うん"], [1.0], 12, False, random.Random(0)
        )
        self.assertEqual(out[0]["turns"][0]["tts_text"], "うん")
        self.assertEqual(out[0]["turns"][0]["original_text"], "はい。")

    def test_a_long_listener_turn_is_left_alone(self) -> None:
        long_turn = "それは本当に大変でしたね、よく話してくださいました"
        source = [dialogue(("moshi", long_turn))]
        out, replaced, kept = rewriter.rewrite(
            source, ["うん"], [1.0], 12, False, random.Random(0)
        )
        self.assertEqual(out[0]["turns"][0]["text"], long_turn)
        self.assertEqual(replaced, {})
        self.assertEqual(kept, {long_turn: 1})

    def test_all_listener_turns_can_be_forced(self) -> None:
        long_turn = "それは本当に大変でしたね、よく話してくださいました"
        out, replaced, _kept = rewriter.rewrite(
            [dialogue(("moshi", long_turn))], ["うん"], [1.0], 12, True, random.Random(0)
        )
        self.assertEqual(out[0]["turns"][0]["text"], "うん")
        self.assertEqual(replaced, {"うん": 1})

    def test_the_weights_are_followed(self) -> None:
        source = [dialogue(*[("moshi", "はい。")] * 400)]
        _out, replaced, _kept = rewriter.rewrite(
            source, ["うん", "はい"], [9.0, 1.0], 12, False, random.Random(0)
        )
        self.assertGreater(replaced["うん"], replaced.get("はい", 0) * 4)

    def test_the_same_seed_gives_the_same_corpus(self) -> None:
        source = [dialogue(*[("moshi", "はい。")] * 30)]
        first, _r, _k = rewriter.rewrite(
            source, ["うん", "そっか"], [1.0, 1.0], 12, False, random.Random(7)
        )
        second, _r, _k = rewriter.rewrite(
            source, ["うん", "そっか"], [1.0, 1.0], 12, False, random.Random(7)
        )
        self.assertEqual(
            [t["text"] for t in first[0]["turns"]],
            [t["text"] for t in second[0]["turns"]],
        )


class CliTest(unittest.TestCase):
    def test_the_vocabulary_file_lists_what_was_actually_used(self) -> None:
        # The splice needs it: a rewritten backchannel not named in the vocab
        # file is treated as out-of-vocabulary and silently left alone.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "in.jsonl"
            source.write_text(
                json.dumps(dialogue(("user", "つかれた"), ("moshi", "はい。")), ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            out = path / "out.jsonl"
            rewriter.main(
                ["--dialogues-jsonl", str(source), "--out-jsonl", str(out), "--text", "うん"]
            )
            written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(written[0]["turns"][1]["text"], "うん")
            vocab = Path(str(out) + ".vocab.txt").read_text(encoding="utf-8")
            self.assertIn("うん", vocab)


class SpeakerOnlyCopyTest(unittest.TestCase):
    """The speaker-only copy is what gets synthesized, so anything dropped from
    it never reaches the audio. Only backchannels may be dropped -- the bank
    supplies those. The opening greeting and the reply to "are you still
    there?" have to be synthesized, and a character count cannot tell them
    apart from a backchannel ("hai, kiite-imasu." and "hai, daijoubu desu yo."
    are both 9 characters, and only the second is in the vocabulary)."""

    VOCAB = REPO_ROOT / "scripts/2026-09-15/aizuchi_vocab_no_bare_un.tsv"

    def run_rewrite(self, *extra: str) -> tuple[list[str], str]:
        rows = dialogue(
            ("moshi", "もしもし、こちら孤独孤立相談窓口になります。"),
            ("user", "少し話してもいいですか。"),
            ("moshi", "はい。"),
            ("user", "あの、聞いていますか。"),
            ("moshi", "はい、聞いていますよ。"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "in.jsonl"
            source.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
            out = path / "out.jsonl"
            user_only = path / "user_only.jsonl"
            rewriter.main([
                "--dialogues-jsonl", str(source), "--out-jsonl", str(out),
                "--user-only-jsonl", str(user_only), "--keep-text", *extra,
            ])
            kept = json.loads(user_only.read_text(encoding="utf-8"))["turns"]
            vocab = Path(str(out) + ".vocab.txt").read_text(encoding="utf-8")
        return [t["text"] for t in kept if t["speaker"] == "moshi"], vocab

    def test_with_the_vocabulary_only_backchannels_are_left_to_the_bank(self) -> None:
        synthesized, vocab = self.run_rewrite("--aizuchi-vocab-file", str(self.VOCAB))
        self.assertEqual(
            synthesized,
            ["もしもし、こちら孤独孤立相談窓口になります。", "はい、聞いていますよ。"],
        )
        # And the splice is only told about the real backchannel, so it never
        # goes looking for a probe reply the bank does not have.
        self.assertIn("はい", vocab)
        self.assertNotIn("聞いていますよ", vocab)

    def test_drop_greeting_leaves_the_probe_reply_alone(self) -> None:
        # The backchannel-only corpus wants no opening, but the reply to
        # "are you still there?" is a real response the listener has to make
        # and the bank has no clip for it.
        synthesized, _vocab = self.run_rewrite(
            "--aizuchi-vocab-file", str(self.VOCAB), "--drop-greeting"
        )
        self.assertEqual(synthesized, ["はい、聞いていますよ。"])

    def test_the_copied_greeting_matches_the_generator(self) -> None:
        import generate_synthetic_moshi_training_data as generator

        self.assertEqual(
            rewriter.AIZUCHI_ONLY_GREETING, generator.AIZUCHI_ONLY_GREETING
        )

    def test_without_it_the_greeting_and_the_reply_are_lost(self) -> None:
        # The behaviour this option exists to prevent.
        synthesized, vocab = self.run_rewrite()
        self.assertNotIn("はい、聞いていますよ。", synthesized)
        self.assertIn("聞いていますよ", vocab)


if __name__ == "__main__":
    unittest.main()
