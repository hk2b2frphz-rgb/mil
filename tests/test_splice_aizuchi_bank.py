from __future__ import annotations

import argparse
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import qwen3_tts_diversity_probe as probe  # noqa: E402
import splice_aizuchi_bank as splice  # noqa: E402

SAMPLE_RATE = 24000


def tone(f0: float, seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * (np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(4 * np.pi * f0 * t))


def clip(seconds: float, amplitude: float = 0.3) -> dict:
    signal = tone(180.0, seconds, amplitude)
    return {
        "wav": f"wav/{seconds}.wav",
        "text": "うん",
        "temperature": 1.0,
        "signal": signal,
        "sample_rate": SAMPLE_RATE,
        "speech_sec": signal.size / SAMPLE_RATE,
    }


def default_args(**overrides) -> argparse.Namespace:
    values = {
        "aizuchi_max_chars": 6,
        "match_top_k": 1,
        "gain": "none",
        "match_text": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BackchannelTest(unittest.TestCase):
    def test_short_utterances_count_and_long_ones_do_not(self) -> None:
        self.assertTrue(splice.is_backchannel("うんうん", 6, None))
        self.assertTrue(splice.is_backchannel("そうなんだ。", 6, None))
        self.assertFalse(
            splice.is_backchannel("それは大変でしたね、よく話してくれました", 6, None)
        )
        self.assertFalse(splice.is_backchannel("   ", 6, None))

    def test_punctuation_does_not_push_a_word_over_the_limit(self) -> None:
        self.assertTrue(splice.is_backchannel("なるほど。", 4, None))

    def test_a_vocabulary_replaces_the_length_rule(self) -> None:
        vocab = {"うん", "そっか"}
        self.assertTrue(splice.is_backchannel("うん。", 99, vocab))
        # Short, but not in the list.
        self.assertFalse(splice.is_backchannel("はい", 99, vocab))


class ContinuerVocabTest(unittest.TestCase):
    VOCAB = REPO_ROOT / "scripts/2026-09-15/continuer_vocab.txt"

    def vocab(self) -> set[str]:
        return {
            line.strip()
            for line in self.VOCAB.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    def test_continuers_are_replaceable(self) -> None:
        vocab = self.vocab()
        for text in ("はい。", "ええ。", "うん。", "うん、うん。", "はい、はい。"):
            with self.subTest(text=text):
                self.assertTrue(splice.is_backchannel(text, 6, vocab))

    def test_assessments_are_not(self) -> None:
        # The complaint this file exists for: "sou-nan-desu-ne" is a reaction
        # to what was said, and replacing it with "un" deletes that reaction.
        vocab = self.vocab()
        for text in (
            "そうなんですね。",
            "そうでしたか。",
            "そうですか。",
            "なるほど。",
            "大丈夫ですよ。",
        ):
            with self.subTest(text=text):
                self.assertFalse(splice.is_backchannel(text, 6, vocab))

    def test_a_character_count_cannot_make_this_distinction(self) -> None:
        # Four characters, and not replaceable -- which is why the vocabulary
        # exists rather than a longer or shorter limit.
        self.assertTrue(splice.is_backchannel("なるほど。", 6, None))
        self.assertFalse(splice.is_backchannel("なるほど。", 6, self.vocab()))


class PickTest(unittest.TestCase):
    def test_the_nearest_length_is_taken_when_k_is_one(self) -> None:
        clips = [clip(0.2), clip(0.5), clip(0.9)]
        chosen, fell_back = splice.pick_clip(clips, 0.52, 1, random.Random(0))
        self.assertAlmostEqual(chosen["speech_sec"], 0.5, places=2)
        self.assertFalse(fell_back)

    def test_the_word_is_matched_before_the_length(self) -> None:
        # A dialogue that alternates "un" and "sokka" must not get "un" for
        # both, or the distinction it was built to carry disappears.
        clips = [clip(0.30), clip(0.32), {**clip(0.80), "text": "そっか"}]
        chosen, fell_back = splice.pick_clip(
            clips, 0.31, 1, random.Random(0), want_text="そっか"
        )
        self.assertEqual(chosen["text"], "そっか")
        self.assertFalse(fell_back)

    def test_a_word_the_bank_lacks_falls_back_and_says_so(self) -> None:
        clips = [clip(0.30), clip(0.32)]
        chosen, fell_back = splice.pick_clip(
            clips, 0.31, 1, random.Random(0), want_text="へー"
        )
        self.assertEqual(chosen["text"], "うん")
        self.assertTrue(fell_back)

    def test_a_wider_k_stops_the_same_clip_coming_back_every_time(self) -> None:
        # Always taking the closest gives a corpus where every backchannel is
        # the same recording; the point of the bank is that they differ.
        clips = [clip(0.30), clip(0.32), clip(0.34), clip(0.36)]
        rng = random.Random(0)
        picked = {
            splice.pick_clip(clips, 0.33, 4, rng)[0]["speech_sec"] for _ in range(20)
        }
        self.assertGreater(len(picked), 1)


class SpliceTest(unittest.TestCase):
    def dialogue(self) -> np.ndarray:
        # Left = listener, right = the other speaker talking throughout.
        stereo = np.zeros((2, SAMPLE_RATE * 4))
        stereo[1, :] = tone(120.0, 4.0, 0.2)[: stereo.shape[-1]]
        stereo[0, SAMPLE_RATE : SAMPLE_RATE + int(0.4 * SAMPLE_RATE)] = tone(
            200.0, 0.4, 0.5
        )
        return stereo

    def rows(self) -> list:
        return [
            ["ええ", [1.0, 1.4], "SPEAKER_MAIN"],
            ["今日は本当に疲れてしまって、何もする気が起きなくて", [0.0, 4.0], "SPEAKER_USER"],
        ]

    def test_only_the_listener_channel_is_touched(self) -> None:
        stereo = self.dialogue()
        out, _rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [clip(0.4)], default_args(),
            random.Random(0), None,
        )
        self.assertEqual(len(swaps), 1)
        np.testing.assert_array_equal(out[1], stereo[1])
        self.assertFalse(np.array_equal(out[0], stereo[0]))

    def test_audio_outside_the_slot_is_left_alone(self) -> None:
        stereo = self.dialogue()
        out, _rows, _swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [clip(0.4)], default_args(),
            random.Random(0), None,
        )
        np.testing.assert_array_equal(out[0, : SAMPLE_RATE], stereo[0, : SAMPLE_RATE])

    def test_a_long_clip_runs_past_the_slot_instead_of_being_cut(self) -> None:
        # Overrunning into the other speaker is what a backchannel does; the
        # slot is where it starts, not a box to fit inside.
        stereo = self.dialogue()
        out, rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [clip(0.9)], default_args(),
            random.Random(0), None,
        )
        self.assertGreater(swaps[0]["length_error_sec"], 0.4)
        self.assertFalse(swaps[0]["truncated"])
        end = int(1.5 * SAMPLE_RATE)
        self.assertGreater(float(np.max(np.abs(out[0, end : end + 100]))), 0.0)
        # The alignment must follow the audio, not the slot it replaced.
        main = [row for row in rows if row[2] == "SPEAKER_MAIN"][0]
        self.assertAlmostEqual(main[1][1] - main[1][0], swaps[0]["placed_sec"], places=2)

    def test_a_clip_running_past_the_end_is_cut_and_says_so(self) -> None:
        stereo = self.dialogue()
        rows = [["ええ", [3.8, 3.9], "SPEAKER_MAIN"]]
        _out, _rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, rows, [clip(0.9)], default_args(),
            random.Random(0), None,
        )
        self.assertTrue(swaps[0]["truncated"])
        self.assertLessEqual(swaps[0]["placed_sec"], 0.21)

    def test_the_original_backchannel_is_removed_not_mixed_over(self) -> None:
        stereo = self.dialogue()
        out, _rows, _swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [clip(0.05)], default_args(),
            random.Random(0), None,
        )
        # The slot was 0.4s and the clip is 0.05s; the rest must be silent, not
        # the tail of the backchannel KABURI synthesized.
        tail = out[0, SAMPLE_RATE + int(0.1 * SAMPLE_RATE) : SAMPLE_RATE + int(0.4 * SAMPLE_RATE)]
        self.assertEqual(float(np.max(np.abs(tail))), 0.0)

    def test_gain_matching_follows_the_replaced_segment(self) -> None:
        stereo = self.dialogue()
        quiet = clip(0.4, amplitude=0.01)
        out, _rows, _swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [quiet], default_args(gain="match"),
            random.Random(0), None,
        )
        placed = out[0, SAMPLE_RATE : SAMPLE_RATE + int(0.4 * SAMPLE_RATE)]
        original = stereo[0, SAMPLE_RATE : SAMPLE_RATE + int(0.4 * SAMPLE_RATE)]
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(placed**2))),
            float(np.sqrt(np.mean(original**2))),
            delta=0.02,
        )

    def test_a_long_listener_turn_is_not_treated_as_a_backchannel(self) -> None:
        stereo = self.dialogue()
        rows = [["それは本当に大変でしたね、よく頑張りました", [1.0, 1.4], "SPEAKER_MAIN"]]
        out, _rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, rows, [clip(0.4)], default_args(),
            random.Random(0), None,
        )
        self.assertEqual(swaps, [])
        np.testing.assert_array_equal(out, stereo)

    def test_the_result_is_not_clipped(self) -> None:
        stereo = self.dialogue()
        out, _rows, _swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, self.rows(), [clip(0.4, amplitude=3.0)],
            default_args(), random.Random(0), None,
        )
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)


class AnchorTransferTest(unittest.TestCase):
    def anchors_of(self, placements) -> list[dict]:
        return splice.backchannel_anchors(placements, 6, None)

    def test_a_backchannel_inside_an_utterance_is_kept_inside(self) -> None:
        # The case the old gap-from-the-end framing could not express: the
        # listener comes in a quarter of the way through, not near the end.
        placements = [
            {"label": "SPEAKER_USER", "text": "今日は本当に疲れてしまって",
             "start_sec": 0.0, "end_sec": 8.0},
            {"label": "SPEAKER_MAIN", "text": "うん", "start_sec": 2.0, "end_sec": 2.3},
        ]
        anchor = self.anchors_of(placements)[0]
        self.assertEqual(anchor["mode"], "inside")
        self.assertAlmostEqual(anchor["value"], 0.25, places=3)

    def test_the_position_survives_a_differently_long_utterance(self) -> None:
        # KABURI ran the utterance for 8s; the real one runs for 4s. A quarter
        # of the way in has to stay a quarter of the way in, not land at 2.0s
        # of a 4s turn, and certainly not after it ends.
        rows = [["今日は本当に疲れてしまって", [10.0, 14.0], "SPEAKER_USER"],
                ["ええ", [14.5, 14.9], "SPEAKER_MAIN"]]
        moved = splice.retime_backchannels(
            rows, [{"anchor": 0, "mode": "inside", "value": 0.25}], 6, None, 20.0
        )
        self.assertAlmostEqual(moved[1], 11.0, places=3)

    def test_a_backchannel_after_the_utterance_keeps_its_gap(self) -> None:
        placements = [
            {"label": "SPEAKER_USER", "text": "今日は疲れました",
             "start_sec": 0.0, "end_sec": 3.0},
            {"label": "SPEAKER_MAIN", "text": "うん", "start_sec": 3.4, "end_sec": 3.7},
        ]
        anchor = self.anchors_of(placements)[0]
        self.assertEqual(anchor["mode"], "after")
        self.assertAlmostEqual(anchor["value"], 0.4, places=3)
        rows = [["今日は疲れました", [0.0, 5.0], "SPEAKER_USER"],
                ["ええ", [5.5, 5.9], "SPEAKER_MAIN"]]
        moved = splice.retime_backchannels(rows, [anchor], 6, None, 20.0)
        self.assertAlmostEqual(moved[1], 5.4, places=3)

    def test_the_anchor_is_the_utterance_in_progress_not_the_first_one(self) -> None:
        placements = [
            {"label": "SPEAKER_USER", "text": "あのう", "start_sec": 0.0, "end_sec": 1.0},
            {"label": "SPEAKER_USER", "text": "今日は本当に疲れてしまって",
             "start_sec": 2.0, "end_sec": 6.0},
            {"label": "SPEAKER_MAIN", "text": "うん", "start_sec": 4.0, "end_sec": 4.3},
        ]
        anchor = self.anchors_of(placements)[0]
        self.assertEqual(anchor["anchor"], 1)
        self.assertEqual(anchor["mode"], "inside")
        self.assertAlmostEqual(anchor["value"], 0.5, places=3)

    def test_a_backchannel_before_anyone_speaks_keeps_its_absolute_time(self) -> None:
        placements = [
            {"label": "SPEAKER_MAIN", "text": "うん", "start_sec": 0.5, "end_sec": 0.8},
            {"label": "SPEAKER_USER", "text": "あのう", "start_sec": 1.0, "end_sec": 2.0},
        ]
        anchor = self.anchors_of(placements)[0]
        self.assertEqual(anchor["mode"], "absolute")
        self.assertAlmostEqual(anchor["value"], 0.5, places=3)

    def test_long_listener_turns_are_not_anchored(self) -> None:
        placements = [
            {"label": "SPEAKER_USER", "text": "つかれた", "start_sec": 0.0, "end_sec": 3.0},
            {"label": "SPEAKER_MAIN",
             "text": "それは大変でしたね、よく話してくれました",
             "start_sec": 3.0, "end_sec": 5.0},
        ]
        self.assertEqual(self.anchors_of(placements), [])

    def test_a_mismatched_count_refuses_rather_than_guessing(self) -> None:
        rows = [["今日は疲れました", [0.0, 3.0], "SPEAKER_USER"],
                ["ええ", [3.5, 3.9], "SPEAKER_MAIN"]]
        anchors = [{"anchor": 0, "mode": "after", "value": 0.1}] * 2
        self.assertEqual(splice.retime_backchannels(rows, anchors, 6, None, 10.0), {})

    def test_a_move_never_lands_outside_the_file(self) -> None:
        rows = [["今日は疲れました", [0.0, 3.0], "SPEAKER_USER"],
                ["ええ", [3.5, 3.9], "SPEAKER_MAIN"]]
        low = splice.retime_backchannels(
            rows, [{"anchor": 0, "mode": "after", "value": -99.0}], 6, None, 10.0
        )
        high = splice.retime_backchannels(
            rows, [{"anchor": 0, "mode": "after", "value": 99.0}], 6, None, 10.0
        )
        self.assertEqual(low[1], 0.0)
        self.assertLessEqual(high[1], 10.0)


class RetimedSpliceTest(unittest.TestCase):
    def test_the_old_position_is_cleared_and_the_new_one_carries_the_audio(self) -> None:
        stereo = np.zeros((2, SAMPLE_RATE * 4))
        stereo[0, int(3.5 * SAMPLE_RATE) : int(3.9 * SAMPLE_RATE)] = tone(200.0, 0.4, 0.5)
        rows = [
            ["今日は本当に疲れてしまって", [0.0, 3.0], "SPEAKER_USER"],
            ["ええ", [3.5, 3.9], "SPEAKER_MAIN"],
        ]
        out, new_rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, rows, [clip(0.3)], default_args(),
            random.Random(0), None, {1: 2.8},
        )
        old = out[0, int(3.5 * SAMPLE_RATE) : int(3.9 * SAMPLE_RATE)]
        new = out[0, int(2.8 * SAMPLE_RATE) : int(3.05 * SAMPLE_RATE)]
        self.assertEqual(float(np.max(np.abs(old))), 0.0)
        self.assertGreater(float(np.max(np.abs(new))), 0.0)
        self.assertAlmostEqual(swaps[0]["moved_sec"], -0.7, places=2)
        main = [row for row in new_rows if row[2] == "SPEAKER_MAIN"][0]
        self.assertAlmostEqual(main[1][0], 2.8, places=2)

    def test_moving_one_backchannel_does_not_erase_another(self) -> None:
        # Erasing after placing would wipe a clip written into the slot a later
        # backchannel used to occupy.
        stereo = np.zeros((2, SAMPLE_RATE * 6))
        for begin in (1.0, 2.0):
            start = int(begin * SAMPLE_RATE)
            burst = tone(200.0, 0.3, 0.5)
            stereo[0, start : start + burst.size] = burst
        rows = [
            ["ええ", [1.0, 1.3], "SPEAKER_MAIN"],
            ["ええ", [2.0, 2.3], "SPEAKER_MAIN"],
        ]
        out, _rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, rows, [clip(0.3)], default_args(),
            random.Random(0), None, {0: 2.0, 1: 4.0},
        )
        self.assertEqual(len(swaps), 2)
        first = out[0, int(2.0 * SAMPLE_RATE) : int(2.3 * SAMPLE_RATE)]
        second = out[0, int(4.0 * SAMPLE_RATE) : int(4.3 * SAMPLE_RATE)]
        self.assertGreater(float(np.max(np.abs(first))), 0.0)
        self.assertGreater(float(np.max(np.abs(second))), 0.0)
        self.assertEqual(
            float(np.max(np.abs(out[0, int(1.0 * SAMPLE_RATE) : int(1.3 * SAMPLE_RATE)]))),
            0.0,
        )


class LayoutTest(unittest.TestCase):
    def test_a_training_set_resolves_to_its_data_stereo(self) -> None:
        # Every TTS route in this repo writes <training_set>/data_stereo/. The
        # splice looked at the training_set itself and so found nothing, even
        # for corpora that were there.
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            training_set = Path(directory) / "training_set"
            (training_set / "data_stereo").mkdir(parents=True)
            self.assertEqual(
                splice.stereo_dir(training_set), training_set / "data_stereo"
            )

    def test_data_stereo_itself_is_accepted(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_stereo"
            path.mkdir()
            self.assertEqual(splice.stereo_dir(path), path)


class PlacementIndexTest(unittest.TestCase):
    def test_placements_are_keyed_by_dialogue_id_not_filename(self) -> None:
        # Sharding renumbers stems, so sample_00001 in a shard is a different
        # dialogue from sample_00001 in the placement run. Only the id holds.
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for stem, dialogue_id in (("sample_00001_a", "d-042"), ("sample_00002_b", "d-007")):
                (path / f"{stem}.placement.json").write_text(
                    json.dumps({"id": dialogue_id, "stem": stem, "placements": []}),
                    encoding="utf-8",
                )
            index = splice.load_placements(path)
            self.assertEqual(sorted(index), ["d-007", "d-042"])
            self.assertEqual(index["d-042"]["stem"], "sample_00001_a")

    def test_an_empty_placement_directory_is_refused(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                splice.load_placements(Path(directory))


class BankLoadTest(unittest.TestCase):
    def build_bank(self, directory: Path) -> None:
        records = []
        for index, seconds in enumerate([0.25, 0.30, 0.35, 2.00]):
            # Leading silence, so the loader has something to trim.
            signal = np.concatenate([np.zeros(int(0.15 * SAMPLE_RATE)), tone(180.0, seconds)])
            name = f"wav/{index:04d}.wav"
            probe.write_wav(directory / name, signal, SAMPLE_RATE)
            records.append(
                {
                    "text": "うん" if index < 4 else "そっか",
                    "temperature": 1.0,
                    "wav": name,
                    "measure": probe.measure(signal, SAMPLE_RATE),
                }
            )
        records.append({**records[0], "text": "そっか"})
        (directory / "samples.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_leading_silence_is_trimmed_so_the_onset_lands_on_the_slot(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.build_bank(path)
            clips = splice.load_bank(path, "うん", None, drop_outliers=False)
            self.assertEqual(len(clips), 4)
            # 0.15s of silence gone, so the clip is about its tone length.
            # The trim runs at the 10ms analysis hop with a 40ms window, so it
            # lands within a frame or so rather than exactly.
            shortest = min(clips, key=lambda c: c["speech_sec"])
            self.assertAlmostEqual(shortest["speech_sec"], 0.25, delta=0.05)

    def test_outliers_are_dropped_by_default(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.build_bank(path)
            kept = splice.load_bank(path, "うん", None, drop_outliers=True)
            self.assertEqual(len(kept), 3)
            self.assertLess(max(c["speech_sec"] for c in kept), 1.0)

    def test_the_text_filter_selects(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.build_bank(path)
            self.assertEqual(len(splice.load_bank(path, "そっか", None, False)), 1)

    def test_an_empty_selection_is_refused_rather_than_silently_empty(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.build_bank(path)
            with self.assertRaises(SystemExit):
                splice.load_bank(path, "ありえない語", None, False)


if __name__ == "__main__":
    unittest.main()
