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
        "placement_mode": "auto",
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


class InsertTest(unittest.TestCase):
    def user_only(self) -> tuple[np.ndarray, list]:
        # What a user-only render looks like: the right channel carries the
        # speech, the listener channel is silent, and there is nothing to
        # replace -- the backchannels have to be put in, not moved.
        stereo = np.zeros((2, SAMPLE_RATE * 8))
        speech = tone(120.0, 6.0, 0.2)
        stereo[1, : speech.size] = speech
        rows = [["今日は本当に疲れてしまって", [0.0, 6.0], "SPEAKER_USER"]]
        return stereo, rows

    def test_a_backchannel_is_written_where_the_anchor_says(self) -> None:
        stereo, rows = self.user_only()
        anchors = [{"anchor": 0, "mode": "inside", "value": 0.5,
                    "text": "うん", "dur_sec": 0.3}]
        out, new_rows, swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.3)],
            default_args(gain="none"), random.Random(0),
        )
        self.assertEqual(len(swaps), 1)
        # Halfway through a 6s utterance.
        self.assertAlmostEqual(swaps[0]["start_sec"], 3.0, places=2)
        window = out[0, int(3.0 * SAMPLE_RATE) : int(3.25 * SAMPLE_RATE)]
        self.assertGreater(float(np.max(np.abs(window))), 0.0)
        # And it overlaps: the user is still speaking there.
        self.assertGreater(float(np.max(np.abs(out[1, int(3.1 * SAMPLE_RATE)]))), 0.0)

    def test_the_user_channel_is_untouched(self) -> None:
        stereo, rows = self.user_only()
        anchors = [{"anchor": 0, "mode": "inside", "value": 0.5,
                    "text": "うん", "dur_sec": 0.3}]
        out, _rows, _swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.3)],
            default_args(gain="none"), random.Random(0),
        )
        np.testing.assert_array_equal(out[1], stereo[1])

    def test_the_alignment_gains_a_row_for_the_inserted_word(self) -> None:
        stereo, rows = self.user_only()
        anchors = [{"anchor": 0, "mode": "after", "value": 0.4,
                    "text": "うん", "dur_sec": 0.3}]
        _out, new_rows, _swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.3)],
            default_args(gain="none"), random.Random(0),
        )
        self.assertEqual(len(new_rows), 2)
        added = [row for row in new_rows if row[2] == "SPEAKER_MAIN"][0]
        self.assertEqual(added[0], "うん")
        self.assertAlmostEqual(added[1][0], 6.4, places=2)

    def test_the_clip_is_chosen_against_kaburi_s_own_duration(self) -> None:
        # There is no slot to measure, so the length KABURI expected is the
        # only handle on how long the backchannel should be.
        stereo, rows = self.user_only()
        anchors = [{"anchor": 0, "mode": "inside", "value": 0.5,
                    "text": "うん", "dur_sec": 0.8}]
        _out, _rows, swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.2), clip(0.8)],
            default_args(gain="none"), random.Random(0),
        )
        self.assertAlmostEqual(swaps[0]["placed_sec"], 0.8, delta=0.05)

    def test_several_backchannels_all_land(self) -> None:
        stereo, rows = self.user_only()
        anchors = [
            {"anchor": 0, "mode": "inside", "value": frac,
             "text": "うん", "dur_sec": 0.3}
            for frac in (0.2, 0.5, 0.8)
        ]
        _out, new_rows, swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.3)],
            default_args(gain="none"), random.Random(0),
        )
        self.assertEqual(len(swaps), 3)
        self.assertEqual(sum(1 for row in new_rows if row[2] == "SPEAKER_MAIN"), 3)


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


class AnchorAcrossDifferentRowSplitsTest(unittest.TestCase):
    """The two sides do not agree on what a "user row" is. KABURI keeps one
    row per clause and drops silence turns; the render merges consecutive user
    turns into one utterance and honours silences. Anchoring by row index made
    most backchannels fall out of range, and the fallback handed the stored
    value back as an absolute time -- a ratio like 0.8 or a gap like 0.4 read
    as seconds -- so they stacked up in the first moments of the call.
    The character offset into the speaker's text is what both sides share."""

    def placements(self, clauses: int) -> list[dict]:
        rows, t = [], 0.0
        for i in range(1, clauses + 1):
            rows.append({"label": "SPEAKER_USER", "text": f"第{i}句ですけれども、",
                         "start_sec": t, "end_sec": t + 2.0})
            rows.append({"label": "SPEAKER_MAIN", "text": "うん",
                         "start_sec": t + 1.6, "end_sec": t + 2.0})
            t += 2.2
        return rows

    def merged_row(self, clauses: int, span: tuple[float, float]) -> list:
        text = "".join(f"第{i}句ですけれども、" for i in range(1, clauses + 1))
        return [text, [span[0], span[1]], "SPEAKER_USER"]

    def placed_times(self, clauses: int = 5) -> list[float]:
        anchors = splice.backchannel_anchors(self.placements(clauses), 6, None)
        rows = [self.merged_row(clauses, (0.3, 11.3))]
        stereo = np.zeros((2, SAMPLE_RATE * 12))
        _out, _rows, swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip(0.4)],
            default_args(), random.Random(0),
        )
        return [swap["start_sec"] for swap in swaps]

    def test_they_do_not_pile_up_at_the_start(self) -> None:
        times = self.placed_times()
        self.assertEqual(len(times), 5)
        self.assertEqual(len({round(t, 2) for t in times}), len(times))

    def test_they_keep_their_order_and_spread(self) -> None:
        times = self.placed_times()
        self.assertEqual(times, sorted(times))
        self.assertGreater(max(times) - min(times), 5.0)

    def test_they_stay_inside_the_utterance_they_belong_to(self) -> None:
        for moment in self.placed_times():
            self.assertGreaterEqual(moment, 0.3)
            self.assertLessEqual(moment, 11.3)

    def test_the_offset_is_proportional_to_the_text_not_the_row_count(self) -> None:
        # Same dialogue, target rows split two different ways: the same
        # backchannel has to land at the same moment either way.
        anchors = splice.backchannel_anchors(self.placements(4), 6, None)
        whole = [self.merged_row(4, (0.0, 8.0))]
        halves = [
            ["".join(f"第{i}句ですけれども、" for i in (1, 2)), [0.0, 4.0], "SPEAKER_USER"],
            ["".join(f"第{i}句ですけれども、" for i in (3, 4)), [4.0, 8.0], "SPEAKER_USER"],
        ]
        for anchor in anchors:
            self.assertAlmostEqual(
                splice.anchor_time(anchor, whole, 0.0),
                splice.anchor_time(anchor, halves, 0.0),
                places=6,
            )

    def test_an_anchor_past_all_of_the_speech_lands_after_it(self) -> None:
        anchor = {"char_offset": 999.0, "after_sec": 0.4, "mode": "after"}
        rows = [["みじかい", [1.0, 3.0], "SPEAKER_USER"]]
        self.assertAlmostEqual(splice.anchor_time(anchor, rows, 0.0), 3.4, places=6)

    def test_an_old_anchor_without_the_offset_still_works(self) -> None:
        anchor = {"anchor": 0, "mode": "inside", "value": 0.25}
        rows = [["今日は本当に疲れてしまって", [10.0, 14.0], "SPEAKER_USER"]]
        self.assertAlmostEqual(splice.anchor_time(anchor, rows, 0.0), 11.0, places=6)


class LevelTest(unittest.TestCase):
    """This module promises it moves the other speaker's audio by not one
    sample. Two ways it used to break that promise, both through the level."""

    def speaker_channel(self, speech_ratio: float, amplitude: float = 0.3) -> np.ndarray:
        signal = np.zeros(SAMPLE_RATE * 10)
        spoken = int(SAMPLE_RATE * 10 * speech_ratio)
        signal[:spoken] = tone(120.0, spoken / SAMPLE_RATE, amplitude)[:spoken]
        return signal

    def insert(self, stereo, clips, anchors, gain="none"):
        rows = [["はなします", [0.0, 9.0], "SPEAKER_USER"]]
        return splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, clips,
            default_args(gain=gain), random.Random(0),
        )

    def test_overlapping_backchannels_do_not_turn_the_speaker_down(self) -> None:
        # Summing two clips can exceed 1.0 on the listener channel. Dividing
        # the whole array by that peak took the speaker down with it (measured
        # 0.90 -> 0.50), so a dialogue's speaker level depended on whether two
        # backchannels happened to land close together.
        stereo = np.zeros((2, SAMPLE_RATE * 10))
        stereo[1] = tone(120.0, 10.0, 0.9)[: stereo.shape[-1]]
        before = stereo[1].copy()
        loud = dict(clip(1.0, amplitude=0.9), text="はい。")
        anchors = [
            {"text": "はい。", "dur_sec": 1.0, "anchor": -1, "mode": "absolute", "value": v}
            for v in (2.0, 2.2)
        ]
        out, _rows, _swaps = self.insert(stereo, [loud], anchors)
        np.testing.assert_array_equal(out[1], before)
        self.assertLessEqual(float(np.max(np.abs(out[0]))), 1.0 + 1e-9)

    def test_gain_matching_follows_the_speech_not_the_silence(self) -> None:
        # The reference used to be the RMS of the whole speaker channel, so a
        # dialogue with more pauses got a quieter listener for no reason
        # (-3 dB at half silence, and a listening call has more than that).
        levels = []
        for ratio in (0.5, 0.9):
            stereo = np.zeros((2, SAMPLE_RATE * 10))
            stereo[1] = self.speaker_channel(ratio)
            anchors = [{"text": "うん", "dur_sec": 0.4, "anchor": 0,
                        "mode": "after", "value": 0.2}]
            out, _rows, _swaps = self.insert(stereo, [clip(0.4)], anchors, gain="match")
            placed = out[0][np.abs(out[0]) > 1e-9]
            levels.append(float(np.sqrt(np.mean(np.square(placed)))))
        # Same speaker loudness, different amounts of silence -> same listener.
        self.assertAlmostEqual(levels[0], levels[1], places=2)

    def test_the_listener_lands_at_the_speakers_speaking_level(self) -> None:
        stereo = np.zeros((2, SAMPLE_RATE * 10))
        speaker = self.speaker_channel(0.5)
        stereo[1] = speaker
        anchors = [{"text": "うん", "dur_sec": 0.4, "anchor": 0,
                    "mode": "after", "value": 0.2}]
        out, _rows, _swaps = self.insert(stereo, [clip(0.4)], anchors, gain="match")
        spoken = speaker[np.abs(speaker) > 1e-9]
        placed = out[0][np.abs(out[0]) > 1e-9]
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(placed)))),
            float(np.sqrt(np.mean(np.square(spoken)))),
            places=2,
        )

    def test_active_rms_ignores_the_silence(self) -> None:
        signal = np.zeros(1000)
        signal[:500] = 0.4
        self.assertAlmostEqual(splice.active_rms(signal), 0.4, places=6)

    def test_active_rms_of_pure_silence_is_zero(self) -> None:
        self.assertEqual(splice.active_rms(np.zeros(100)), 0.0)


class AlignmentEndTest(unittest.TestCase):
    """Bank clips keep their trailing silence on purpose (the 5%-of-peak
    silence test would otherwise clip the soft decay of "un"), but the
    alignment must still close at the end of the SOUND. alignment_words splits
    an entry longer than 8 characters into words spread across its span, so an
    end inflated by trailing silence puts the later words where nothing is
    audible -- teaching the model to run text on with no audio under it."""

    def clip_with_trailing_silence(self, speech: float, silence: float) -> dict:
        signal = np.concatenate([
            tone(180.0, speech), np.zeros(int(SAMPLE_RATE * silence))
        ])
        return {
            "wav": "wav/padded.wav", "text": "そうだったんですね。",
            "temperature": 1.0, "signal": signal, "sample_rate": SAMPLE_RATE,
            "speech_sec": speech,
        }

    def insert(self, clip: dict) -> list:
        stereo = np.zeros((2, SAMPLE_RATE * 10))
        rows = [["今日は疲れました", [0.0, 8.0], "SPEAKER_USER"]]
        anchors = [{
            "text": clip["text"], "dur_sec": clip["speech_sec"],
            "anchor": 0, "mode": "after", "value": 0.5,
        }]
        _out, new_rows, _swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip],
            default_args(match_text=True, aizuchi_max_chars=12), random.Random(0),
        )
        return [row for row in new_rows if row[2] == "SPEAKER_MAIN"]

    def test_the_entry_ends_at_the_end_of_the_sound(self) -> None:
        placed = self.insert(self.clip_with_trailing_silence(0.4, 0.6))[0]
        self.assertAlmostEqual(placed[1][1] - placed[1][0], 0.4, places=3)

    def test_every_split_word_lands_on_audio(self) -> None:
        from alignment_words import split_utterance_alignments

        placed = self.insert(self.clip_with_trailing_silence(0.4, 0.6))[0]
        words, _stats = split_utterance_alignments([placed])
        self.assertGreater(len(words), 1)  # long enough to be split
        audio_ends = placed[1][0] + 0.4
        for word in words:
            self.assertLessEqual(word[1][0], audio_ends + 1e-6, word)

    def test_the_trailing_silence_is_still_written_to_the_audio(self) -> None:
        # Only the alignment is shortened. Cutting the audio would take the
        # decay with it, which is the reason the tail is kept in the first place.
        clip = self.clip_with_trailing_silence(0.4, 0.6)
        stereo = np.zeros((2, SAMPLE_RATE * 10))
        rows = [["今日は疲れました", [0.0, 8.0], "SPEAKER_USER"]]
        anchors = [{"text": clip["text"], "dur_sec": 0.4, "anchor": 0,
                    "mode": "after", "value": 0.5}]
        _out, _rows, swaps = splice.insert_backchannels(
            stereo, SAMPLE_RATE, rows, anchors, [clip],
            default_args(match_text=True, aizuchi_max_chars=12), random.Random(0),
        )
        self.assertAlmostEqual(swaps[0]["placed_sec"], 1.0, places=2)

    def test_a_clip_with_no_trailing_silence_is_unchanged(self) -> None:
        placed = self.insert(self.clip_with_trailing_silence(0.4, 0.0))[0]
        self.assertAlmostEqual(placed[1][1] - placed[1][0], 0.4, places=3)

    def test_truncation_still_shortens_the_entry(self) -> None:
        # Cut by the end of the dialogue: the entry follows the audio that
        # actually fit, not the clip's nominal speech length.
        clip = dict(self.clip_with_trailing_silence(0.9, 0.0))
        self.assertAlmostEqual(
            splice.speech_written(clip, int(SAMPLE_RATE * 0.2), SAMPLE_RATE), 0.2
        )

    def test_a_clip_without_a_measurement_falls_back_to_the_audio(self) -> None:
        self.assertAlmostEqual(
            splice.speech_written({}, int(SAMPLE_RATE * 0.3), SAMPLE_RATE), 0.3
        )


class FadeTest(unittest.TestCase):
    """A backchannel sits alone in an otherwise silent listener channel, so a
    discontinuity at its edge is audible as a click rather than being masked.
    The tail matters most: a clip cut off at the end of the dialogue used to
    stop mid-waveform, because the fade was applied before the truncation and
    went out with the part that got cut."""

    def steady(self, seconds: float) -> np.ndarray:
        # Never near zero, so any cut leaves a step.
        return 0.5 * np.sin(2 * np.pi * 200 * np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE)

    def test_the_tail_is_taken_to_zero(self) -> None:
        faded = splice.fade_out(self.steady(0.5), SAMPLE_RATE)
        self.assertAlmostEqual(float(faded[-1]), 0.0, places=6)

    def test_the_head_is_taken_from_zero(self) -> None:
        faded = splice.fade_in(self.steady(0.5), SAMPLE_RATE)
        self.assertAlmostEqual(float(faded[0]), 0.0, places=6)

    def test_fading_out_after_truncation_still_lands_on_zero(self) -> None:
        # The ordering the bug was about: cut first, then fade what remains.
        cut = splice.fade_in(self.steady(0.5), SAMPLE_RATE)[: int(SAMPLE_RATE * 0.3)]
        self.assertGreater(abs(float(cut[-1])), 0.01)
        self.assertAlmostEqual(float(splice.fade_out(cut, SAMPLE_RATE)[-1]), 0.0, places=6)

    def test_the_fade_out_is_longer_than_the_click_guard_at_the_head(self) -> None:
        self.assertGreater(splice.FADE_OUT_SEC, splice.FADE_IN_SEC)

    def test_the_body_of_the_clip_is_left_alone(self) -> None:
        signal = self.steady(0.5)
        faded = splice.fade_out(splice.fade_in(signal, SAMPLE_RATE), SAMPLE_RATE)
        head = int(SAMPLE_RATE * splice.FADE_IN_SEC)
        tail = int(SAMPLE_RATE * splice.FADE_OUT_SEC)
        np.testing.assert_allclose(faded[head:-tail], signal[head:-tail])

    def test_a_clip_shorter_than_the_fade_still_works(self) -> None:
        tiny = self.steady(0.01)
        faded = splice.fade_out(tiny, SAMPLE_RATE)
        self.assertEqual(faded.size, tiny.size)
        self.assertAlmostEqual(float(faded[-1]), 0.0, places=6)

    def test_a_truncated_splice_ends_quietly(self) -> None:
        # End to end: a clip that runs past the end of the dialogue.
        stereo = np.zeros((2, SAMPLE_RATE * 4))
        rows = [["ええ", [3.8, 3.9], "SPEAKER_MAIN"]]
        out, _rows, swaps = splice.splice_dialogue(
            stereo, SAMPLE_RATE, rows, [clip(0.9)], default_args(),
            random.Random(0), None,
        )
        self.assertTrue(swaps[0]["truncated"])
        self.assertLess(abs(float(out[0, -1])), 1e-6)


if __name__ == "__main__":
    unittest.main()
