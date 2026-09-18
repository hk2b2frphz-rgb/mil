from __future__ import annotations

import hashlib
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import qwen3_tts_diversity_probe as probe  # noqa: E402

SAMPLE_RATE = 24000


def tone(f0: float, seconds: float, amplitude: float = 0.3) -> np.ndarray:
    """A voiced-looking buzz: fundamental plus one harmonic."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * (np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(4 * np.pi * f0 * t))


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds))


def sample(text: str, signal: np.ndarray) -> dict:
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    return {
        "text": text,
        "audio_sha1": hashlib.sha1(pcm.tobytes()).hexdigest(),
        "measure": probe.measure(signal, SAMPLE_RATE),
        "features": probe.feature_sequence(signal, SAMPLE_RATE),
    }


class MeasureTest(unittest.TestCase):
    def test_pitch_and_length_are_recovered(self) -> None:
        result = probe.measure(tone(180.0, 0.5), SAMPLE_RATE)
        self.assertAlmostEqual(result["f0_median_hz"], 180.0, delta=4.0)
        self.assertAlmostEqual(result["duration_sec"], 0.5, delta=0.01)

    def test_leading_and_trailing_silence_is_not_counted_as_speech(self) -> None:
        padded = np.concatenate([silence(0.2), tone(180.0, 0.4), silence(0.3)])
        result = probe.measure(padded, SAMPLE_RATE)
        self.assertAlmostEqual(result["duration_sec"], 0.9, delta=0.01)
        self.assertAlmostEqual(result["speech_sec"], 0.4, delta=0.06)
        self.assertGreater(result["lead_silence_sec"], 0.1)
        self.assertGreater(result["trail_silence_sec"], 0.1)

    def test_a_gap_splits_the_voiced_runs(self) -> None:
        # This is what separates "un" from "un-un" in the report.
        one = probe.measure(tone(180.0, 0.4), SAMPLE_RATE)
        two = probe.measure(
            np.concatenate([tone(180.0, 0.3), silence(0.15), tone(170.0, 0.3)]),
            SAMPLE_RATE,
        )
        self.assertEqual(one["n_voiced_runs"], 1)
        self.assertEqual(two["n_voiced_runs"], 2)

    def test_silence_does_not_raise(self) -> None:
        result = probe.measure(silence(0.5), SAMPLE_RATE)
        self.assertEqual(result["speech_sec"], 0.0)
        self.assertEqual(result["f0_median_hz"], 0.0)
        self.assertEqual(result["n_voiced_runs"], 0)


class DistanceTest(unittest.TestCase):
    def test_a_sample_is_zero_distance_from_itself(self) -> None:
        features = probe.feature_sequence(tone(180.0, 0.5), SAMPLE_RATE)
        self.assertEqual(probe.dtw_distance(features, features), 0.0)

    def test_a_different_contour_is_further_than_a_similar_one(self) -> None:
        flat = probe.feature_sequence(tone(180.0, 0.5), SAMPLE_RATE)
        nearly_flat = probe.feature_sequence(tone(182.0, 0.5), SAMPLE_RATE)
        broken = probe.feature_sequence(
            np.concatenate([tone(180.0, 0.2), silence(0.2), tone(180.0, 0.2)]),
            SAMPLE_RATE,
        )
        self.assertLess(
            probe.dtw_distance(flat, nearly_flat), probe.dtw_distance(flat, broken)
        )

    def test_an_empty_sequence_is_distance_zero(self) -> None:
        features = probe.feature_sequence(tone(180.0, 0.3), SAMPLE_RATE)
        self.assertEqual(probe.dtw_distance(features, np.zeros((0, 2))), 0.0)


class OctaveGuardTest(unittest.TestCase):
    def test_a_sub_peak_near_the_reference_beats_a_stronger_octave_away(self) -> None:
        # The failure this guard exists for: one frame's best ACF peak sits an
        # octave off, and the group's F0 sd blows up (sokkaa read 66 Hz sd).
        self.assertEqual(
            probe.choose_candidate([(360.0, 0.90), (180.0, 0.80)], reference=180.0),
            180.0,
        )

    def test_a_clearly_better_candidate_still_wins(self) -> None:
        # The penalty is a tiebreaker, not an override: a real octave jump that
        # scores far higher must survive.
        self.assertEqual(
            probe.choose_candidate([(360.0, 0.95), (180.0, 0.40)], reference=180.0),
            360.0,
        )

    def test_without_a_reference_the_top_candidate_is_taken(self) -> None:
        self.assertEqual(
            probe.choose_candidate([(200.0, 0.9), (100.0, 0.8)], reference=0.0), 200.0
        )
        self.assertEqual(probe.choose_candidate([], reference=180.0), 0.0)

    def test_the_reference_ignores_a_minority_of_octave_errors(self) -> None:
        frames = [[(180.0, 0.9)]] * 8 + [[(360.0, 0.9)]] * 2
        self.assertEqual(probe.reference_f0(frames), 180.0)

    def test_a_steady_tone_is_not_moved_by_the_guard(self) -> None:
        f0 = probe.estimate_f0(tone(180.0, 0.6), SAMPLE_RATE)
        voiced = f0[f0 > 0]
        self.assertGreater(voiced.size, 10)
        self.assertLess(float(np.std(voiced)), 6.0)


class BeatTest(unittest.TestCase):
    def beats(self, signal: np.ndarray) -> int:
        return probe.energy_beats(probe.frame_rms(signal, SAMPLE_RATE))

    def test_one_continuous_sound_is_one_beat(self) -> None:
        self.assertEqual(self.beats(tone(180.0, 0.5)), 1)

    def test_two_sounds_split_by_a_dip_are_two_beats(self) -> None:
        # What "unun" should look like, and what voiced_runs cannot see when the
        # two halves are joined without an unvoiced gap.
        signal = np.concatenate(
            [tone(180.0, 0.25), tone(180.0, 0.08, amplitude=0.02), tone(170.0, 0.25)]
        )
        self.assertEqual(self.beats(signal), 2)
        self.assertEqual(
            probe.measure(signal, SAMPLE_RATE)["n_voiced_runs"], 1
        )

    def test_a_shallow_dip_is_not_a_beat(self) -> None:
        signal = np.concatenate(
            [tone(180.0, 0.25), tone(180.0, 0.08, amplitude=0.28), tone(170.0, 0.25)]
        )
        self.assertEqual(self.beats(signal), 1)

    def test_silence_has_no_beats(self) -> None:
        self.assertEqual(self.beats(silence(0.3)), 0)


class ReportTest(unittest.TestCase):
    def build(self, jitter: float) -> dict:
        rng = np.random.default_rng(0)
        groups = []
        for text, f0, seconds in (("un", 180.0, 0.4), ("uuun", 150.0, 0.9)):
            samples = [
                sample(
                    text,
                    tone(
                        f0 + rng.normal(0.0, jitter * 20.0),
                        seconds + rng.normal(0.0, jitter * 0.1),
                    ),
                )
                for _ in range(5)
            ]
            groups.append({"text": text, "temperature": None, "samples": samples})
        return probe.build_report(groups, {"run_id": "test"})

    def test_identical_outputs_are_counted_as_duplicates(self) -> None:
        # The headline failure mode: greedy decoding hands back the same audio
        # every time, and a bank of 200 is really a bank of 20.
        one = sample("un", tone(180.0, 0.4))
        report = probe.build_report(
            [{"text": "un", "temperature": None, "samples": [one, dict(one), dict(one)]}],
            {},
        )
        self.assertEqual(report["totals"]["samples"], 3)
        self.assertEqual(report["totals"]["unique_audio"], 1)
        self.assertEqual(report["totals"]["duplicate_audio"], 2)
        self.assertEqual(report["groups"][0]["within_dtw"], 0.0)

    def test_more_jitter_widens_the_within_text_spread(self) -> None:
        quiet = self.build(jitter=0.1)
        noisy = self.build(jitter=1.0)
        self.assertLess(
            quiet["totals"]["within_group_duration_cv_mean"],
            noisy["totals"]["within_group_duration_cv_mean"],
        )

    def test_separation_falls_as_the_within_text_spread_grows(self) -> None:
        # Both reports have the same two texts, so the between-text distance is
        # fixed; only the denominator moves.
        quiet = self.build(jitter=0.1)
        noisy = self.build(jitter=1.0)
        self.assertGreater(
            quiet["totals"]["duration_separation"],
            noisy["totals"]["duration_separation"],
        )

    def test_the_markdown_names_every_text(self) -> None:
        rendered = probe.format_report(self.build(jitter=0.5))
        self.assertIn("| un |", rendered)
        self.assertIn("| uuun |", rendered)
        self.assertIn("分離比", rendered)

    def spread_groups(self, durations_a, durations_b) -> dict:
        return probe.build_report(
            [
                {
                    "text": name,
                    "temperature": None,
                    "samples": [sample(name, tone(180.0, seconds)) for seconds in lengths],
                }
                for name, lengths in (("un", durations_a), ("uuun", durations_b))
            ],
            {},
        )

    def test_nearest_d_falls_below_one_once_the_groups_overlap(self) -> None:
        # d is the column that tells "the means are close" apart from "each
        # group is wide"; the overall ratio cannot. Same two means in both
        # reports, so only the within-group spread moves.
        tight = self.spread_groups([0.38, 0.40, 0.42], [0.53, 0.55, 0.57])
        wide = self.spread_groups([0.10, 0.40, 0.70], [0.25, 0.55, 0.85])
        self.assertGreater(tight["groups"][0]["nearest_d"], 1.0)
        self.assertLess(wide["groups"][0]["nearest_d"], 1.0)
        self.assertEqual(wide["totals"]["nearest_d_overlapping"], 2)

    def test_two_islands_are_read_as_accent_not_as_a_tracker_fault(self) -> None:
        # What "sokka" looks like: a geminate stop splits the voiced part in
        # two, the halves sit at different pitches, and the whole-utterance
        # median lands on whichever half happens to be longer. The voice is
        # steady; the summary statistic is not.
        low_first = np.concatenate(
            [tone(130.0, 0.30), silence(0.10), tone(200.0, 0.12)]
        )
        high_first = np.concatenate(
            [tone(130.0, 0.12), silence(0.10), tone(200.0, 0.30)]
        )
        samples = [sample("sokka", low_first), sample("sokka", high_first)]
        for entry in samples:
            self.assertEqual(entry["measure"]["n_voiced_runs"], 2)
            self.assertGreater(entry["measure"]["f0_step_semitones"], 3.0)
        report = probe.build_report(
            [{"text": "sokka", "temperature": None, "samples": samples}], {}
        )
        group = report["groups"][0]
        self.assertTrue(group["f0_suspect"])
        self.assertEqual(group["f0_suspect_reason"], "accent")
        # The first island is the stable read, which is the point of the
        # column: run order is fixed for a given word, so the head stays put
        # while the whole-utterance median flips between the islands.
        self.assertLess(group["f0_head_hz"]["sd"], group["f0_median_hz"]["sd"])

    def test_an_octave_split_is_named_as_such(self) -> None:
        samples = [sample("un", tone(f0, 0.4)) for f0 in (180.0, 180.0, 360.0)]
        report = probe.build_report(
            [{"text": "un", "temperature": None, "samples": samples}], {}
        )
        self.assertEqual(report["groups"][0]["f0_suspect_reason"], "octave")

    def test_a_wild_f0_is_flagged_rather_than_averaged_away(self) -> None:
        wild = [sample("sokkaa", tone(f0, 0.5)) for f0 in (90.0, 120.0, 260.0)]
        steady = [sample("un", tone(f0, 0.5)) for f0 in (178.0, 180.0, 182.0)]
        report = probe.build_report(
            [
                {"text": "sokkaa", "temperature": None, "samples": wild},
                {"text": "un", "temperature": None, "samples": steady},
            ],
            {},
        )
        flagged = report["totals"]["f0_suspect_groups"]
        self.assertEqual(len(flagged), 1)
        self.assertTrue(flagged[0].startswith("sokkaa("), flagged)
        self.assertIn("要確認", probe.format_report(report))


class OutlierTest(unittest.TestCase):
    def test_a_runaway_draw_is_split_out_of_the_spread(self) -> None:
        # The case that made a lower temperature look more diverse: one draw
        # five times the median length. cv counts it, cv* does not.
        lengths = [0.30, 0.32, 0.34, 0.30, 1.70]
        kept, long_tail, short_tail = probe.split_outliers(lengths)
        self.assertEqual(long_tail, [1.70])
        self.assertEqual(short_tail, [])
        self.assertEqual(len(kept), 4)

    def test_a_collapsed_draw_counts_too(self) -> None:
        kept, long_tail, short_tail = probe.split_outliers([0.30, 0.32, 0.34, 0.05])
        self.assertEqual(short_tail, [0.05])
        self.assertEqual(long_tail, [])
        self.assertEqual(len(kept), 3)

    def test_an_empty_group_does_not_raise(self) -> None:
        self.assertEqual(probe.split_outliers([]), ([], [], []))

    def test_the_trimmed_cv_undoes_a_tail_driven_spread(self) -> None:
        tailed = probe.build_report(
            [
                {
                    "text": "un",
                    "temperature": None,
                    "samples": [
                        sample("un", tone(180.0, seconds))
                        for seconds in (0.30, 0.32, 0.34, 0.30, 1.70)
                    ],
                }
            ],
            {},
        )
        group = tailed["groups"][0]
        self.assertEqual(group["outlier_long"], 1)
        self.assertEqual(tailed["totals"]["outliers"], 1)
        self.assertLess(group["speech_sec_trimmed"]["cv"], group["speech_sec"]["cv"] / 2)
        self.assertIn("外れ", probe.format_report(tailed))


class TemperatureTest(unittest.TestCase):
    def group(self, temperature: float, jitter: float) -> dict:
        rng = np.random.default_rng(int(temperature * 100))
        return {
            "text": "un",
            "temperature": temperature,
            "samples": [
                sample("un", tone(180.0, 0.4 + rng.normal(0.0, jitter)))
                for _ in range(5)
            ],
        }

    def test_groups_are_keyed_by_text_and_temperature(self) -> None:
        report = probe.build_report(
            [self.group(0.7, 0.01), self.group(1.3, 0.06)], {}
        )
        keys = [group["key"] for group in report["groups"]]
        self.assertEqual(keys, ["un @T0.7", "un @T1.3"])
        self.assertEqual(report["totals"]["texts"], 1)
        self.assertEqual(report["totals"]["groups"], 2)

    def test_the_effect_table_lines_the_temperatures_up(self) -> None:
        report = probe.build_report(
            [self.group(0.7, 0.01), self.group(1.3, 0.06)], {}
        )
        rows = report["temperature_effect"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["temperatures"], [0.7, 1.3])
        # Wider draws at the higher temperature: that is the whole question.
        self.assertLess(rows[0]["duration_cv"][0], rows[0]["duration_cv"][1])
        self.assertIn("temperature を振ったとき", probe.format_report(report))

    def test_one_temperature_alone_produces_no_effect_table(self) -> None:
        report = probe.build_report([self.group(0.7, 0.01)], {})
        self.assertEqual(report["temperature_effect"], [])

    def test_cross_temperature_pairs_are_not_counted_as_between_text(self) -> None:
        # Same word at two temperatures is not "two spellings"; counting it as
        # such would inflate the separation ratio.
        same = probe.build_report([self.group(0.7, 0.01), self.group(1.3, 0.01)], {})
        self.assertEqual(same["totals"]["between_text_dtw_mean"], 0.0)


class SamplesTest(unittest.TestCase):
    def test_records_group_by_text_and_temperature_in_first_seen_order(self) -> None:
        records = [
            {"text": "un", "temperature": 0.7},
            {"text": "sokka", "temperature": 0.7},
            {"text": "un", "temperature": 1.3},
            {"text": "un", "temperature": 0.7},
        ]
        groups = probe.group_samples(records)
        self.assertEqual(
            [(g["text"], g["temperature"], len(g["samples"])) for g in groups],
            [("un", 0.7, 2), ("sokka", 0.7, 1), ("un", 1.3, 1)],
        )

    def test_wav_round_trips_through_the_reader(self) -> None:
        import tempfile

        signal = tone(180.0, 0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.wav"
            probe.write_wav(path, signal, SAMPLE_RATE)
            restored, rate = probe.read_wav(path)
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertEqual(restored.size, signal.size)
            # 16bit quantization is the only loss.
            self.assertLess(float(np.max(np.abs(restored - signal))), 1e-3)

    def test_temperatures_parse(self) -> None:
        self.assertEqual(probe.parse_temperatures("0.7, 1.0,1.3"), [0.7, 1.0, 1.3])
        self.assertEqual(probe.parse_temperatures(""), [None])

    def test_listening_picks_span_the_extremes(self) -> None:
        samples = [
            {"measure": {"speech_sec": 0.9}, "wav": "long.wav"},
            {"measure": {"speech_sec": 0.2}, "wav": "short.wav"},
            {"measure": {"speech_sec": 0.5}, "wav": "mid.wav"},
        ]
        picks = probe.listening_picks(samples)
        self.assertEqual(picks["shortest"], "short.wav")
        self.assertEqual(picks["longest"], "long.wav")
        self.assertEqual(picks["median"], "mid.wav")


class PlanTest(unittest.TestCase):
    def test_the_default_texts_cover_the_real_recording_top_four(self) -> None:
        # From the 116 measured backchannels: these four were 76% of them, and
        # none of them is in the synthetic vocabulary the corpora use today.
        for text in ("うん", "うーん", "そっか", "うんうん"):
            self.assertIn(text, probe.DEFAULT_TEXTS)

    def test_texts_file_drops_blanks_and_comments(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "texts.txt"
            path.write_text("# note\n\nうん\n  そっか  \n", encoding="utf-8")
            self.assertEqual(probe.load_texts(path), ["うん", "そっか"])

    def test_write_wav_round_trips_and_digests_the_quantized_audio(self) -> None:
        import tempfile

        signal = tone(180.0, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.wav"
            digest = probe.write_wav(path, signal, SAMPLE_RATE)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), SAMPLE_RATE)
                self.assertEqual(handle.getnframes(), signal.size)
            # Same audio -> same digest, which is how duplicates are spotted.
            second = probe.write_wav(Path(directory) / "b.wav", signal, SAMPLE_RATE)
            self.assertEqual(digest, second)


if __name__ == "__main__":
    unittest.main()
