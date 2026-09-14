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
            groups.append({"text": text, "samples": samples})
        return probe.build_report(groups, {"run_id": "test"})

    def test_identical_outputs_are_counted_as_duplicates(self) -> None:
        # The headline failure mode: greedy decoding hands back the same audio
        # every time, and a bank of 200 is really a bank of 20.
        one = sample("un", tone(180.0, 0.4))
        report = probe.build_report(
            [{"text": "un", "samples": [one, dict(one), dict(one)]}], {}
        )
        self.assertEqual(report["totals"]["samples"], 3)
        self.assertEqual(report["totals"]["unique_audio"], 1)
        self.assertEqual(report["totals"]["duplicate_audio"], 2)
        self.assertEqual(report["groups"][0]["within_dtw"], 0.0)

    def test_more_jitter_widens_the_within_text_spread(self) -> None:
        quiet = self.build(jitter=0.1)
        noisy = self.build(jitter=1.0)
        self.assertLess(
            quiet["totals"]["within_text_duration_cv_mean"],
            noisy["totals"]["within_text_duration_cv_mean"],
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
