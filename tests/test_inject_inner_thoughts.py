from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import inject_inner_thoughts as inj  # noqa: E402

try:
    import soundfile as _sf  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def make_payload(
    *,
    greeting_start: float,
    aizuchi_frequency_label: str | None,
) -> dict:
    """1 挨拶 + 1 相づちだけの、テストに足る最小の sidecar payload を作る。"""
    entries = [
        ["もしもし、こちら孤独孤立相談窓口になります。",
         [greeting_start, greeting_start + 2.0], inj.MOSHI_LABEL],
        ["最近、あまり眠れていなくて。", [greeting_start + 2.5, greeting_start + 5.0],
         "SPEAKER_USER"],
        ["はい。", [greeting_start + 5.0, greeting_start + 5.5], inj.MOSHI_LABEL],
    ]
    payload = {
        "alignments_utterance": entries,
        "metadata": {"dialogue": {}},
    }
    if aizuchi_frequency_label is not None:
        payload["metadata"]["dialogue"]["aizuchi_frequency_label"] = aizuchi_frequency_label
    return payload


class DensityItemsTest(unittest.TestCase):
    """density_items() -- the piece that decides WHAT to inject; the actual
    time placement is inject_one()'s job and is exercised separately below."""

    def test_a_label_produces_exactly_one_tag_at_turn_index_zero(self) -> None:
        payload = make_payload(greeting_start=1.0, aizuchi_frequency_label="density=0.75")
        items = inj.density_items(payload)
        self.assertEqual(items, [{"turn_index": 0, "text": "<相槌:density=0.75>"}])

    def test_no_label_yields_no_items(self) -> None:
        payload = make_payload(greeting_start=1.0, aizuchi_frequency_label=None)
        self.assertEqual(inj.density_items(payload), [])

    def test_empty_label_yields_no_items(self) -> None:
        payload = make_payload(greeting_start=1.0, aizuchi_frequency_label="")
        self.assertEqual(inj.density_items(payload), [])

    def test_rule_and_llm_labels_are_carried_through_unchanged(self) -> None:
        for label in ("eager", "llm", "reserved"):
            payload = make_payload(greeting_start=1.0, aizuchi_frequency_label=label)
            self.assertEqual(inj.density_items(payload), [{"turn_index": 0, "text": f"<相槌:{label}>"}])


class DensityInjectionEndToEndTest(unittest.TestCase):
    """main() with --from-aizuchi-density, against real files on disk."""

    def run_injector(self, data_dir: Path, out_dir: Path, **extra_args: str) -> str:
        import contextlib
        import io
        import unittest.mock as mock

        argv = [
            "inject_inner_thoughts.py",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
            "--from-aizuchi-density",
        ]
        for key, value in extra_args.items():
            argv.append(f"--{key.replace('_', '-')}")
            argv.append(value)
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
            rc = inj.main()
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_a_tag_is_injected_when_there_is_enough_lead_in_silence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            payload = make_payload(greeting_start=4.0, aizuchi_frequency_label="density=0.75")
            (data_dir / "sample_001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            output = self.run_injector(data_dir, out_dir)
            self.assertIn("density_labeled", output)

            written = json.loads((out_dir / "sample_001.json").read_text(encoding="utf-8"))
            entries = written["alignments_utterance"]
            tags = [e for e in entries if e[0] == "<相槌:density=0.75>"]
            self.assertEqual(len(tags), 1)
            # Placed before the greeting, not overlapping it.
            self.assertLessEqual(tags[0][1][1], 4.0)
            self.assertEqual(tags[0][2], inj.MOSHI_LABEL)
            self.assertEqual(written["metadata"]["inner_thoughts"]["source"], "aizuchi_density")

    def test_no_lead_in_silence_drops_the_tag_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            # Greeting starts at t=0.0: there is no room before it at all.
            payload = make_payload(greeting_start=0.0, aizuchi_frequency_label="density=0.75")
            (data_dir / "sample_001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            output = self.run_injector(data_dir, out_dir)
            self.assertIn("WARNING", output)
            self.assertIn("density_tag_dropped", output)
            self.assertFalse((out_dir / "sample_001.json").exists())

    def test_missing_label_is_not_reported_as_a_dropped_tag(self) -> None:
        # A dialogue with no aizuchi_frequency_label at all (e.g. multi-agent
        # mode) is simply not in scope for this tag, not a failure to warn
        # about -- density_tag_dropped must stay at the WARNING's numerator
        # only for samples that had a label to place.
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            payload = make_payload(greeting_start=0.0, aizuchi_frequency_label=None)
            (data_dir / "sample_001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            output = self.run_injector(data_dir, out_dir)
            self.assertNotIn("WARNING", output)


class LabelSurvivesTheTtsLoaderTest(unittest.TestCase):
    """The tag is only reachable if aizuchi_frequency_label survives from
    dialogues.jsonl into the sidecar the injector reads. generate_qwen3_tts_data
    (the TTS step this corpus's pipeline actually runs, via
    run_bank_stereo_test.sh) rebuilds each dialogue through an explicit
    whitelist, so an unlisted field is dropped silently -- which is exactly
    what happened until this was wired up."""

    def test_the_jsonl_loader_keeps_the_label(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import generate_qwen3_tts_data as q

        row = {
            "id": "case_a",
            "category": "c",
            "risk_level": "low",
            "title": "t",
            "aizuchi_frequency_label": "density=0.75",
            "turns": [
                {"speaker": "moshi", "text": "もしもし。"},
                {"speaker": "user", "text": "はなします。"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dialogues.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = q.load_dialogues_from_jsonl(path)
        self.assertEqual(loaded[0]["aizuchi_frequency_label"], "density=0.75")

    def test_a_dialogue_without_the_label_loads_as_none(self) -> None:
        import generate_qwen3_tts_data as q

        row = {
            "id": "case_a", "category": "c", "risk_level": "low", "title": "t",
            "turns": [{"speaker": "user", "text": "はなします。"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dialogues.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = q.load_dialogues_from_jsonl(path)
        self.assertIsNone(loaded[0]["aizuchi_frequency_label"])


@unittest.skipUnless(HAS_SOUNDFILE, "soundfile not installed in this environment")
class PadPreservesAudioFormatTest(unittest.TestCase):
    """soundfile writes WAV as PCM_16 unless told otherwise, so padding a
    32-bit float or 24-bit corpus would silently requantize every sample."""

    def test_the_subtype_and_samples_survive_the_pad(self) -> None:
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            for subtype in ("PCM_16", "FLOAT", "PCM_24"):
                src = Path(tmp) / f"src_{subtype}.wav"
                dst = Path(tmp) / f"dst_{subtype}.wav"
                original = np.random.RandomState(0).uniform(-0.5, 0.5, (2400, 2))
                sf.write(str(src), original, 24000, subtype=subtype)

                inj.pad_lead_in_wav(src, dst, 1.0)

                self.assertEqual(sf.info(str(dst)).subtype, subtype, subtype)
                self.assertEqual(sf.info(str(dst)).samplerate, 24000, subtype)
                written, _ = sf.read(str(dst), always_2d=True)
                source, _ = sf.read(str(src), always_2d=True)
                self.assertEqual(len(written), len(source) + 24000, subtype)
                self.assertEqual(np.abs(written[:24000]).max(), 0.0, subtype)
                np.testing.assert_allclose(written[24000:], source, atol=1e-9)


class ShiftAlignmentsTest(unittest.TestCase):
    def test_every_entry_start_and_end_move_by_the_same_offset(self) -> None:
        entries = [["a", [0.0, 1.0], "SPEAKER_MAIN"], ["b", [1.5, 3.0], "SPEAKER_USER"]]
        shifted = inj.shift_alignments(entries, 2.0)
        self.assertEqual(shifted, [
            ["a", [2.0, 3.0], "SPEAKER_MAIN"],
            ["b", [3.5, 5.0], "SPEAKER_USER"],
        ])

    def test_zero_offset_is_a_no_op(self) -> None:
        entries = [["a", [0.3, 1.1], "SPEAKER_MAIN"]]
        self.assertEqual(inj.shift_alignments(entries, 0.0), entries)


@unittest.skipUnless(HAS_SOUNDFILE, "soundfile not installed in this environment")
class PadLeadInEndToEndTest(unittest.TestCase):
    """--pad-lead-in-sec: the fix for the case ShiftAlignmentsTest and
    DensityInjectionEndToEndTest.test_no_lead_in_silence_drops_the_tag_and_warns
    document -- a dialogue whose greeting starts at t=0 has nowhere to put the
    density tag, so instead of hoping for pre-existing silence, this pads the
    audio itself and shifts every timestamp, guaranteeing room."""

    def make_wav(self, path: Path, duration_sec: float, sample_rate: int = 24000) -> None:
        import numpy as np
        import soundfile as sf

        samples = np.zeros((int(duration_sec * sample_rate), 2), dtype="float32")
        sf.write(str(path), samples, sample_rate)

    def test_a_tag_that_would_otherwise_be_dropped_now_fits(self) -> None:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            # Greeting at t=0.0 -- no natural silence at all, the case that
            # dropped the tag in DensityInjectionEndToEndTest above.
            payload = make_payload(greeting_start=0.0, aizuchi_frequency_label="density=0.75")
            (data_dir / "sample_001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            self.make_wav(data_dir / "sample_001.wav", duration_sec=6.0)

            argv = [
                "inject_inner_thoughts.py",
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--from-aizuchi-density",
                "--pad-lead-in-sec", "3.0",
            ]
            import contextlib
            import io
            import unittest.mock as mock

            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                rc = inj.main()
            self.assertEqual(rc, 0)
            self.assertNotIn("WARNING", buf.getvalue())

            written = json.loads((out_dir / "sample_001.json").read_text(encoding="utf-8"))
            entries = written["alignments_utterance"]
            tags = [e for e in entries if e[0] == "<相槌:density=0.75>"]
            self.assertEqual(len(tags), 1)
            greeting = next(e for e in entries if e[2] == inj.MOSHI_LABEL and "もしもし" in e[0])
            # The greeting itself must have shifted by the pad amount.
            self.assertAlmostEqual(greeting[1][0], 3.0, places=2)

            data, sample_rate = sf.read(str(out_dir / "sample_001.wav"))
            self.assertAlmostEqual(len(data) / sample_rate, 9.0, places=1)


if __name__ == "__main__":
    unittest.main()
