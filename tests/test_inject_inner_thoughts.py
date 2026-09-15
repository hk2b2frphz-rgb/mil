from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import inject_inner_thoughts as inj  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
