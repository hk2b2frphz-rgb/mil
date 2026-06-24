from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class TTSComparisonImportTests(unittest.TestCase):
    def test_driver_imports_and_default_fixtures_validate(self) -> None:
        module = importlib.import_module("scripts.run_tts_comparison")
        paths = [
            Path("tests/fixtures/listening_dialogues.jsonl"),
            Path("tests/fixtures/aizuchi_dialogues.jsonl"),
        ]
        dialogues = module.load_and_validate_dialogues(paths)
        self.assertTrue(dialogues)
        self.assertTrue(all(isinstance(path, Path) for path in paths))

    def test_backends_render_single_taste_condition(self) -> None:
        module = importlib.import_module("scripts.run_tts_comparison")
        requested = ["on", "off", "mild"]
        for backend in ("cosyvoice3", "qwen3", "kokoro"):
            self.assertEqual(module.conditions_for_backend(backend, requested), ["taste"])

    def test_index_tolerates_legacy_sidecars_without_voice_mode(self) -> None:
        module = importlib.import_module("scripts.run_tts_comparison")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            dialogue_dir = out_dir / "legacy_case"
            dialogue_dir.mkdir()
            wav_path = dialogue_dir / "qwen3__on.wav"
            wav_path.write_bytes(b"")
            (dialogue_dir / "qwen3__on.json").write_text(
                json.dumps(
                    {
                        "dialogue_id": "legacy_case",
                        "title": "legacy",
                        "backend": "qwen3",
                        "emotion_condition": "on",
                    }
                ),
                encoding="utf-8",
            )
            module.write_indexes(out_dir)
            self.assertIn("fixed preset voices", (out_dir / "INDEX.html").read_text())


if __name__ == "__main__":
    unittest.main()
