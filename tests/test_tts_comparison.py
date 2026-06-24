from __future__ import annotations

import importlib
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


if __name__ == "__main__":
    unittest.main()
