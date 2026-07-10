from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
for directory in (REPO_ROOT, EVAL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval.build_full_duplex_ja_dataset import render_scenario
from eval.local_baseline_common import GemmaLLM, validate_spoken_response, worker_environment
from eval.run_local_baseline_full_duplex import process_sample, seed_local
from scripts.gemma_dialogue_worker import result_to_text


class CascadeBaselineTests(unittest.TestCase):
    def test_result_to_text_selects_only_final_assistant_message(self) -> None:
        result = [
            {
                "generated_text": [
                    {"role": "user", "content": "利用者の発話"},
                    {"role": "assistant", "content": "相談員の応答"},
                ]
            }
        ]
        self.assertEqual(result_to_text(result), "相談員の応答")

    def test_background_speech_uses_distinct_speaker_and_clean_removes_it(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_synthesize(
            text: str,
            sample_rate: int,
            speed: float,
            tts_backend: str,
            tts_engine: object | None,
            speaker: str,
        ) -> np.ndarray:
            calls.append((text, speaker))
            return np.ones(10, dtype=np.float32)

        scenario = {
            "id": "background_test",
            "task": "background_speech",
            "paired": True,
            "timeline": [
                {"type": "speech", "text": "相談本文"},
                {
                    "type": "overlap_speech",
                    "text": "駅のアナウンス",
                    "event": "overlap",
                    "gain": 0.35,
                },
            ],
        }
        with patch("eval.build_full_duplex_ja_dataset.synthesize", fake_synthesize):
            noisy, clean, metadata = render_scenario(
                scenario,
                100,
                1.0,
                "qwen3",
                object(),
                "model",
                "Ono_Anna",
                tts_background_speaker="Uncle_Fu",
            )

        self.assertEqual(calls, [("相談本文", "Ono_Anna"), ("駅のアナウンス", "Uncle_Fu")])
        np.testing.assert_array_equal(noisy[:10], np.ones(10, dtype=np.float32))
        self.assertLess(float(np.max(np.abs(noisy[10:]))), 0.2)
        np.testing.assert_array_equal(clean[10:], np.zeros(10, dtype=np.float32))
        self.assertEqual(metadata["tts_background_speaker"], "Uncle_Fu")
        self.assertEqual(metadata["background_acoustic_profile"]["gain_db"], -15.0)

    def test_process_sample_runs_noisy_and_clean_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            source = dataset / "background_speech" / "case"
            source.mkdir(parents=True)
            (source / "clean_input.wav").touch()
            seen: list[str] = []
            with patch(
                "eval.run_local_baseline_full_duplex.process_variant",
                side_effect=lambda _sample, _seed, variant, *_args: seen.append(variant),
            ):
                process_sample(
                    {"id": "case", "task": "background_speech", "path": "background_speech/case"},
                    0,
                    dataset,
                    Path(tmp) / "out",
                    argparse.Namespace(),
                    "qwen3",
                    None,
                    None,
                    None,
                    None,
                    "cascade",
                )
            self.assertEqual(seen, ["input", "clean_input"])

    def test_seed_local_is_repeatable_for_numpy(self) -> None:
        seed_local(17)
        first = np.random.random(4)
        seed_local(17)
        np.testing.assert_array_equal(first, np.random.random(4))

    def test_spoken_response_rejects_prompt_leak(self) -> None:
        self.assertEqual(validate_spoken_response("相談員： お話しください"), "お話しください")
        with self.assertRaisesRegex(RuntimeError, "leaked user"):
            validate_spoken_response("利用者: 相談です\n応答です")

    def test_gemma_uses_persistent_seeded_worker_protocol(self) -> None:
        class RecordingStdin(io.StringIO):
            def close(self) -> None:
                self.snapshot = self.getvalue()
                super().close()

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = RecordingStdin()
                self.stdout = io.StringIO(
                    '{"ready": true}\n{"text": "応答本文"}\n'
                )

            def poll(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                return 0

        process = FakeProcess()
        with patch("eval.local_baseline_common.subprocess.Popen", return_value=process):
            llm = GemmaLLM(model="test-model", python_exe="python")
            llm.load()
            text, _wall = llm.respond("相談内容", seed=23)
            payload = json.loads(process.stdin.getvalue().splitlines()[0])
            llm.close()

        self.assertEqual(text, "応答本文")
        self.assertEqual(payload["seed"], 23)
        self.assertIn("相談内容", payload["prompt"])

    def test_worker_drops_malformed_proxy(self) -> None:
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:bad"}, clear=True):
            environment = worker_environment()
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertNotIn("HTTP_PROXY", environment)


if __name__ == "__main__":
    unittest.main()
