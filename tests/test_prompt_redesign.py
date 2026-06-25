from __future__ import annotations

import json
import unittest
from pathlib import Path
from random import Random

from scripts.generate_synthetic_moshi_training_data import (
    LISTENING_DIALOGUE_EXEMPLARS,
    build_llm_prompt,
)


class PromptRedesignTests(unittest.TestCase):
    def test_prompt_is_compact_transcript_format(self) -> None:
        # Default: 1 few-shot exemplar, transcript (pipe) output format.
        prompt = build_llm_prompt({"id": "minimal"}, Random(0))

        self.assertIn("傾聴中心", prompt)
        self.assertIn("話者|発話", prompt)
        self.assertIn("silence|秒数", prompt)
        # The output format no longer has an emotion column.
        self.assertNotIn("話者|感情|発話", prompt)
        # Exactly one exemplar by default: the first exemplar's first turn text
        # is present, the second exemplar's is not.
        self.assertEqual(len(LISTENING_DIALOGUE_EXEMPLARS), 5)
        self.assertIn(LISTENING_DIALOGUE_EXEMPLARS[0]["turns"][0]["text"], prompt)
        self.assertNotIn(LISTENING_DIALOGUE_EXEMPLARS[1]["turns"][0]["text"], prompt)

    def test_fewshot_count_is_configurable(self) -> None:
        p0 = build_llm_prompt({"id": "m"}, Random(0), num_fewshot=0)
        p3 = build_llm_prompt({"id": "m"}, Random(0), num_fewshot=3)
        self.assertNotIn(LISTENING_DIALOGUE_EXEMPLARS[0]["turns"][0]["text"], p0)
        self.assertIn(LISTENING_DIALOGUE_EXEMPLARS[2]["turns"][0]["text"], p3)
        self.assertLess(len(p0), len(p3))

    def test_prompt_keeps_moshi_polite_and_listening_forward(self) -> None:
        prompt = build_llm_prompt({"id": "minimal"}, Random(0))

        self.assertIn("moshi は丁寧語で傾聴中心", prompt)
        self.assertIn("単独の「うん」「うんうん」は使わない", prompt)
        self.assertIn("相手の言葉の反映", prompt)
        self.assertIn("感情の言語化", prompt)

    def test_listening_exemplars_do_not_teach_moshi_casual_backchannels(self) -> None:
        path = Path("tests/fixtures/listening_dialogues.jsonl")
        casual = {"うん。", "うんうん。", "うん…。", "うんうん、"}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            for turn in row["turns"]:
                if turn["speaker"] == "moshi":
                    self.assertNotIn(
                        turn["text"],
                        casual,
                        f"{path}:{line_number} teaches casual moshi backchannel",
                    )


if __name__ == "__main__":
    unittest.main()
