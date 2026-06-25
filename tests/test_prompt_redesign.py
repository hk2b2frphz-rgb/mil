from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
