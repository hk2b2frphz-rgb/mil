from __future__ import annotations

import unittest
from random import Random

from scripts.generate_synthetic_moshi_training_data import (
    LISTENING_DIALOGUE_EXEMPLARS,
    build_gemma_prompt,
)


class PromptRedesignTests(unittest.TestCase):
    def test_prompt_contains_listening_redesign_and_all_exemplars(self) -> None:
        prompt = build_gemma_prompt({"id": "minimal"}, Random(0))

        self.assertIn("EMOTION MIRRORING", prompt)
        self.assertIn("user の悲しさ、不安、孤独", prompt)
        self.assertIn("BACKCHANNEL DIVERSITY", prompt)
        self.assertIn("同じ対話内で同じ相槌を繰り返さない", prompt)
        self.assertIn("GENTLE PROBING", prompt)
        self.assertIn("開かれた問いで穏やかに掘り下げ", prompt)
        self.assertEqual(len(LISTENING_DIALOGUE_EXEMPLARS), 5)
        for exemplar in LISTENING_DIALOGUE_EXEMPLARS:
            self.assertIn(exemplar["id"], prompt)


if __name__ == "__main__":
    unittest.main()
