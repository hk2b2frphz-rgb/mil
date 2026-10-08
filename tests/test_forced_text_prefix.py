from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from response_recorder import ForcedTextPrefix  # noqa: E402


class _Token:
    """Stands in for the tensor moshi hands to on_text_hook.

    The whole mechanism rests on that tensor being written IN PLACE: moshi
    calls the hook, then passes the same object to the depformer (which makes
    the audio) and writes it into the streaming cache. A hook that rebound a
    local name instead would change nothing.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def fill_(self, value: int) -> None:
        self.value = value


class ForcedTextPrefixTest(unittest.TestCase):
    def test_the_prefix_is_forced_then_the_model_speaks_for_itself(self) -> None:
        forced = ForcedTextPrefix([101, 102, 103])
        sampled = [_Token(999) for _ in range(5)]
        for token in sampled:
            forced(token)
        self.assertEqual([t.value for t in sampled], [101, 102, 103, 999, 999])
        self.assertTrue(forced.done)


if __name__ == "__main__":
    unittest.main()
