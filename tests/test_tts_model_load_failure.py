from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.generate_qwen3_tts_data import cache_mixed_role_stage


class ModelLoadFailureTest(unittest.TestCase):
    def run_stage(self, tts, cache):
        jobs = [SimpleNamespace(stem="a"), SimpleNamespace(stem="b")]
        request = SimpleNamespace(speaker_role="user")
        with patch(
            "scripts.generate_qwen3_tts_data.plan_render_job_requests",
            return_value=[request],
        ), patch(
            "scripts.generate_qwen3_tts_data._request_stage_role",
            return_value="user",
        ):
            return cache_mixed_role_stage(
                jobs, SimpleNamespace(dialogue_batch_size=16), tts, cache,
                "user", greeting_pcm=None,
            )

    def test_proxy_failure_aborts_without_retrying_each_dialogue(self):
        tts = Mock(sample_rate=24000)
        tts.load.side_effect = ConnectionError("proxy hostname cannot be resolved")
        cache = Mock()
        cache.load.return_value = None
        with self.assertRaisesRegex(ConnectionError, "proxy hostname"):
            self.run_stage(tts, cache)
        tts.load.assert_called_once_with()
        tts.synthesize_many.assert_not_called()
        cache.store.assert_not_called()

    def test_cached_audio_needs_no_model_load_or_network(self):
        tts = Mock(sample_rate=24000)
        cache = Mock()
        cache.load.return_value = (np.ones(10), 24000)
        self.assertEqual(self.run_stage(tts, cache), {})
        tts.load.assert_not_called()
        tts.synthesize_many.assert_not_called()

    def test_bad_audio_request_is_still_isolated(self):
        tts = Mock(sample_rate=24000)
        tts.synthesize_many.side_effect = [
            ValueError("bad batch"), ValueError("bad request"), [np.ones(10)],
        ]
        cache = Mock()
        cache.load.return_value = None
        with self.assertLogs(level="WARNING"):
            failures = self.run_stage(tts, cache)
        self.assertEqual(set(failures), {"a"})
        self.assertEqual(tts.synthesize_many.call_count, 3)
        cache.store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
