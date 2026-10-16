from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_kaburi_tts_data import (
    CHARS_PER_SEC,
    estimate_seconds,
    load_dialogues,
    split_in_half,
    split_turns_by_estimate,
)


def _turn(speaker: str, text: str) -> dict[str, str]:
    return {"speaker": speaker, "text": text, "tts_text": text}


class SplitTurnsByEstimateTest(unittest.TestCase):
    def test_keeps_short_dialogue_in_one_chunk(self) -> None:
        turns = [_turn("user", "あ" * 10), _turn("moshi", "うん")]
        chunks = split_turns_by_estimate(turns, chunk_sec=26.0, max_utts=64)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], turns)

    def test_splits_when_the_estimate_exceeds_the_canvas(self) -> None:
        # 80 chars ~= 11.1s each with its gap, so two fit in a 26s chunk and a
        # third does not.
        turns = [_turn("user", "あ" * 80) for _ in range(6)]
        chunks = split_turns_by_estimate(turns, chunk_sec=26.0, max_utts=64)
        self.assertEqual([len(c) for c in chunks], [2, 2, 2])
        self.assertEqual([t for c in chunks for t in c], turns)

    def test_respects_the_utterance_count_limit(self) -> None:
        turns = [_turn("moshi", "うん") for _ in range(10)]
        chunks = split_turns_by_estimate(turns, chunk_sec=26.0, max_utts=4)
        self.assertEqual([len(c) for c in chunks], [4, 4, 2])

    def test_an_over_long_single_utterance_stays_alone(self) -> None:
        long_turn = _turn("user", "あ" * 1000)
        turns = [long_turn, _turn("moshi", "うん")]
        chunks = split_turns_by_estimate(turns, chunk_sec=26.0, max_utts=64)
        self.assertEqual([len(c) for c in chunks], [1, 1])

    def test_estimate_counts_text_and_gaps(self) -> None:
        turns = [_turn("user", "あ" * 75), _turn("moshi", "")]
        self.assertAlmostEqual(estimate_seconds(turns), 75 / CHARS_PER_SEC + 0.8)


class SplitInHalfTest(unittest.TestCase):
    def test_splits_evenly_and_loses_nothing(self) -> None:
        turns = [_turn("user", str(i)) for i in range(5)]
        head, tail = split_in_half(turns)
        self.assertEqual(head + tail, turns)
        self.assertEqual(len(head), 2)

    def test_two_turns_split_into_singletons(self) -> None:
        turns = [_turn("user", "a"), _turn("moshi", "b")]
        head, tail = split_in_half(turns)
        self.assertEqual(len(head), 1)
        self.assertEqual(len(tail), 1)


class LoadDialoguesTest(unittest.TestCase):
    def _write(self, rows: list[dict]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.close()
        return Path(handle.name)

    def test_drops_silence_turns_and_keeps_tts_text(self) -> None:
        path = self._write([
            {
                "id": "d1",
                "title": "t",
                "turns": [
                    {"speaker": "user", "text": "今日は疲れた", "tts_text": "きょうはつかれた"},
                    {"speaker": "silence", "duration_sec": 3.0},
                    {"speaker": "moshi", "text": "そうなんですね"},
                    {"speaker": "other", "text": "無視される"},
                    {"speaker": "user", "text": "   "},
                ],
            }
        ])
        try:
            dialogues = load_dialogues(path)
        finally:
            path.unlink()
        self.assertEqual(len(dialogues), 1)
        turns = dialogues[0]["turns"]
        self.assertEqual([t["speaker"] for t in turns], ["user", "moshi"])
        self.assertEqual(turns[0]["text"], "今日は疲れた")
        self.assertEqual(turns[0]["tts_text"], "きょうはつかれた")
        # tts_text の無いターンは text をそのまま合成に回す。
        self.assertEqual(turns[1]["tts_text"], "そうなんですね")

    def test_skips_dialogues_with_no_renderable_turn(self) -> None:
        path = self._write([
            {"id": "empty", "turns": [{"speaker": "silence", "duration_sec": 2.0}]},
            {"id": "ok", "turns": [{"speaker": "user", "text": "はい"}]},
        ])
        try:
            dialogues = load_dialogues(path)
        finally:
            path.unlink()
        self.assertEqual([d["id"] for d in dialogues], ["ok"])


if __name__ == "__main__":
    unittest.main()
