from __future__ import annotations

import json
import unittest
from pathlib import Path
from random import Random

from scripts.generate_synthetic_moshi_training_data import (
    DialogueTurn,
    LISTENING_DIALOGUE_EXEMPLARS,
    build_aizuchi_agent_prompt,
    build_judge_agent_prompt,
    build_llm_prompt,
    build_moshi_agent_prompt,
    build_user_agent_profile,
    build_user_agent_prompt,
    clean_agent_utterance,
    fallback_moshi_utterance,
    messages_to_completion_prompt,
    openai_generation_url,
    openai_models_url,
    parse_aizuchi_insertions,
    split_text_into_clauses,
    user_turns_with_aizuchi,
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

    def test_multi_agent_prompts_keep_randomness_on_user_side(self) -> None:
        use_case = {
            "id": "case",
            "category": "loneliness_light",
            "risk_level": "low",
            "situation": "休日の孤独感",
            "user_profile": "20代後半、一人暮らし",
            "opening": "休日が少しつらいです。",
            "talking_points": ["SNS", "夕方", "予定がない"],
        }
        profile = build_user_agent_profile(use_case, Random(0))

        user_prompt = build_user_agent_prompt(use_case, profile, [], 0, 5)
        self.assertIn("今回の userAI のランダム設定", user_prompt.user)
        self.assertIn(profile["opening_style"], user_prompt.user)
        self.assertIn("出力は user の次の発話本文だけ", user_prompt.system)

        turns = [DialogueTurn("user", "休日になると、誰とも話さないまま夕方になります。")]
        moshi_prompt = build_moshi_agent_prompt(use_case, turns)
        self.assertIn("孤独・孤立相談窓口の相談員 moshi", moshi_prompt.system)
        self.assertIn("「うん」「うんうん」は使いません", moshi_prompt.user)
        self.assertNotIn("model_backchannel", moshi_prompt.user)
        self.assertNotIn("今回の userAI のランダム設定", moshi_prompt.user)

        judge_prompt = build_judge_agent_prompt(
            use_case, turns + [DialogueTurn("moshi", "夕方がつらいんですね。")], 2, 3, 5
        )
        self.assertIn('{"complete": true/false', judge_prompt.system)

        aizuchi_prompt = build_aizuchi_agent_prompt(
            use_case,
            [],
            turns[0].text,
            "夕方がつらいんですね。",
            1,
        )
        self.assertIn('"insertions"', aizuchi_prompt.system)
        self.assertIn("今回の moshi 本応答", aizuchi_prompt.user)

    def test_aizuchi_insertions_split_user_turns_into_current_format(self) -> None:
        text = "休日になると、誰とも話さないまま夕方になって、急に寂しくなります。"
        clauses = split_text_into_clauses(text)
        insertions = parse_aizuchi_insertions(
            '{"insertions":[{"after_clause":1,"text":"はい。"}]}',
            clauses,
            max_insertions=1,
        )
        turns = user_turns_with_aizuchi(text, insertions)

        self.assertEqual([turn.speaker for turn in turns], ["user", "moshi", "user"])
        self.assertEqual(turns[1].text, "はい。")
        self.assertEqual(turns[1].timing, "sequential")
        self.assertIsNone(turns[1].event)

    def test_openai_endpoint_urls_support_chat_and_completions(self) -> None:
        self.assertEqual(
            openai_generation_url("http://127.0.0.1:8080/v1", "chat"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            openai_generation_url("http://127.0.0.1:8080/v1", "completions"),
            "http://127.0.0.1:8080/v1/completions",
        )
        self.assertEqual(
            openai_models_url("http://127.0.0.1:8080/v1/completions"),
            "http://127.0.0.1:8080/v1/models",
        )

    def test_messages_to_completion_prompt_uses_chatml_shape(self) -> None:
        prompt = messages_to_completion_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertIn("<|im_start|>system\nsys<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nhello<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_clean_agent_utterance_removes_completion_special_tokens(self) -> None:
        self.assertEqual(clean_agent_utterance("<|im_end|>", "moshi"), "")
        self.assertEqual(
            clean_agent_utterance("<|im_start|>assistant\nmoshi|はい。<|im_end|>", "moshi"),
            "はい。",
        )

    def test_fallback_moshi_utterance_is_nonempty_and_polite(self) -> None:
        text = fallback_moshi_utterance(
            [DialogueTurn("user", "夕方になると、急にひとりだなって感じます。")]
        )

        self.assertIn("話してくださってありがとうございます", text)
        self.assertNotIn("うん", text)


if __name__ == "__main__":
    unittest.main()
