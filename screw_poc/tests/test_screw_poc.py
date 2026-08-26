from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from screw_poc.scripts.build_training_subsets import build_subset
from screw_poc.scripts.generate_dialogues import (
    DEFAULT_KNOWLEDGE,
    DEFAULT_POLICY,
    generate_evaluation,
    generate_train,
    load_knowledge,
    load_policy,
    validate_inputs,
    write_jsonl,
)
from screw_poc.scripts.score_predictions import expected_texts, score
from screw_poc.scripts.validate_dataset import read_jsonl


class ScrewPocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(DEFAULT_POLICY)
        cls.knowledge = load_knowledge(DEFAULT_KNOWLEDGE)
        validate_inputs(cls.policy, cls.knowledge)
        cls.train = generate_train(
            cls.knowledge, cls.policy, random.Random(cls.policy["seed"]), 1000
        )
        cls.evaluation = generate_evaluation(
            cls.knowledge, cls.policy, random.Random(cls.policy["seed"] + 1), 200
        )

    def test_knowledge_and_voice_contract(self) -> None:
        self.assertEqual(len(self.knowledge), 30)
        self.assertEqual(len({row.knowledge_id for row in self.knowledge}), 30)
        self.assertEqual(self.policy["voices"]["system"], "Ono_Anna")
        self.assertNotIn("Ono_Anna", self.policy["voices"]["users"])

    def test_canonical_dataset_counts_and_coverage(self) -> None:
        self.assertEqual(len(self.train), 1000)
        self.assertEqual(len(self.evaluation), 200)
        self.assertEqual(
            len({row["knowledge_id"] for row in self.train if row["knowledge_id"]}),
            30,
        )
        self.assertEqual(Counter(row["flow"] for row in self.train)["out_of_scope"], 100)
        self.assertFalse(any(row.get("duplex_task") for row in self.train))

    def test_system_technical_answers_are_locked_to_knowledge_db(self) -> None:
        answers = {row.knowledge_id: row.final_text for row in self.knowledge}
        for dialogue in self.train + self.evaluation:
            knowledge_id = dialogue.get("knowledge_id")
            if knowledge_id:
                self.assertEqual(dialogue["expected_answer"], answers[knowledge_id])
            for turn in dialogue["turns"]:
                if turn.get("response_pattern") in {"ANSWER", "STOP"}:
                    self.assertEqual(turn["text"], answers[knowledge_id])

    def test_unresolved_flow_uses_three_question_levels_then_escalates(self) -> None:
        for dialogue in (row for row in self.train if row["flow"] == "unresolved"):
            patterns = dialogue["expected_response_patterns"]
            asks = [pattern for pattern in patterns if pattern.startswith("ASK_")]
            self.assertEqual(len(asks), 3)
            self.assertTrue(asks[0].endswith("_L1"))
            self.assertTrue(asks[1].endswith("_L2"))
            self.assertTrue(asks[2].endswith("_L3"))
            self.assertEqual(patterns[-1], "ESCALATE")

    def test_all_dialogues_are_strictly_sequential(self) -> None:
        for dialogue in self.train + self.evaluation:
            with self.subTest(dialogue=dialogue["id"]):
                self.assertFalse(dialogue.get("duplex_task"))
                self.assertTrue(
                    all(
                        turn.get("timing", "sequential") == "sequential"
                        for turn in dialogue["turns"]
                    )
                )

    def test_comparison_subsets_keep_all_knowledge(self) -> None:
        for target in (100, 300, 1000):
            subset = build_subset(self.train, target)
            self.assertEqual(len(subset), target)
            self.assertEqual(
                len({row["knowledge_id"] for row in subset if row["knowledge_id"]}),
                30,
            )
        flows_100 = Counter(row["flow"] for row in build_subset(self.train, 100))
        self.assertEqual(flows_100["unresolved"], 30)

    def test_jsonl_round_trip_is_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogues.jsonl"
            write_jsonl(path, self.train[:2])
            self.assertEqual(read_jsonl(path), self.train[:2])

    def test_perfect_predictions_score_one(self) -> None:
        references = self.evaluation[:3]
        predictions = [
            {
                "id": row["id"],
                "response_texts": expected_texts(row),
                "response_patterns": row["expected_response_patterns"],
            }
            for row in references
        ]
        result = score(references, predictions)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["final_response_exact_accuracy"], 1.0)
        self.assertEqual(result["full_response_sequence_exact_accuracy"], 1.0)
        self.assertEqual(result["final_response_character_error_rate"], 0.0)
        self.assertEqual(result["response_pattern_sequence_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
