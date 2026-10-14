"""背景拡張がカード・プロンプト・再開時の整合性に届くことを検証する。"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts import build_use_cases as cases
from scripts import generate_synthetic_moshi_training_data as gen
from scripts.use_case_situations import CATALOG_VERSION, SITUATION_GROUPS

ROOT = Path(__file__).resolve().parents[1]


class BackgroundCatalogTest(unittest.TestCase):
    def card_for(self, token):
        situation = next(row for row in cases.SITUATIONS if row[0] == token)
        # Exercise the actual sampler with this background, including all its
        # random attribute selection; don't construct an artificial card.
        with patch.object(cases, "SITUATIONS", [situation]):
            return cases.build_one(random.Random(7), 1)

    def test_catalog_ids_and_texts_are_unique_and_have_valid_profiles(self):
        self.assertGreaterEqual(len(cases.SITUATIONS), 400)
        self.assertEqual(len({row[0] for row in cases.SITUATIONS}), len(cases.SITUATIONS))
        self.assertEqual(len({row[1] for row in cases.SITUATIONS}), len(cases.SITUATIONS))
        for token, _, opening, _, _ in cases.SITUATIONS:
            self.assertIn(opening, cases.OPENING_TEMPLATES)
            for pool in cases.profile_pools(token):
                self.assertTrue(pool, token)
        self.assertLessEqual(set(cases.SITUATION_PROFILE_CONSTRAINTS), {row[0] for row in cases.SITUATIONS})

    def test_generation_reaches_every_background_and_remains_reproducible(self):
        def sample(seed):
            rng = random.Random(seed)
            return [cases.build_one(rng, index) for index in range(10000)]
        cards = sample(0)
        self.assertEqual(cards, sample(0))
        self.assertEqual({card.situation_id for card in cards}, {row[0] for row in cases.SITUATIONS})
        self.assertEqual(len({card.id for card in cards}), len(cards))
        for card in cards:
            if card.situation_kind != "distress":
                self.assertEqual(card.risk_level, "low")
                self.assertNotIn(card.conversation_type, {"venting", "seeking", "rambling"})
                self.assertNotIn(card.emotional_state, {"涙ぐんでいる", "ふさぎ込んでいる", "気分が沈んでいる"})

    def test_role_constraints_apply_to_new_backgrounds(self):
        for token, occupation in (("business_owner_no_peer", "経営者|自営業"),
                                  ("class_no_group", "学生"),
                                  ("pension_budget_adjustment", "退職した人")):
            card = self.card_for(token)
            self.assertRegex(card.user_profile, occupation)
        self.assertNotIn("退職した人", cases.profile_pools("work_mistake_rumination")[2])
        self.assertEqual(cases.profile_pools("grandchild_boundary")[0], ["50代", "60代", "70代"])

    def test_meta_background_keeps_its_motivation_in_prompt_and_profile(self):
        card = asdict(self.card_for("wondering_if_ai"))
        profile = gen.build_user_agent_profile(card, random.Random(0), listener_only=True)
        prompt = gen.build_aizuchi_only_user_prompt(
            card, profile, [], 0, 12,
            {"probe": False, "allow_pause": False, "utterance_length": "medium"},
        )
        self.assertEqual(card["topic"], card["situation"])
        self.assertIn(card["situation"], prompt.user)
        self.assertIn(card["situation_guidance"], prompt.user)
        self.assertIn("相手が説明したことにして", prompt.user)
        self.assertIn("相手の正体や仕組みを推測で確定しません", prompt.system)
        self.assertNotIn("途中で寂しさがにじむ", profile["opening_style"])
        self.assertNotIn("不安が少し和らぐ", profile["emotional_arc"])

    def test_positive_background_has_no_distress_opening(self):
        for token, _ in SITUATION_GROUPS["positive_sharing"]:
            card = self.card_for(token)
            self.assertEqual(card.opening_kind, "positive")
            self.assertIn(card.opening, cases.OPENING_TEMPLATES["positive"])

    def test_full_duplex_entrypoint_preserves_background_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.jsonl"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/build_full_duplex_training_use_cases.py"),
                "--out-path", str(output), "--num", "1000", "--seed", "0", "--listening-ratio", "1",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1000)
            self.assertTrue(all(row["situation_catalog_version"] == CATALOG_VERSION for row in rows))
            self.assertTrue(any(row["situation_kind"] == "meta" for row in rows))
            self.assertTrue(all(row["duplex_task"] is None for row in rows))

    def test_saved_dialogue_roundtrip_retains_background_metadata(self):
        data = dict(id="test", source_use_case="test", situation_id="wondering_if_ai",
                    situation_group="ai_curiosity", situation_kind="meta",
                    turns=[{"speaker": "user", "text": "声で話す感じが気になったんです。"}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogues.jsonl"
            path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
            dialogue = gen.load_dialogues_from_jsonl(path)[0]
            self.assertEqual(dialogue.situation_id, data["situation_id"])
            self.assertEqual(dialogue.situation_group, data["situation_group"])
            self.assertEqual(dialogue.situation_kind, data["situation_kind"])

    def test_resume_refuses_previous_catalog_without_overwriting_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["generator", "--out-dir", directory]):
                args = gen.parse_args()
            args.dialogue_generation_mode = "aizuchi-only"
            args.resume = True
            protocol = gen.continuous_listener_protocol(args)
            protocol.pop("situation_catalog_version")
            path = Path(directory) / "dialogues.jsonl"
            before = json.dumps({"id": "old", "turns": [], "listener_protocol": protocol}) + "\n"
            path.write_text(before, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                gen.write_dialogues_only(args)
            self.assertEqual(path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
