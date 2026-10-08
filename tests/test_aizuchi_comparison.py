from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aizuchi_comparison as cmp


def load_dated(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/2026-10-07" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_dated("report_comparison")
bank = load_dated("check_bank")
VOCAB = ["はい。", "ええ。", "そうですか。", "そうなんですね。"]


def reaction(index, word="はい。"):
    return {"after_clause": index, "text": word}


def answer(items):
    return json.dumps({"reactions": items}, ensure_ascii=False)


def row(ident="case1", density=0.5):
    return {"id": ident, "aizuchi_frequency_label": f"density={density}", "turns": [
        {"speaker": "user", "text": "昨日は仕事がとても大変だったんですけど、"},
        {"speaker": "moshi", "text": "はい。", "event": "model_backchannel"},
        {"speaker": "user", "text": "今日は少し落ち着いたのでゆっくりと家で過ごしています。"},
        {"speaker": "moshi", "text": "はい。", "event": "model_backchannel"},
        {"speaker": "silence", "duration_sec": 3.0},
        {"speaker": "user", "text": "聞いていますか？"},
        {"speaker": "moshi", "text": "聞いていますよ。", "note": "確認への返答"},
    ]}


class PlacementTests(unittest.TestCase):
    def test_paired_text_silence_and_probe_are_preserved(self):
        source = row()
        original = deepcopy(source)
        calls = []
        def caller(prompt):
            calls.append(prompt)
            return answer([reaction(2, "ええ。")])
        out, trace = cmp.reposition(source, VOCAB, caller=caller)
        self.assertEqual(source, original)
        self.assertEqual(cmp.fixed_content(source), cmp.fixed_content(out))
        self.assertEqual(len(calls), 1)  # No placement request for the probe.
        self.assertEqual(sum(cmp.is_backchannel(t) for t in out["turns"]), 1)
        self.assertEqual(out["turns"][-3:], source["turns"][-3:])
        self.assertEqual(trace["sentences"][0]["cap"], 2)

    def test_zero_density_does_not_call_ai(self):
        with patch.object(cmp, "request_llm", side_effect=AssertionError("not allowed")):
            out, trace = cmp.reposition(row(density=0), VOCAB, caller=cmp.request_llm)
        self.assertFalse(any(cmp.is_backchannel(t) for t in out["turns"]))
        self.assertFalse(trace["sentences"])

    def test_density_caps_increase_without_random_sampling(self):
        parts = ["a", "b", "c", "d"]
        self.assertEqual([cmp.reaction_cap(d, parts, 3) for d in (0, .1, .5, 1)], [0, 1, 2, 3])
        self.assertEqual(cmp.reaction_cap(1, ["one"], 3), 1)

    def test_validation_rejects_duplicate_missing_end_bad_vocab_and_rapid_insertions(self):
        parts = ["短い、", "長い文の内容を十分に長く話している区間です、", "終わり。"]
        invalid = [
            [reaction(3), reaction(3, "ええ。")],
            [reaction(1)],
            [reaction(3, "助言します。")],
            [reaction(2), reaction(3, "ええ。")],
            [reaction(True)],
            [reaction(3.0)],
            [reaction(0)],
        ]
        for items in invalid:
            with self.subTest(items=items), self.assertRaises(ValueError):
                cmp.validate_reactions(answer(items), parts, VOCAB, 3, 24, [])

    def test_recent_two_phrase_repeat_is_rejected(self):
        for word in VOCAB[:2]:
            with self.assertRaises(ValueError):
                cmp.validate_reactions(answer([reaction(1, word)]), ["話。"], VOCAB, 1, 24, VOCAB[:2])

    def test_cross_sentence_gap_is_checked(self):
        parts = ["短い、", "話を十分に長く続けて自然な受け止めの区切りを待つのです。"]
        with self.assertRaises(ValueError):
            cmp.validate_reactions(answer([reaction(1), reaction(2, "ええ。")]), parts, VOCAB, 2, 24, [], True)
        self.assertEqual(len(cmp.validate_reactions(answer([reaction(2)]), parts, VOCAB, 2, 24, [], True)), 1)

    def test_short_adjacent_sentences_are_merged_with_exact_text(self):
        text = "つらいです。悲しいです。でも今日は少し落ち着きました。短いです。"
        units = list(cmp.sentence_units([{"speaker": "user", "text": text}]))
        self.assertEqual("".join(units), text)
        self.assertTrue(all(len(u) >= 24 for u in units))

    def test_retries_are_audited_and_no_random_fallback_is_used(self):
        replies = iter(["invalid", answer([reaction(1)]), answer([reaction(2)])])
        out, trace = cmp.reposition(row(), VOCAB, caller=lambda _: next(replies))
        self.assertEqual(len(trace["sentences"][0]["attempts"]), 3)
        self.assertIn("error", trace["sentences"][0]["attempts"][0])
        with self.assertRaisesRegex(ValueError, "failed three times"):
            cmp.reposition(row(), VOCAB, caller=lambda _: "bad json")

    def test_clauses_do_not_drop_whitespace_or_punctuation(self):
        for text in ["今日は、 休みです。", "句読点なし", "？！", " a,b!c?", "改行\nします。"]:
            self.assertEqual("".join(cmp.clauses(text)), text)

    def test_numbers_are_not_split_into_insertion_candidates(self):
        self.assertEqual(cmp.clauses("7.5時間で1,000件でした、疲れました。"), ["7.5時間で1,000件でした、", "疲れました。"])

    def test_no_future_sentence_in_prompt(self):
        text = "一つ目の文章は内容を十分に長く話して区切りを作ります。二つ目の文章も十分な長さで話してから終わります。"
        source = {"id": "x", "aizuchi_frequency_label": "density=0.1", "turns": [{"speaker": "user", "text": text}]}
        calls = []
        replies = iter([answer([reaction(1)]), answer([reaction(1, "ええ。")])])
        cmp.reposition(source, VOCAB, caller=lambda p: (calls.append(p), next(replies))[1])
        self.assertNotIn("二つ目", calls[0])
        self.assertIn("一つ目", calls[1])


class DataTests(unittest.TestCase):
    def test_freeze_drops_greeting_and_locks_source(self):
        import rewrite_aizuchi_vocab as rewrite
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root / "in.jsonl", root / "shared/dialogues.jsonl"
            rows = [row("a"), row("b")]
            rows[0]["turns"].insert(0, {"speaker": "moshi", "text": rewrite.AIZUCHI_ONLY_GREETING})
            cmp.write_rows(source, rows)
            cmp.freeze(source, out, 2)
            self.assertEqual(cmp.read_rows(out)[0]["turns"][0]["speaker"], "user")
            cmp.freeze(source, out, 2)
            rows[0]["turns"][1]["text"] = "ええ。"
            cmp.write_rows(source, rows)
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                cmp.freeze(source, out, 2)

    def test_freeze_refuses_missing_density_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
            for rows in [[row(), row()], [row("a"), {**row("b"), "aizuchi_frequency_label": "llm"}]]:
                cmp.write_rows(source, rows)
                with self.assertRaises(ValueError):
                    cmp.freeze(source, out, 2)

    def test_ai_resume_preserves_order_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out, vocab = root / "source.jsonl", root / "ai/dialogues.jsonl", root / "words.tsv"
            cmp.write_rows(source, [row("a"), row("b")])
            vocab.write_text("\n".join(VOCAB), encoding="utf-8")
            args = argparse.Namespace(source=source, out=out, vocab=vocab, concurrency=1, max_per_sentence=3, min_chars=24)
            with patch.object(cmp, "reposition", side_effect=lambda r, *a: ({**r}, {"id": r["id"]})) as mock:
                cmp.run_ai(args)
                cmp.run_ai(args)
                self.assertEqual(mock.call_count, 2)
            self.assertEqual([r["id"] for r in cmp.read_rows(out)], ["a", "b"])
            args.max_per_sentence = 2
            with self.assertRaises(ValueError):
                cmp.run_ai(args)

    def test_assemble_reorders_all_arms_and_refuses_a_missing_dialogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shared/dialogues.jsonl"
            cmp.write_rows(source, [row("a"), row("b")])
            for arm in cmp.ARMS:
                suffix = "placement_bank/shard_000_conditioned" if arm == "real_v2" else "merged_conditioned"
                training = root / arm / "tts" / suffix / "training_set"
                audio = training / "data_stereo"
                audio.mkdir(parents=True)
                records = []
                for ident in ["b", "a"]:
                    wav = audio / f"{arm}_{ident}.wav"
                    wav.write_bytes(b"audio")
                    cmp.write_json(wav.with_suffix(".json"), {"metadata": {"dialogue": {"id": ident}}})
                    records.append({"path": str(wav), "duration": 10})
                cmp.write_rows(training / "synthetic_moshi_train.jsonl", records)
            args = argparse.Namespace(source=source, root=root)
            cmp.assemble(args)
            for arm in cmp.ARMS:
                records = cmp.read_rows(root / arm / "paired/training_set/synthetic_moshi_train.jsonl")
                self.assertTrue(records[0]["path"].endswith(f"{arm}_a.wav"))
                self.assertTrue(records[1]["path"].endswith(f"{arm}_b.wav"))
            manifest = root / "real_v2/tts/placement_bank/shard_000_conditioned/training_set/synthetic_moshi_train.jsonl"
            cmp.write_rows(manifest, cmp.read_rows(manifest)[:1])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                cmp.assemble(args)

    def test_bank_refuses_missing_word_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, vocab = root / "samples.jsonl", root / "vocab.txt"
            vocab.write_text("はい。\nええ。\n", encoding="utf-8")
            (root / "a.wav").write_bytes(b"audio")
            cmp.write_rows(manifest, [{"text": "はい。", "wav": "a.wav"}])
            with self.assertRaisesRegex(ValueError, "lacks"):
                bank.check_bank(manifest, vocab)
            cmp.write_rows(manifest, [{"text": "はい。", "wav": "a.wav"}, {"text": "ええ。", "wav": "missing.wav"}])
            with self.assertRaisesRegex(ValueError, "Missing bank audio"):
                bank.check_bank(manifest, vocab)


class ReportTests(unittest.TestCase):
    def test_complete_report_has_three_model_columns_and_separate_teacher_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/run"
            cmp.write_rows(root / "shared/dialogues.jsonl", [row("a"), row("b")])
            cmp.write_rows(root / "ai_placement/dialogue/llm_dialogues/dialogues.jsonl", [row("a"), row("b")])
            vocab = repo / "scripts/2026-09-15/aizuchi_vocab_no_bare_un.tsv"
            vocab.parent.mkdir(parents=True)
            vocab.write_text("\n".join(VOCAB), encoding="utf-8")
            for arm in cmp.ARMS:
                folder = repo / "eval_runs/full_duplex" / f"exp_{arm}/benchmark_results"
                cmp.write_json(folder / "summary.json", {"evaluation": {"status": "complete"}, "n": 1,
                    "tasks": {"backchannel": {"means": {"backchannel_count": 2}}}})
                cmp.write_rows(folder / "per_case.jsonl", [{"task": "backchannel", "case_id": "same", "seed": 0,
                    "assistant_text": "はい。はい。"}])
            report.build_report(root, "exp", repo)
            result = json.loads((root / "comparison/results.json").read_text(encoding="utf-8"))
            self.assertEqual(set(result["metrics"]), set(cmp.ARMS))
            for metrics in result["metrics"].values():
                self.assertEqual(metrics["output_text_backchannel.consecutive_same_vocab_phrase"], 1)
                self.assertIn("training_corpus.backchannels", metrics)
            self.assertIn("real_v2,traditional_overlap,ai_placement", (root / "comparison/results.csv").read_text(encoding="utf-8-sig"))

    def test_composite_phrase_is_not_counted_as_two_repetitions(self):
        cases = [{"task": "backchannel", "assistant_text": "はい、はい。ええ。ええ。はい、はい。"}]
        stats = report.output_repeat_stats(cases, ["はい。", "はい、はい。", "ええ。"])
        self.assertEqual(stats["vocab_phrase_count"], 4)
        self.assertEqual(stats["consecutive_same_vocab_phrase"], 1)

    def test_report_refuses_partial_and_different_trial_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "data/run"
            for arm in cmp.ARMS:
                folder = repo / "eval_runs/full_duplex" / f"exp_{arm}/benchmark_results"
                cmp.write_json(folder / "summary.json", {"evaluation": {"status": "complete"}, "n": 1, "tasks": {}})
                cmp.write_rows(folder / "per_case.jsonl", [{"task": "backchannel", "case_id": arm, "seed": 0}])
            with self.assertRaisesRegex(ValueError, "IDs/seeds differ"):
                report.build_report(root, "exp", repo)
            cmp.write_json(repo / "eval_runs/full_duplex/exp_real_v2/benchmark_results/summary.json",
                           {"evaluation": {"status": "partial"}, "n": 1, "tasks": {}})
            with self.assertRaisesRegex(ValueError, "partial"):
                report.build_report(root, "exp", repo)


BASH = Path(r"C:\Program Files\Git\bin\bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")


@unittest.skipUnless(BASH.is_file(), "bash is unavailable")
class PBSTests(unittest.TestCase):
    def shell(self, root, command, **extra):
        env = dict(os.environ, COMPARE_N="20", COMPARE_ID="test_compare", PBS_O_WORKDIR=str(root), **extra)
        return subprocess.run([str(BASH), "-c", command], cwd=root, env=env, capture_output=True, text=True)

    def test_legacy_and_ai_render_use_the_same_direct_tts_without_bank_or_kaburi(self):
        for arm in ("traditional_overlap", "ai_placement"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / "scripts/2026-10-07", root / "scripts/2026-10-07")
                cmp.write_rows(root / "data/runs/test_compare/shared/dialogues.jsonl", [row()])
                cmp.write_rows(root / "data/runs/test_compare/ai_placement/dialogue/llm_dialogues/dialogues.jsonl", [row()])
                # No bank exists. Executing the old route must fail this test.
                (root / "scripts/run_bank_stereo_test.sh").write_text("exit 77\n", encoding="ascii")
                stub = root / "scripts/run_qwen_tts_vllm_3000_4gpu.pbs"
                stub.write_text('printf "%s\\n" "$DIALOGUES_JSONL" "$OUT_ROOT" "$SPARE_RATIO" "$QWEN_NO_OPENING_GREETING" "$QWEN_VOICE_MODE" "$REF_RANK" > routed.txt\n', encoding="ascii")
                result = self.shell(root, f"bash scripts/2026-10-07/render_{arm}.pbs", OUT_ROOT="wrong", FULL_RENDER="1")
                self.assertEqual(result.returncode, 0, result.stderr)
                routes = (root / "routed.txt").read_text().splitlines()
                expected = "/shared/dialogues.jsonl" if arm == "traditional_overlap" else "/ai_placement/dialogue/llm_dialogues/dialogues.jsonl"
                self.assertTrue(routes[0].endswith(expected))
                self.assertTrue(routes[1].endswith(f"/{arm}/tts"))
                self.assertEqual(routes[2:], ["0", "1", "mixed", "1"])

    def test_ai_condition_tags_each_tts_shard_then_merges_the_direct_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "scripts/2026-10-07", root / "scripts/2026-10-07")
            dated = root / "scripts/2026-09-15"
            dated.mkdir()
            (dated / "real_aizuchi_condition_10000_v2.pbs").write_text(
                'printf "%s\\t%s\\n" "$SRC_SHARD" "$DST_SHARD" >> conditioned.txt\n', encoding="ascii")
            command = '''
uv() { printf '%s\n' "$*" > merged.txt; }
export -f uv
bash scripts/2026-10-07/condition_ai_placement.pbs
'''
            result = self.shell(root, command, COMPARE_SHARDS="2")
            self.assertEqual(result.returncode, 0, result.stderr)
            paths = (root / "conditioned.txt").read_text().splitlines()
            self.assertEqual(len(paths), 2)
            for i, line in enumerate(paths):
                src, dst = line.split("\t")
                self.assertTrue(src.endswith(f"/ai_placement/tts/shard_{i:03d}"))
                self.assertTrue(dst.endswith(f"/ai_placement/tts/conditioned/shard_{i:03d}"))
            self.assertIn("/ai_placement/tts/merged_conditioned", (root / "merged.txt").read_text())
            self.assertNotIn("placement_bank", (root / "merged.txt").read_text())

    def test_old_ai_kaburi_data_cannot_be_reused_for_the_new_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "scripts/2026-10-07", root / "scripts/2026-10-07")
            cmp.write_rows(root / "data/runs/test_compare/shared/dialogues.jsonl", [row()])
            cmp.write_rows(root / "data/runs/test_compare/ai_placement/dialogue/llm_dialogues/dialogues.jsonl", [row()])
            (root / "data/runs/test_compare/ai_placement/tts/placement_bank").mkdir(parents=True)
            result = self.shell(root, "bash scripts/2026-10-07/render_ai_placement.pbs")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("use a new COMPARE_ID", result.stderr)

    def test_fullft_runner_honors_safe_checkpoint_and_refuses_a_foreign_directory(self):
        source = (ROOT / "scripts/run_nu_fullft_experiment.sh").read_text(encoding="utf-8")
        block = source[source.index('NU_RESUME="${NU_RESUME:-1}"'):source.index("# Early stopping on eval loss")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints/step_10").mkdir(parents=True)
            (root / "checkpoints/step_99").mkdir()
            (root / "foreign/step_10").mkdir(parents=True)
            (root / "resume.sh").write_text('set -euo pipefail\nNU_OUTPUT_DIR="$PWD/checkpoints"\nNU_RESUME_FLAG=--resume_from_checkpoint\nLAUNCH_CMD=()\n' + block
                                           + '\nprintf "%s\\n" "${LAUNCH_CMD[@]}"\n', encoding="utf-8")
            result = self.shell(root, 'export NU_RESUME_STEP_DIR="$PWD/checkpoints/step_10"; bash resume.sh')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("step_10", result.stdout)
            self.assertNotIn("step_99", result.stdout)
            result = self.shell(root, 'export NU_RESUME_STEP_DIR="$PWD/foreign/step_10"; bash resume.sh')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NU_RESUME_STEP_DIR", result.stderr)

    def test_train_chain_exports_an_early_stopped_run_and_only_continues_on_progress(self):
        for status in (0, 124, 2):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / "scripts/2026-10-07", root / "scripts/2026-10-07")
                cmp.write_json(root / "data/runs/test_compare/paired_summary.json", {"samples": 20})
                cmp.write_rows(root / "data/runs/test_compare/real_v2/paired/training_set/synthetic_moshi_train.jsonl", [{"path": "mock.wav"}])
                for name in ("setup_proxy.sh", "setup_pbs_distributed.sh"):
                    (root / "scripts" / name).write_text("true\n", encoding="ascii")
                (root / "scripts/train_timebox.sh").write_text(
                    f'timebox_init() {{ TIMEBOX_DEADLINE=0; }}\n'
                    f'timebox_run() {{ touch "$PWD/ran"; shift; "$@"; return {status}; }}\n'
                    'timebox_latest_fullft_ckpt() { if [[ -f "$PWD/ran" ]]; then printf "100\\t/mock/checkpoint\\n"; else return 1; fi; }\n', encoding="ascii")
                (root / "scripts/run_nu_fullft_experiment.sh").write_text(
                    'mkdir -p "experiments/$1"\nprintf "[early-stopping] stopping at step 3\\n" > "experiments/$1/run_nu_mock.log"\n'
                    'printf "%s\\n" "$HP_SEED" "$HP_LR" "$NU_RESUME" "$NU_OUTPUT_DIR" "$SKIP_AUTO_POSTPROCESS" > training_config.txt\n', encoding="ascii")
                functions = '''
qsub() { printf '%s\n' "$*" >> jobs_called.txt; echo 999.server; }
python3() { echo /mock/checkpoint; }
uv() {
    echo "$*" >> uv_calls.txt
    if [[ "$*" == *export_fullft_checkpoint.py* ]]; then
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == --out-dir ]]; then
                mkdir -p "$2"; echo weights > "$2/model.safetensors"; break
            fi
            shift
        done
    fi
}
export -f qsub python3 uv
bash scripts/2026-10-07/train_real_v2.pbs
'''
                result = self.shell(root, functions)
                if status == 2:
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse((root / "jobs_called.txt").exists())
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                call = (root / "jobs_called.txt").read_text()
                if status == 0:
                    self.assertIn("eval_real_v2.pbs", call)
                    self.assertNotIn("train_real_v2.pbs", call)
                    self.assertIn("select_best_checkpoint.py", (root / "uv_calls.txt").read_text())
                    self.assertTrue((root / "data/runs/test_compare/jobs/real_v2_model.txt").is_file())
                else:
                    self.assertIn("train_real_v2.pbs", call)
                    self.assertIn("CHAIN_INDEX=2", call)
                    self.assertNotIn("eval_real_v2.pbs", call)

    def test_all_dated_jobs_are_ascii_and_shell_syntax_is_valid(self):
        files = list((ROOT / "scripts/2026-10-07").glob("*.pbs"))
        self.assertEqual(len(files), 17)
        for file in files + list((ROOT / "scripts/2026-10-07").glob("*.sh")):
            if file.suffix == ".pbs":
                file.read_bytes().decode("ascii")
            result = subprocess.run([str(BASH), "-n", str(file)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_submission_graph_keeps_legacy_and_ai_independent_of_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "scripts/2026-10-07", root / "scripts/2026-10-07")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            qsub = bin_dir / "qsub"
            qsub.write_text('#!/usr/bin/env bash\nn=$(cat "$PWD/count" 2>/dev/null || echo 0)\nn=$((n+1))\necho "$n" > "$PWD/count"\necho "$*" >> "$PWD/calls"\necho "$n.server"\n', encoding="ascii")
            qsub.chmod(0o755)
            env = dict(os.environ, COMPARE_N="20", COMPARE_ID="test_compare", PBS_O_WORKDIR=str(root))
            # Override qsub using an exported Bash function; this also avoids
            # Windows PATH separator conversion for a temporary bin directory.
            command = 'qsub() { bash "$PWD/bin/qsub" "$@"; }; export -f qsub; bash scripts/2026-10-07/submit_comparison.sh'
            result = subprocess.run([str(BASH), "-c", command], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 13)
            legacy = next(c for c in calls if c.endswith("render_traditional_overlap.pbs"))
            self.assertIn("depend=afterok:1.server", legacy)
            self.assertNotIn("2.server", legacy)
            ai = next(c for c in calls if c.endswith("render_ai_placement.pbs"))
            self.assertIn("depend=afterok:1.server:3.server", ai)
            self.assertNotIn("2.server", ai)
            baseline = next(c for c in calls if c.endswith("render_real_v2.pbs"))
            self.assertIn("depend=afterok:1.server:2.server", baseline)
            pairing = next(c for c in calls if c.endswith("assemble_paired_data.pbs"))
            self.assertIn("depend=afterok:5.server:7.server:9.server", pairing)
            self.assertEqual(sum("depend=afterok:10.server" in c for c in calls), 3)
            self.assertFalse(any("eval_" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
