from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prune_fullft_checkpoints as pruner  # noqa: E402


def metrics_line(step: int, loss: float) -> str:
    payload = {"split": "eval", "step": step, "metrics": {"loss/total": loss}}
    return "MILTO_METRICS " + json.dumps(payload)


class PrunerTest(unittest.TestCase):
    def build(self, steps: dict[int, float | None], *, aged: bool = True) -> tuple[Path, Path]:
        """step -> eval loss（None は eval をまだ通っていない step）。"""
        root = Path(tempfile.mkdtemp())
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        lines = []
        for step, loss in steps.items():
            tag = checkpoints / f"step_{step}" / "pytorch_model"
            tag.mkdir(parents=True)
            (tag / "shard.bin").write_bytes(b"0" * 64)
            if loss is not None:
                lines.append(metrics_line(step, loss))
        log = root / "run_nu_1.log"
        log.write_text("\n".join(lines), encoding="utf-8")
        if aged:
            old = time.time() - 3600
            for path in checkpoints.glob("step_*"):
                os.utime(path, (old, old))
        return checkpoints, log

    def run_pruner(self, checkpoints: Path, logs: list[Path], keep: int = 3) -> list[int]:
        argv = ["prune", "--checkpoints-dir", str(checkpoints), "--keep", str(keep),
                "--log-file", *[str(p) for p in logs]]
        old_argv = sys.argv
        sys.argv = argv
        try:
            self.assertEqual(pruner.main(), 0)
        finally:
            sys.argv = old_argv
        return sorted(int(p.name.split("_")[-1]) for p in checkpoints.glob("step_*"))

    def test_keeps_the_best_k_and_the_newest(self) -> None:
        checkpoints, log = self.build(
            {120: 2.0, 240: 1.5, 360: 1.8, 480: 1.2, 600: 1.9, 720: None}
        )
        # 720 は eval 前だが、次のジョブが再開する先なので残る。
        self.assertEqual(self.run_pruner(checkpoints, [log]), [240, 360, 480, 720])

    def test_the_newest_survives_even_with_the_worst_loss(self) -> None:
        checkpoints, log = self.build({120: 1.0, 240: 1.1, 360: 1.2, 480: 9.9})
        self.assertIn(480, self.run_pruner(checkpoints, [log], keep=2))

    def test_nothing_is_removed_when_no_metric_can_be_read(self) -> None:
        # ログの書式が変わった / 最初の eval がまだ、のどちらでも起こる。ここで
        # 消すと最新以外を全部落とすことになるので、何もしないのが正しい。
        checkpoints, _log = self.build({120: 1.0, 240: 1.1, 360: 1.2})
        empty = checkpoints.parent / "empty.log"
        empty.write_text("nothing to see here", encoding="utf-8")
        self.assertEqual(self.run_pruner(checkpoints, [empty]), [120, 240, 360])

    def test_metrics_split_across_logs_are_all_read(self) -> None:
        # 再開して走らせるとログがジョブごとに分かれる。片方しか渡さないと、
        # もう片方にしか metric の無い step が消える。
        checkpoints, log_one = self.build({120: 1.0, 240: 1.1, 360: 5.0, 480: 5.1})
        log_two = checkpoints.parent / "run_nu_2.log"
        log_two.write_text("\n".join([metrics_line(360, 5.0), metrics_line(480, 5.1)]),
                           encoding="utf-8")
        log_one.write_text("\n".join([metrics_line(120, 1.0), metrics_line(240, 1.1)]),
                           encoding="utf-8")
        self.assertEqual(self.run_pruner(checkpoints, [log_one, log_two], keep=2),
                         [120, 240, 480])

    def test_a_directory_written_just_now_is_left_alone(self) -> None:
        # 書き出し途中の checkpoint を消さないための歯止め。
        checkpoints, log = self.build({120: 9.0, 240: 1.0, 360: 1.1, 480: 1.2},
                                      aged=False)
        self.assertEqual(self.run_pruner(checkpoints, [log], keep=2),
                         [120, 240, 360, 480])

    def test_a_directory_without_the_tag_is_left_alone(self) -> None:
        # 書き出しの途中か、そもそも別物。checkpoint として数えないし、消しも
        # しない。「最新」を決めるときの候補にも入らない。
        checkpoints, log = self.build({120: 1.0, 240: 1.1, 360: 9.0})
        (checkpoints / "step_999").mkdir()
        survivors = self.run_pruner(checkpoints, [log], keep=1)
        self.assertTrue((checkpoints / "step_999").is_dir())
        # 360 が最新として残り、120 が metric で残り、240 は落ちる。
        self.assertEqual(survivors, [120, 360, 999])

    def test_previous_best_with_missing_log_is_not_deleted(self) -> None:
        checkpoints, log = self.build({120: None, 240: 1.0, 360: 2.0})
        self.assertEqual(self.run_pruner(checkpoints, [log], keep=1), [120, 240, 360])

    def test_checkpoint_scores_survive_loss_of_previous_job_logs(self) -> None:
        checkpoints, log = self.build({120: None, 240: 1.0, 360: 2.0})
        (checkpoints / "step_120" / "miltoka_eval_metrics.json").write_text(
            json.dumps({"step": 120, "eval_loss_by_step": {"120": 0.5}}), encoding="utf-8"
        )
        self.assertEqual(self.run_pruner(checkpoints, [log], keep=1), [120, 360])

    def test_persisted_scores_override_another_runs_log(self) -> None:
        checkpoints, log = self.build({120: 9.0, 240: 1.0, 360: 2.0})
        (checkpoints / "step_120" / "miltoka_eval_metrics.json").write_text(
            json.dumps({"step": 120, "eval_loss_by_step": {"120": 0.5}}), encoding="utf-8"
        )
        self.assertEqual(self.run_pruner(checkpoints, [log], keep=1), [120, 360])

    def test_known_unevaluated_resume_checkpoint_can_be_pruned(self) -> None:
        checkpoints, log = self.build({120: None, 240: 1.0, 360: 2.0})
        (checkpoints / "step_120" / "miltoka_eval_metrics.json").write_text(
            json.dumps({"step": 120, "eval_loss_by_step": {}}), encoding="utf-8"
        )
        old = time.time() - 3600
        os.utime(checkpoints / "step_120", (old, old))
        self.assertEqual(self.run_pruner(checkpoints, [log], keep=1), [240, 360])


class KeepSetTest(unittest.TestCase):
    def test_everything_survives_when_there_are_fewer_than_k(self) -> None:
        steps = {10: Path("a"), 20: Path("b")}
        survivors = pruner.keep_set(steps, {10: 1.0, 20: 2.0}, keep=3)
        self.assertEqual(survivors, {10, 20})

    def test_ties_prefer_the_later_step(self) -> None:
        # 同じ loss なら、学習が進んでいる方を残す。
        steps = {10: Path("a"), 20: Path("b"), 30: Path("c")}
        survivors = pruner.keep_set(steps, {10: 1.0, 20: 1.0, 30: 2.0}, keep=1)
        self.assertEqual(survivors, {20, 30})


if __name__ == "__main__":
    unittest.main()
