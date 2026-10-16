from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_nu_dialogue_repo as patcher
import select_best_checkpoint as selector


# Exercise the actual patched training/save blocks without loading a GPU model.
UPSTREAM_SHAPE = '''import argparse
import json
import math
import os
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

def _log_miltoka_metrics(split, step, epoch, metrics):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X updates steps.",
    )
    return parser.parse_args()

def main(args, accelerator, losses):
    current_steps = 0
    starting_epoch = 0
    for epoch in range(1):
        for current_steps in range(args.start_step + 1, args.max_train_steps + 1):
            if True:
                accelerator.weight = current_steps
                if current_steps in losses:
                    eval_metrics_for_log = {"loss/total": losses[current_steps]}
                    _log_miltoka_metrics("eval", current_steps, epoch, eval_metrics_for_log)
                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.save_state(output_dir)
    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.save_state(output_dir)
'''


class Accelerator:
    def __init__(self, *, main=True, fail=False):
        self.is_main_process = main
        self.fail = fail
        self.saved = []
        self.weight = None
        self.barriers = 0

    def wait_for_everyone(self):
        self.barriers += 1

    def save_state(self, path):
        if self.fail:
            raise OSError("checkpoint disk full")
        self.saved.append(self.weight)
        if self.is_main_process:
            tag = Path(path) / "pytorch_model"
            tag.mkdir(parents=True, exist_ok=True)
            (tag / "mp_rank_00_model_states.pt").write_text(str(self.weight))


class BestCheckpointPatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "finetune.py"
        self.source.write_text(UPSTREAM_SHAPE, encoding="utf-8")
        patcher.patch_finetune_keep_best_only(self.source)
        self.assertTrue(patcher.patch_finetune_save_best(self.source))
        self.namespace = {}
        exec(compile(self.source.read_text(encoding="utf-8"), str(self.source), "exec"), self.namespace)

    def train(self, losses, *, start=0, end=120, save_steps=58, keep=True, accelerator=None):
        args = SimpleNamespace(output_dir=str(self.root / "checkpoints"),
                               start_step=start, max_train_steps=end,
                               save_steps=save_steps, keep_best_only=keep)
        accelerator = accelerator or Accelerator()
        self.namespace["main"](args, accelerator, losses)
        return accelerator

    def test_best_between_periodic_saves_retains_the_evaluated_weights(self):
        accelerator = self.train({29: 2.0, 58: 1.5, 87: 0.8, 116: 1.2})
        self.assertEqual(accelerator.saved, [29, 58, 87, 116, 120])
        checkpoints = self.root / "checkpoints"
        self.assertEqual(sorted(p.name for p in checkpoints.iterdir()), ["step_120", "step_87"])
        self.assertEqual((checkpoints / "step_87/pytorch_model/mp_rank_00_model_states.pt").read_text(), "87")
        self.assertEqual(selector.load_fullft_checkpoint_metrics(checkpoints, "loss/total")[87], 0.8)

    def test_resume_preserves_the_previous_best_without_old_logs(self):
        self.train({29: 0.5, 58: 1.0}, end=60)
        accelerator = self.train({87: 0.8, 116: 0.7}, start=60)
        self.assertEqual(accelerator.saved, [116, 120])
        self.assertTrue((self.root / "checkpoints/step_29").is_dir())

    def test_resume_saves_a_new_best_between_periodic_saves(self):
        self.train({29: 0.5}, end=60)
        accelerator = self.train({87: 0.4, 116: 0.6}, start=60)
        self.assertEqual(accelerator.saved, [87, 116, 120])
        self.assertTrue((self.root / "checkpoints/step_87").is_dir())

    def test_no_duplicate_save_when_best_periodic_and_final_step_coincide(self):
        accelerator = self.train({29: 0.5}, end=29, save_steps=29)
        self.assertEqual(accelerator.saved, [29])

    def test_invalid_scores_do_not_create_best_checkpoints(self):
        accelerator = self.train({29: float("nan"), 58: float("inf"), 87: None}, save_steps=None)
        self.assertEqual(accelerator.saved, [120])

    def test_saving_improvements_does_not_require_pruning(self):
        accelerator = self.train({29: 0.5, 58: 0.7, 87: 0.4}, keep=False)
        self.assertIn(87, accelerator.saved)
        self.assertTrue((self.root / "checkpoints/step_29").is_dir())

    def test_all_ranks_save_but_only_main_writes_metrics(self):
        accelerator = self.train({29: 0.5}, end=29, accelerator=Accelerator(main=False))
        self.assertEqual(accelerator.saved, [29])
        self.assertGreaterEqual(accelerator.barriers, 2)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_failed_save_does_not_publish_checkpoint_metrics(self):
        with self.assertRaises(OSError):
            self.train({29: 0.5}, accelerator=Accelerator(fail=True))
        self.assertEqual(list(self.root.rglob("miltoka_eval_metrics.json")), [])

    def test_legacy_unscored_checkpoint_is_not_deleted_after_resume(self):
        legacy = self.root / "checkpoints/step_10/pytorch_model"
        legacy.mkdir(parents=True)
        (legacy / "mp_rank_00_model_states.pt").write_text("10")
        self.train({29: 0.5})
        self.assertTrue(legacy.is_dir())

    def test_patch_upgrades_existing_install_and_is_idempotent(self):
        before = self.source.read_text(encoding="utf-8")
        self.assertFalse(patcher.patch_finetune_save_best(self.source))
        self.assertEqual(self.source.read_text(encoding="utf-8"), before)

    def test_selector_uses_checkpoint_scores_over_conflicting_stdout(self):
        self.train({29: 0.5, 58: 1.0}, end=58, keep=False)
        log = self.root / "conflicting.log"
        log.write_text('MILTO_METRICS {"split":"eval","step":29,"metrics":{"loss/total":9.0}}\n',
                       encoding="utf-8")
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        out = StringIO()
        argv = ["select", "--mode", "fullft", "--log-file", str(log),
                "--checkpoints-dir", str(self.root / "checkpoints")]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            selector.main()
        self.assertEqual(json.loads(out.getvalue())["step"], 29)

    def test_selector_can_recover_best_without_stdout_log(self):
        self.train({29: 2.0, 58: 1.5, 87: 0.8, 116: 1.2})
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        out = StringIO()
        argv = ["select", "--mode", "fullft", "--log-file", str(self.root / "missing.log"),
                "--checkpoints-dir", str(self.root / "checkpoints")]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            selector.main()
        result = json.loads(out.getvalue())
        self.assertEqual(result["step"], 87)
        self.assertEqual(result["global_best_step"], 87)


if __name__ == "__main__":
    unittest.main()
