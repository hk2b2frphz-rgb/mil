import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.snapshot_fullft_checkpoint import snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "checkpoints" / "nu_run" / "step_120"
        self.tag = self.source / "pytorch_model"
        self.tag.mkdir(parents=True)
        (self.tag / "mp_rank_00_model_states.pt").write_bytes(b"model")
        (self.tag / "zero_pp_rank_0_optim_states.pt").write_bytes(b"optimizer")
        self.destination = self.root / "copy" / "step_120"
        self.output = self.root / "exported"

    def copy(self):
        snapshot(self.source, self.destination, self.output, stable_seconds=0)

    def test_copy_is_independent_and_source_unchanged(self):
        original = {p: p.read_bytes() for p in self.tag.iterdir()}
        self.copy()
        for path, content in original.items():
            copied = self.destination / path.relative_to(self.source)
            self.assertEqual(copied.read_bytes(), content)
            copied.write_bytes(b"changed by export")
            self.assertEqual(path.read_bytes(), content)

    def test_training_tree_and_ancestor_destinations_rejected(self):
        for target in (self.source / "copy", self.source.parent / "step_121", self.root):
            with self.subTest(target=target), self.assertRaises(ValueError):
                snapshot(self.source, target, self.output, stable_seconds=0)

    def test_output_inside_training_tree_rejected(self):
        self.output = self.source.parent / "exported"
        with self.assertRaises(ValueError):
            self.copy()
        self.assertFalse(self.destination.exists())

    def test_existing_destination_is_not_overwritten(self):
        self.destination.mkdir(parents=True)
        marker = self.destination / "keep"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            self.copy()
        self.assertEqual(marker.read_text(), "keep")

    def test_missing_shard_rejected(self):
        (self.tag / "zero_pp_rank_0_optim_states.pt").unlink()
        with self.assertRaises(ValueError):
            self.copy()
        self.assertFalse(self.destination.exists())

    def test_writing_checkpoint_rejected_before_copy(self):
        with patch("scripts.snapshot_fullft_checkpoint.time.sleep",
                   side_effect=lambda _: (self.tag / "new_rank.pt").write_bytes(b"new")):
            with self.assertRaisesRegex(ValueError, "changing"):
                self.copy()
        self.assertFalse(self.destination.exists())

    def test_change_during_copy_prevents_export(self):
        import shutil
        real_copy = shutil.copytree

        def changing_copy(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(src) == self.source:
                (self.tag / "mp_rank_00_model_states.pt").write_bytes(b"new model")
            return result

        with patch("scripts.snapshot_fullft_checkpoint.shutil.copytree", side_effect=changing_copy):
            with self.assertRaisesRegex(ValueError, "during copy"):
                self.copy()


if __name__ == "__main__":
    unittest.main()
