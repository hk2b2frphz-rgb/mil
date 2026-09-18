from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.resolve_clone_refs import resolve_from_clone_out_dir, select_references


class ResolveCloneRefsTests(unittest.TestCase):
    def test_clone_output_can_be_resolved_without_generator_module(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ref_dir = root / "ref00_in-context"
            ref_dir.mkdir()
            (ref_dir / "reference.wav").write_bytes(b"wav")
            (ref_dir / "manifest.json").write_text(
                '{"reference":{"wav":"missing.wav","text":"テスト音声",'
                '"timeline_index":3,"duration_sec":4.2}}',
                encoding="utf-8",
            )

            resolved = resolve_from_clone_out_dir(root, 0, "in-context")

            self.assertEqual(resolved["wav"], str(ref_dir / "reference.wav"))
            self.assertEqual(resolved["text"], "テスト音声")

    def test_reference_selection_prefers_clean_in_range_segment(self) -> None:
        segments = [
            {"speaker": "A", "start": 0, "end": 5, "text": "採用"},
            {"speaker": "A", "start": 0, "end": 8, "text": "重畳", "overlap_sec": 1},
            {"speaker": "A", "start": 0, "end": 9, "text": "相槌", "is_aizuchi": True},
        ]

        self.assertEqual(select_references(segments, "A", 3, 12, 1), [segments[0]])


if __name__ == "__main__":
    unittest.main()
