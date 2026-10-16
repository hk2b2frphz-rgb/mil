from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "patch_nu_dialogue_repo", REPO_ROOT / "scripts" / "patch_nu_dialogue_repo.py"
)
patcher = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(patcher)


class ResumeRunIdPatchTest(unittest.TestCase):
    """run_id を持たない checkpoint から再開できるようにするパッチ。

    これが無いと KeyError: 'run_id' で落ちる。checkpoint 自体は健全なのに
    tracking のメタデータが 1 つ足りないだけなので、学習を最初からやり直す
    のは高すぎる。
    """

    def write(self, line: str) -> Path:
        path = Path(tempfile.mkdtemp()) / "finetune.py"
        path.write_text(
            "def main(args, prev_config, resume):\n"
            "    if resume:\n"
            f"{line}"
            "    return args\n",
            encoding="utf-8",
        )
        return path

    def assert_patched(self, path: Path) -> str:
        src = path.read_text(encoding="utf-8")
        self.assertIn("AUTO_PATCH_RESUME_WITHOUT_RUN_ID", src)
        self.assertIn('.get("run_id")', src)
        self.assertNotIn('["run_id"]', src)
        self.assertNotIn("['run_id']", src)
        compile(src, str(path), "exec")
        return src

    def test_double_quoted_lookup(self) -> None:
        path = self.write('        args.run_id_to_resume = prev_config["run_id"]\n')
        self.assertTrue(patcher.patch_finetune_resume_run_id(path))
        self.assert_patched(path)

    def test_single_quoted_lookup(self) -> None:
        path = self.write("        args.run_id_to_resume = prev_config['run_id']\n")
        self.assertTrue(patcher.patch_finetune_resume_run_id(path))
        self.assert_patched(path)

    def test_other_variable_names(self) -> None:
        # 上流が変数名を変えても当たるように。
        path = self.write('        cfg.run_id_to_resume = saved_config["run_id"]\n')
        self.assertTrue(patcher.patch_finetune_resume_run_id(path))
        src = self.assert_patched(path)
        self.assertIn('cfg.run_id_to_resume = saved_config.get("run_id")', src)

    def test_extra_whitespace(self) -> None:
        path = self.write('        args.run_id_to_resume  =  prev_config["run_id"]  \n')
        self.assertTrue(patcher.patch_finetune_resume_run_id(path))
        self.assert_patched(path)

    def test_running_it_twice_changes_nothing(self) -> None:
        path = self.write('        args.run_id_to_resume = prev_config["run_id"]\n')
        self.assertTrue(patcher.patch_finetune_resume_run_id(path))
        before = path.read_text(encoding="utf-8")
        self.assertFalse(patcher.patch_finetune_resume_run_id(path))
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_it_refuses_rather_than_patching_nothing(self) -> None:
        # 当たらないまま「済んだ」ことにすると、学習が step 0 から始まってから
        # 気付くことになる。見つからないなら止まる。
        path = Path(tempfile.mkdtemp()) / "finetune.py"
        path.write_text("print('no resume logic here')\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            patcher.patch_finetune_resume_run_id(path)

    def test_the_patched_line_behaves(self) -> None:
        path = self.write('        args.run_id_to_resume = prev_config["run_id"]\n')
        patcher.patch_finetune_resume_run_id(path)
        namespace: dict = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)

        class Args:
            run_id_to_resume = "unset"

        # キーが無くても落ちず、None になる。
        args = namespace["main"](Args(), {}, True)
        self.assertIsNone(args.run_id_to_resume)
        # あれば従来どおり引き継ぐ。
        args = namespace["main"](Args(), {"run_id": "abc123"}, True)
        self.assertEqual(args.run_id_to_resume, "abc123")


if __name__ == "__main__":
    unittest.main()
