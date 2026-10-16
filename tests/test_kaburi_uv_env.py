from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts/kaburi_uv_env.sh"

# Every function scripts/fix_kaburi_audio_io.sh, setup_kaburi_env.sh and
# run_kaburi_tts.pbs call after sourcing the helper. A missing one only shows
# up on the cluster, as "kaburi_tls_flag: command not found" mid-repair.
PUBLIC_FUNCTIONS = (
    "kaburi_tls_flag",
    "kaburi_uv_flags",
    "kaburi_uv_run",
    "kaburi_uv_sync_args",
    "kaburi_export_pythonpath",
    "kaburi_export_ldpath",
)


def run_bash(script: str, env: dict[str, str] | None = None) -> str:
    full = f'set -euo pipefail\nsource "{HELPER}"\n{script}'
    result = subprocess.run(
        ["bash", "-c", full],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp", **(env or {})},
    )
    if result.returncode != 0:
        raise AssertionError(
            f"bash failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


class HelperDefinitionsTest(unittest.TestCase):
    def test_every_public_function_is_defined(self) -> None:
        for name in PUBLIC_FUNCTIONS:
            with self.subTest(function=name):
                kind = run_bash(f"type -t {name}")
                self.assertEqual(kind, "function")


class UvInvocationTest(unittest.TestCase):
    def test_run_targets_the_checkout_and_keeps_the_repaired_env(self) -> None:
        out = run_bash(
            'KABURI_REPO=/tmp/kab kaburi_uv_run CMD; printf "%s" "${CMD[*]}"'
        )
        self.assertIn("--project /tmp/kab", out)
        self.assertIn("--with pyopenjtalk", out)
        # setup_kaburi_env.sh may repair the venv past the lockfile; an exact
        # re-sync on every run would undo that.
        self.assertIn("--no-sync", out)
        self.assertTrue(out.endswith("python"), out)

    def test_tls_flag_is_one_of_the_two_uv_spellings(self) -> None:
        flag = run_bash("kaburi_tls_flag")
        self.assertIn(flag, ("--system-certs", "--native-tls"))

    def test_tls_can_be_turned_off(self) -> None:
        out = run_bash(
            'KABURI_NATIVE_TLS=0 KABURI_REPO=/tmp/kab kaburi_uv_run CMD;'
            ' printf "%s" "${CMD[*]}"'
        )
        self.assertNotIn("--system-certs", out)
        self.assertNotIn("--native-tls", out)

    def test_pypi_torch_adds_no_sources(self) -> None:
        out = run_bash(
            'KABURI_TORCH_FROM_PYPI=1 KABURI_REPO=/tmp/kab kaburi_uv_run CMD;'
            ' printf "%s" "${CMD[*]}"'
        )
        self.assertIn("--no-sources", out)

    def test_sync_args_are_flags_only(self) -> None:
        out = run_bash('kaburi_uv_sync_args ARGS; printf "%s" "${ARGS[*]}"')
        self.assertNotIn("uv", out)
        for token in out.split():
            self.assertTrue(token.startswith("--"), token)

    def test_explicit_interpreter_wins(self) -> None:
        out = run_bash(
            'KABURI_PYTHON=/bin/sh KABURI_REPO=/tmp/kab kaburi_uv_run CMD;'
            ' printf "%s" "${CMD[*]}"'
        )
        self.assertEqual(out, "/bin/sh")

    def test_a_missing_explicit_interpreter_is_rejected(self) -> None:
        full = (
            f'source "{HELPER}"\n'
            'KABURI_PYTHON=/nope/python kaburi_uv_run CMD && echo UNEXPECTED'
        )
        result = subprocess.run(
            ["bash", "-c", full], capture_output=True, text=True, cwd=REPO_ROOT
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("UNEXPECTED", result.stdout)


class PathExportTest(unittest.TestCase):
    def test_pythonpath_carries_the_checkout_and_its_scripts(self) -> None:
        out = run_bash('KABURI_REPO=/tmp/kab kaburi_export_pythonpath; printf "%s" "$PYTHONPATH"')
        self.assertEqual(out, "/tmp/kab:/tmp/kab/scripts")

    def test_pythonpath_keeps_what_was_already_there(self) -> None:
        out = run_bash(
            'PYTHONPATH=/pre KABURI_REPO=/tmp/kab kaburi_export_pythonpath;'
            ' printf "%s" "$PYTHONPATH"'
        )
        self.assertEqual(out, "/tmp/kab:/tmp/kab/scripts:/pre")

    def test_pythonpath_does_not_stack_on_repeat(self) -> None:
        out = run_bash(
            'KABURI_REPO=/tmp/kab; kaburi_export_pythonpath; kaburi_export_pythonpath;'
            ' printf "%s" "$PYTHONPATH"'
        )
        self.assertEqual(out, "/tmp/kab:/tmp/kab/scripts")


if __name__ == "__main__":
    unittest.main()
