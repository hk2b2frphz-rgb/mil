from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = ROOT / "scripts" / "patch_vllm_omni_qwen3_tts_v100.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("patch_vllm_omni_v100", PATCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch_module = _load_patch_module()


def _upstream_qwen_dir() -> Path | None:
    """Find the optional v0.22.0 inspection checkout used by local reviews."""
    temp_root = Path.home() / "AppData" / "Local" / "Temp"
    if not temp_root.is_dir():
        return None
    candidates = sorted(temp_root.glob("vllm-omni-inspect-*/vllm_omni/model_executor/models/qwen3_tts"))
    return candidates[-1] if candidates else None


def test_source_transforms_replace_only_runtime_bfloat16_casts() -> None:
    talker = """
        model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self.encoder.to(dtype=torch.bfloat16)
        a = torch.bfloat16
        b = torch.bfloat16
        c = torch.bfloat16
        d = torch.bfloat16
        e = torch.bfloat16
        f = torch.bfloat16
        g = torch.bfloat16
        h = torch.bfloat16
        i = torch.bfloat16
        j = torch.bfloat16
        k = torch.bfloat16
        self._prompt_builder = Qwen3TTSPromptEmbedsBuilder(
            config=self.config,
            talker_config=self.talker_config,
            model_path=self.model_path,
        )
"""
    builder = """
    def __init__(
        self,
        *,
        config: Qwen3TTSConfig,
        talker_config: Qwen3TTSTalkerConfig,
        model_path: str,
    ):
        self._config = config
        self._talker_config = talker_config
        self._model_path = model_path
        a = torch.bfloat16
        b = torch.bfloat16
        c = torch.bfloat16
        d = torch.bfloat16
        e = torch.bfloat16
        f = torch.bfloat16
        g = torch.bfloat16
        h = torch.bfloat16
"""

    patched_talker, changed_talker = patch_module.patch_talker_source(talker)
    patched_builder, changed_builder = patch_module.patch_builder_source(builder)

    assert changed_talker and changed_builder
    assert patched_talker.count("torch.bfloat16") == 1  # safe default only
    assert patched_talker.count("self._model_dtype") == 14
    assert "model_dtype=self._model_dtype," in patched_talker
    assert "torch.bfloat16" not in patched_builder
    assert "model_dtype: torch.dtype," in patched_builder
    assert patch_module.PATCH_MARKER in patched_talker
    assert patch_module.PATCH_MARKER in patched_builder

    assert patch_module.patch_talker_source(patched_talker) == (patched_talker, False)
    assert patch_module.patch_builder_source(patched_builder) == (patched_builder, False)


def test_patch_rejects_unknown_layout() -> None:
    with pytest.raises(patch_module.PatchError, match="expected 12"):
        patch_module.patch_talker_source("x = torch.bfloat16\n")
    with pytest.raises(patch_module.PatchError, match="expected 8"):
        patch_module.patch_builder_source("x = torch.bfloat16\n")


def test_patch_against_vllm_omni_022_source_when_available(tmp_path: Path) -> None:
    upstream = _upstream_qwen_dir()
    if upstream is None:
        pytest.skip("optional vLLM-Omni v0.22.0 inspection checkout is unavailable")

    qwen_dir = tmp_path / "qwen3_tts"
    qwen_dir.mkdir()
    for name in (patch_module.TALKER_NAME, patch_module.BUILDER_NAME):
        shutil.copy2(upstream / name, qwen_dir / name)

    result = subprocess.run(
        [
            sys.executable,
            str(PATCH_SCRIPT),
            "--qwen-dir",
            str(qwen_dir),
            "--skip-version-check",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    check = subprocess.run(
        [
            sys.executable,
            str(PATCH_SCRIPT),
            "--qwen-dir",
            str(qwen_dir),
            "--skip-version-check",
            "--check",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    compile(
        (qwen_dir / patch_module.TALKER_NAME).read_text(encoding="utf-8"),
        patch_module.TALKER_NAME,
        "exec",
    )
    compile(
        (qwen_dir / patch_module.BUILDER_NAME).read_text(encoding="utf-8"),
        patch_module.BUILDER_NAME,
        "exec",
    )
