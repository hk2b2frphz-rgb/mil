#!/usr/bin/env python3
"""Make vLLM-Omni 0.22.0 Qwen3-TTS honor its configured model dtype.

The 0.22.0 Qwen3-TTS implementation hard-codes bfloat16 for the talker,
reference-audio encoder, and speaker encoder even when the stage YAML selects
float16.  NVIDIA V100 (sm70) does not support bfloat16.  This small,
version-pinned source patch replaces those internal casts with the dtype that
vLLM already resolved from ``model_config.dtype``.

The operation is idempotent and atomic.  It intentionally fails on an
unrecognised source layout instead of partially patching a newer release.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import tempfile


SUPPORTED_VERSION = "0.22.0"
QWEN_RELATIVE = Path("model_executor/models/qwen3_tts")
TALKER_NAME = "qwen3_tts_talker.py"
BUILDER_NAME = "prompt_embeds_builder.py"
PATCH_MARKER = "miltoka V100 patch: follow vLLM's configured model dtype"


class PatchError(RuntimeError):
    """Raised when the installed source does not match the pinned layout."""


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one source marker, found {count}")
    return text.replace(old, new, 1)


def patch_talker_source(source: str) -> tuple[str, bool]:
    """Return the patched talker source and whether it changed."""
    hardcoded = [
        line
        for line in source.splitlines()
        if "torch.bfloat16" in line and "model_dtype = getattr" not in line
    ]
    already_patched = PATCH_MARKER in source
    if already_patched:
        if hardcoded:
            raise PatchError(
                f"{TALKER_NAME}: patch marker exists but "
                f"{len(hardcoded)} hard-coded bfloat16 cast(s) remain"
            )
        if "model_dtype=self._model_dtype," not in source:
            raise PatchError(f"{TALKER_NAME}: patched builder dtype argument is missing")
        return source, False

    if len(hardcoded) != 12:
        raise PatchError(
            f"{TALKER_NAME}: expected 12 hard-coded bfloat16 casts in "
            f"vLLM-Omni {SUPPORTED_VERSION}, found {len(hardcoded)}"
        )

    source = _replace_once(
        source,
        '        model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)\n',
        '        model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)\n'
        f"        # {PATCH_MARKER}.\n"
        "        self._model_dtype = model_dtype\n",
        label=f"{TALKER_NAME}: model dtype",
    )
    source = _replace_once(
        source,
        "            talker_config=self.talker_config,\n"
        "            model_path=self.model_path,\n",
        "            talker_config=self.talker_config,\n"
        "            model_dtype=self._model_dtype,\n"
        "            model_path=self.model_path,\n",
        label=f"{TALKER_NAME}: prompt builder",
    )

    lines: list[str] = []
    replacements = 0
    for line in source.splitlines(keepends=True):
        if "torch.bfloat16" in line and "model_dtype = getattr" not in line:
            replacements += line.count("torch.bfloat16")
            line = line.replace("torch.bfloat16", "self._model_dtype")
        lines.append(line)
    if replacements != 12:
        raise PatchError(
            f"{TALKER_NAME}: expected to replace 12 casts, replaced {replacements}"
        )
    return "".join(lines), True


def patch_builder_source(source: str) -> tuple[str, bool]:
    """Return the patched prompt-builder source and whether it changed."""
    hardcoded_count = source.count("torch.bfloat16")
    already_patched = PATCH_MARKER in source
    if already_patched:
        if hardcoded_count:
            raise PatchError(
                f"{BUILDER_NAME}: patch marker exists but "
                f"{hardcoded_count} hard-coded bfloat16 cast(s) remain"
            )
        if "model_dtype: torch.dtype," not in source:
            raise PatchError(f"{BUILDER_NAME}: model_dtype constructor argument is missing")
        return source, False

    if hardcoded_count != 8:
        raise PatchError(
            f"{BUILDER_NAME}: expected 8 hard-coded bfloat16 casts in "
            f"vLLM-Omni {SUPPORTED_VERSION}, found {hardcoded_count}"
        )

    source = _replace_once(
        source,
        "        talker_config: Qwen3TTSTalkerConfig,\n"
        "        model_path: str,\n",
        "        talker_config: Qwen3TTSTalkerConfig,\n"
        "        model_dtype: torch.dtype,\n"
        "        model_path: str,\n",
        label=f"{BUILDER_NAME}: constructor",
    )
    source = _replace_once(
        source,
        "        self._talker_config = talker_config\n"
        "        self._model_path = model_path\n",
        "        self._talker_config = talker_config\n"
        f"        # {PATCH_MARKER}.\n"
        "        self._model_dtype = model_dtype\n"
        "        self._model_path = model_path\n",
        label=f"{BUILDER_NAME}: model dtype",
    )
    source = source.replace("torch.bfloat16", "self._model_dtype")
    return source, True


def _atomic_write(path: Path, source: str) -> None:
    compile(source, str(path), "exec")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def patch_qwen_tree(qwen_dir: Path, *, check: bool = False) -> bool:
    """Patch or validate a Qwen3-TTS source directory.

    Returns ``True`` when at least one file was changed.
    """
    talker_path = qwen_dir / TALKER_NAME
    builder_path = qwen_dir / BUILDER_NAME
    for path in (talker_path, builder_path):
        if not path.is_file():
            raise PatchError(f"required vLLM-Omni source file is missing: {path}")

    talker_source = talker_path.read_text(encoding="utf-8")
    builder_source = builder_path.read_text(encoding="utf-8")
    talker_patched, talker_changed = patch_talker_source(talker_source)
    builder_patched, builder_changed = patch_builder_source(builder_source)

    if check and (talker_changed or builder_changed):
        raise PatchError(
            "vLLM-Omni Qwen3-TTS is not V100-safe yet; rerun "
            "scripts/setup_vllm_omni_v100_env.sh (FRESH is not required)"
        )
    if check:
        return False

    if talker_changed:
        _atomic_write(talker_path, talker_patched)
    if builder_changed:
        _atomic_write(builder_path, builder_patched)

    # Re-read both files so a failed or external concurrent write is detected.
    patch_talker_source(talker_path.read_text(encoding="utf-8"))
    patch_builder_source(builder_path.read_text(encoding="utf-8"))
    return talker_changed or builder_changed


def installed_qwen_dir() -> Path:
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise PatchError("vllm_omni is not installed in this Python environment")
    return Path(next(iter(spec.submodule_search_locations))) / QWEN_RELATIVE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-dir",
        type=Path,
        help="Qwen3-TTS package directory (mainly for tests); defaults to the installed package",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that the patch is already present without changing files",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="skip installed distribution version validation (only for source-tree tests)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_version_check:
        try:
            version = importlib.metadata.version("vllm-omni")
        except importlib.metadata.PackageNotFoundError as exc:
            raise PatchError("vllm-omni distribution metadata is missing") from exc
        if version != SUPPORTED_VERSION:
            raise PatchError(
                f"unsupported vllm-omni {version}; this patch is pinned to "
                f"{SUPPORTED_VERSION}"
            )

    qwen_dir = args.qwen_dir or installed_qwen_dir()
    changed = patch_qwen_tree(qwen_dir, check=args.check)
    if args.check:
        print(f"vLLM-Omni {SUPPORTED_VERSION} Qwen3-TTS V100 dtype patch: OK")
    elif changed:
        print(f"Patched vLLM-Omni {SUPPORTED_VERSION} Qwen3-TTS for V100 float16")
    else:
        print(f"vLLM-Omni {SUPPORTED_VERSION} Qwen3-TTS V100 dtype patch already applied")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
