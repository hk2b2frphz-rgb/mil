#!/usr/bin/env python3
"""Duck-typed adapters used by the TTS listening comparison harness."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.generate_qwen3_tts_data import _resample
except ImportError:
    from generate_qwen3_tts_data import _resample

logger = logging.getLogger(__name__)


def load_reference_manifest(path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Load the role-indexed refs.json emitted by build_moss_reference_voices.py."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    roles = data.get("roles")
    if not isinstance(roles, dict):
        raise ValueError(f"Invalid reference manifest (missing roles object): {path}")

    audio_paths: dict[str, Path] = {}
    transcripts: dict[str, str] = {}
    for role in ("user", "moshi", "other", "background"):
        entry = roles.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid reference manifest (missing role {role!r}): {path}")
        audio_path = Path(str(entry.get("path", "")))
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"Reference WAV for {role!r} does not exist: {audio_path}")
        transcript = str(entry.get("transcript", "")).strip()
        if not transcript:
            raise ValueError(f"Reference transcript for {role!r} is empty: {path}")
        audio_paths[role] = audio_path
        transcripts[role] = transcript
    return audio_paths, transcripts


class CosyVoiceTTS:
    """CosyVoice zero-shot voice cloning for smoke-listening runs."""

    supports_instruct = True

    def __init__(
        self,
        model_id: str,
        refs_json: Path,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.refs_json = refs_json
        self.device = device
        self.model: Any = None
        self.sample_rate = 0
        self.ref_audio_paths, self.ref_texts = load_reference_manifest(refs_json)

    def load(self) -> None:
        if self.model is not None:
            return

        # NOTE: The official CosyVoice repository is not currently packaged as a
        # stable PyPI wheel. The PBS job clones it recursively and sets this path.
        repo = os.environ.get("COSYVOICE_REPO")
        if repo:
            repo_path = Path(repo).resolve()
            sys.path.insert(0, str(repo_path))
            sys.path.insert(0, str(repo_path / "third_party" / "Matcha-TTS"))
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "CosyVoice requires the official FunAudioLLM/CosyVoice checkout "
                "on COSYVOICE_REPO plus the isolated dependencies in "
                "scripts/run_tts_comparison.pbs."
            ) from exc

        model_dir = (
            os.environ.get("COSYVOICE3_MODEL_DIR")
            or os.environ.get("COSYVOICE_MODEL_DIR")
            or os.environ.get("COSYVOICE2_MODEL_DIR")
        )
        if not model_dir:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(self.model_id)

        # NOTE/FIXME(cluster verification): current upstream AutoModel selects
        # CUDA internally. Its public constructor does not consistently expose a
        # device argument across CosyVoice2 revisions.
        self.model = AutoModel(model_dir=model_dir)
        self.sample_rate = int(self.model.sample_rate)
        logger.info("CosyVoice loaded from %s at %d Hz", model_dir, self.sample_rate)

    @staticmethod
    def _role(speaker_role: str, speaker_override: str | None) -> str:
        if speaker_override in {"user", "moshi", "other", "background"}:
            return str(speaker_override)
        return speaker_role if speaker_role in {"user", "moshi"} else "user"

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        self.load()
        role = self._role(speaker_role, speaker_override)
        prompt_wav = str(self.ref_audio_paths[role])

        if instruct:
            prompt = instruct.rstrip()
            if not prompt.endswith("<|endofprompt|>"):
                prompt += "<|endofprompt|>"
            iterator = self.model.inference_instruct2(
                text, prompt, prompt_wav, stream=False
            )
        else:
            iterator = self.model.inference_zero_shot(
                text,
                f"You are a helpful assistant.<|endofprompt|>{self.ref_texts[role]}",
                prompt_wav,
                stream=False,
            )

        chunks: list[np.ndarray] = []
        for item in iterator:
            audio = item["tts_speech"]
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks")
        return np.concatenate(chunks).astype(np.float32, copy=False)


CosyVoice2TTS = CosyVoiceTTS
CosyVoice3TTS = CosyVoiceTTS


# Misaki Japanese G2P preserves ASCII punctuation but drops full-width forms.
# This changes only the text passed to synthesis, never the training transcript.
_G2P_PUNCT = str.maketrans({
    "。": ".", "、": ",", "！": "!", "？": "?", "：": ":", "；": ";",
})


def punctuation_for_g2p(text: str) -> str:
    return text.translate(_G2P_PUNCT)


class RoleRoutedTTS:
    """Send selected speaker roles to another TTS backend."""

    supports_instruct = False

    def __init__(self, default: Any, by_role: dict[str, Any]) -> None:
        self.default = default
        self.by_role = dict(by_role)

    @property
    def sample_rate(self) -> int:
        for backend in (self.default, *self.by_role.values()):
            rate = getattr(backend, "sample_rate", None)
            if rate:
                return int(rate)
        raise RuntimeError("no backend reported a sample rate")

    def backend_for(self, speaker_role: str) -> Any:
        return self.by_role.get(speaker_role, self.default)

    def load(self) -> None:
        for backend in (self.default, *self.by_role.values()):
            loader = getattr(backend, "load", None)
            if callable(loader):
                loader()

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        backend = self.backend_for(speaker_role)
        if not getattr(backend, "supports_instruct", False):
            instruct = None
        return backend.synthesize(
            text, speaker_role, instruct=instruct, speaker_override=speaker_override
        )

    def release(self) -> None:
        for backend in (self.default, *self.by_role.values()):
            releaser = getattr(backend, "release", None)
            if callable(releaser):
                releaser()


class KokoroTTS:
    """Kokoro fixed Japanese voices; cloning and emotion instructions unsupported."""

    supports_instruct = False

    def __init__(
        self,
        device: str = "cuda",
        voice_moshi: str = "jf_alpha",
        voice_user: str = "jm_kumo",
        voice_other: str = "jf_gongitsune",
        voice_background: str = "jf_tebukuro",
        model_id: str = "hexgrad/Kokoro-82M",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.voices = {
            "moshi": voice_moshi,
            "user": voice_user,
            "other": voice_other,
            "background": voice_background,
        }
        self.pipeline: Any = None
        self.sample_rate = 24000

    def load(self) -> None:
        if self.pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro requires kokoro>=0.9.4 and misaki[ja] in its isolated env. "
                f"Underlying import error: {exc}"
            ) from exc
        self.pipeline = KPipeline(
            lang_code="j",
            repo_id=self.model_id,
            device=self.device,
        )
        logger.info("Kokoro loaded with fixed Japanese voices at %d Hz", self.sample_rate)

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        self.load()
        override = str(speaker_override) if speaker_override else ""
        if override in self.voices:
            voice = self.voices[override]
        elif override.startswith(("jf_", "jm_")):
            # Kokoro の日本語話者IDを直接指定できる（対話ごとの user 話者
            # ローテーション用。generate_qwen3_tts_data.py --kokoro-user-voices）。
            voice = override
        else:
            voice = self.voices.get(speaker_role, self.voices["user"])
        if instruct:
            logger.debug("Kokoro ignores instruct=%r", instruct[:48])

        chunks: list[np.ndarray] = []
        for result in self.pipeline(punctuation_for_g2p(text), voice=voice, speed=1.0):
            audio = result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")
        audio = np.concatenate(chunks)
        return _resample(audio, 24000, self.sample_rate)
