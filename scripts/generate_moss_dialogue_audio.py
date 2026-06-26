#!/usr/bin/env python3
"""Generate one native MOSS-TTSD mono mix per complete dialogue."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from generate_qwen3_tts_data import (
    load_dialogues_from_jsonl,
    safe_stem,
    write_wav,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SPEAKER_TAGS = {"moshi": "[S1]", "user": "[S2]"}
SPEAKER_ROLES = {"S1": "moshi", "S2": "user"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MOSS-TTSD naturalness audition: generate a native multi-speaker "
            "mono dialogue mix."
        )
    )
    parser.add_argument("--dialogues-jsonl", required=True, type=Path)
    parser.add_argument("--moss-refs-json", required=True, type=Path)
    parser.add_argument(
        "--moss-model",
        default="OpenMOSS-Team/MOSS-TTSD-v1.0",
    )
    parser.add_argument(
        "--moss-codec-model",
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dialogue_audio"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()
    if not args.dialogues_jsonl.is_file():
        parser.error(f"dialogues JSONL does not exist: {args.dialogues_jsonl}")
    if not args.moss_refs_json.is_file():
        parser.error(f"MOSS refs JSON does not exist: {args.moss_refs_json}")
    return args


def build_dialogue_text(dialogue: dict[str, Any]) -> str:
    """Keep textual turn order while deliberately discarding scripted timing."""
    parts: list[str] = []
    for turn in dialogue.get("turns", []):
        speaker = str(turn.get("speaker") or "").strip().lower()
        if speaker == "silence":
            continue
        tag = SPEAKER_TAGS.get(speaker)
        text = str(turn.get("text") or "").strip()
        if tag and text:
            parts.append(f"{tag} {text}")
    return " ".join(parts)


def load_two_speaker_references(
    refs_json: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    with refs_json.open(encoding="utf-8") as f:
        data = json.load(f)
    roles = data.get("roles", data)

    paths: dict[str, Path] = {}
    texts: dict[str, str] = {}
    for role in ("moshi", "user"):
        entry = roles.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing role {role!r} in {refs_json}")
        path = Path(str(entry.get("path") or ""))
        if not path.is_absolute():
            path = refs_json.parent / path
        transcript = str(entry.get("transcript") or "").strip()
        if not path.is_file():
            raise FileNotFoundError(f"Reference WAV for {role!r} not found: {path}")
        if not transcript:
            raise ValueError(f"Reference transcript for {role!r} is empty")
        paths[role] = path
        texts[role] = transcript
    return paths, texts


class MossDialogueGenerator:
    """Small isolation layer for the still-unverified MOSS dialogue API."""

    def __init__(
        self,
        model_name: str,
        codec_model_name: str,
        ref_paths: dict[str, Path],
        ref_texts: dict[str, str],
        device: str,
        dtype: str,
    ) -> None:
        self.model_name = model_name
        self.codec_model_name = codec_model_name
        self.ref_paths = ref_paths
        self.ref_texts = ref_texts
        self.device_name = "cuda:0" if device == "cuda" else device
        self.dtype_name = dtype
        self.model: Any = None
        self.processor: Any = None
        self.prompt_wavs: dict[str, Any] = {}
        self.sample_rate = 0
        self.last_full_text_sent = ""
        # Reference encodings are identical for every dialogue (same prompt
        # wavs), so they are computed once in load() and reused, instead of
        # re-running the codec encoder on each generate() call.
        self.reference_codes: Any = None
        self.prompt_audio_codes: Any = None

    def load(self) -> None:
        if self.model is not None:
            return

        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.dtype_name]

        if self.device_name.startswith("cuda"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # ASSUMPTION: the separately published codec remains compatible with the
        # processor's encode/decode hooks in the Transformers 5 remote code.
        self.processor.audio_tokenizer = AutoModel.from_pretrained(
            self.codec_model_name,
            trust_remote_code=True,
        ).to(self.device_name)
        self.processor.audio_tokenizer.eval()
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device_name)
        self.model.eval()

        self.sample_rate = int(self.processor.model_config.sampling_rate)
        for role, path in self.ref_paths.items():
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if int(sr) != self.sample_rate:
                wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
            self.prompt_wavs[role] = wav

        # Encode the references once. These inputs never change across dialogues,
        # so caching here removes two codec-encode passes from every generate().
        speaker1 = self.prompt_wavs["moshi"]
        speaker2 = self.prompt_wavs["user"]
        # ASSUMPTION: encode_audios_from_wav preserves list order, so reference
        # index 0 maps to [S1] and reference index 1 maps to [S2].
        self.reference_codes = self.processor.encode_audios_from_wav(
            [speaker1, speaker2],
            sampling_rate=self.sample_rate,
        )
        # ASSUMPTION: the continuation prompt audio is the two references
        # concatenated in [S1], [S2] order, as shown in the model card.
        concat_prompt_wav = torch.cat([speaker1, speaker2], dim=-1)
        self.prompt_audio_codes = self.processor.encode_audios_from_wav(
            [concat_prompt_wav],
            sampling_rate=self.sample_rate,
        )[0]

    def _build_user_message(
        self,
        *,
        text: str,
        prompt_text_speaker1: str,
        prompt_text_speaker2: str,
    ) -> list[list[Any]]:
        """Translate named two-speaker inputs into the model-card continuation API."""
        assert self.processor is not None
        assert self.reference_codes is not None
        assert self.prompt_audio_codes is not None

        # ASSUMPTION: prompt transcripts must prefix the requested dialogue in
        # the same text field and retain their matching [S1]/[S2] tags.
        full_text = (
            f"[S1] {prompt_text_speaker1} "
            f"[S2] {prompt_text_speaker2} "
            f"{text}"
        )
        self.last_full_text_sent = full_text
        # ASSUMPTION: current remote code exposes build_user_message(text,
        # reference) plus a continuation assistant message, rather than literal
        # prompt_audio_speaker1/... keyword parameters. reference_codes and
        # prompt_audio_codes are cached in load() (identical for every dialogue).
        return [[
            self.processor.build_user_message(
                text=full_text,
                reference=self.reference_codes,
            ),
            self.processor.build_assistant_message(
                audio_codes_list=[self.prompt_audio_codes],
            ),
        ]]

    def generate(self, text: str) -> np.ndarray:
        self.load()
        assert self.model is not None
        assert self.processor is not None
        import torch

        conversations = self._build_user_message(
            text=text,
            prompt_text_speaker1=self.ref_texts["moshi"],
            prompt_text_speaker2=self.ref_texts["user"],
        )
        batch = self.processor(conversations, mode="continuation")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # ASSUMPTION: one generate call with the model-card decoding
            # defaults is sufficient for each complete pilot dialogue.
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=2000,
            )
        messages = self.processor.decode(outputs)
        if not messages or not messages[0].audio_codes_list:
            raise RuntimeError("MOSS-TTSD returned no decoded dialogue audio")

        # ASSUMPTION: decode() returns the complete speaker-mixed waveform as
        # audio_codes_list[0], rather than one waveform per speaker.
        audio = messages[0].audio_codes_list[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim == 2:
            # ASSUMPTION: if decode exposes channels, averaging them is the
            # correct listening-only mono mix.
            audio = audio.mean(axis=0)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        return audio.astype(np.float32, copy=False)


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    dialogues = load_dialogues_from_jsonl(args.dialogues_jsonl)
    ref_paths, ref_texts = load_two_speaker_references(args.moss_refs_json)
    generator = MossDialogueGenerator(
        model_name=args.moss_model,
        codec_model_name=args.moss_codec_model,
        ref_paths=ref_paths,
        ref_texts=ref_texts,
        device=args.device,
        dtype=args.dtype,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for index, dialogue in enumerate(dialogues, start=1):
        dialogue_text = build_dialogue_text(dialogue)
        if not dialogue_text:
            logger.warning("Skipping dialogue with no user/moshi text: %r", dialogue.get("id"))
            continue
        dialogue_id = safe_stem(
            str(dialogue.get("id") or ""),
            f"dialogue_{index:03d}",
        )
        logger.info("[%d/%d] Generating %s", index, len(dialogues), dialogue_id)
        mono = generator.generate(dialogue_text)
        wav_path = args.out_dir / f"{dialogue_id}.wav"
        write_wav(wav_path, mono[np.newaxis, :], generator.sample_rate)
        write_metadata(wav_path.with_suffix(".json"), {
            "id": dialogue_id,
            "mode": "moss-ttsd-naturalness-audition",
            "dialogue_text": dialogue_text,
            "full_text_sent": generator.last_full_text_sent,
            "speaker_mapping": SPEAKER_ROLES,
            "channels": 1,
            "sample_rate": generator.sample_rate,
            "moss_model": args.moss_model,
            "moss_codec_model": args.moss_codec_model,
        })
        logger.info("Wrote mono dialogue mix: %s", wav_path)


if __name__ == "__main__":
    main()
