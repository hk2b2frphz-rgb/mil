#!/usr/bin/env python3
"""Render fixed Japanese dialogues across TTS backends for listening selection."""

from __future__ import annotations

import argparse
import html
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scripts.generate_qwen3_tts_data import (
        Dialogue,
        DialogueTurn,
        EMOTION_PRESETS,
        MossTTSD,
        Qwen3TTS,
        build_segments,
        load_dialogues_from_jsonl,
        render_stereo,
        resolve_emotion,
        safe_stem,
        validate_duplex_dialogue,
        write_wav,
    )
    from scripts.tts_comparison_backends import (
        CosyVoice2TTS,
        KokoroTTS,
        load_reference_manifest,
    )
except ImportError:
    from generate_qwen3_tts_data import (
        Dialogue,
        DialogueTurn,
        EMOTION_PRESETS,
        MossTTSD,
        Qwen3TTS,
        build_segments,
        load_dialogues_from_jsonl,
        render_stereo,
        resolve_emotion,
        safe_stem,
        validate_duplex_dialogue,
        write_wav,
    )
    from tts_comparison_backends import (
        CosyVoice2TTS,
        KokoroTTS,
        load_reference_manifest,
    )

logger = logging.getLogger(__name__)

DEFAULT_DIALOGUES = (
    Path("tests/fixtures/listening_dialogues.jsonl"),
    Path("tests/fixtures/aizuchi_dialogues.jsonl"),
)
SUPPORTED_BACKENDS = ("qwen3", "moss-ttsd", "cosyvoice2", "kokoro")
SUPPORTED_CONDITIONS = ("on", "off", "mild")
BACKEND_EMOTION_SUPPORT = {
    "qwen3": True,
    "moss-ttsd": False,
    "cosyvoice2": True,
    "kokoro": False,
}


def csv_choices(value: str, allowed: Iterable[str], flag: str) -> list[str]:
    choices = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(choices) - set(allowed))
    if invalid:
        raise ValueError(f"{flag} contains unsupported values: {invalid}")
    if not choices:
        raise ValueError(f"{flag} must not be empty")
    return list(dict.fromkeys(choices))


def load_and_validate_dialogues(paths: Sequence[Path]) -> list[dict[str, Any]]:
    dialogues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, Path):
            raise TypeError(f"dialogue paths must be pathlib.Path objects: {path!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Dialogue fixture does not exist: {path}")
        loaded = load_dialogues_from_jsonl(path)
        if not loaded:
            raise ValueError(f"Dialogue fixture is empty: {path}")
        for dialogue in loaded:
            dialogue_id = str(dialogue["id"])
            if dialogue_id in seen:
                raise ValueError(f"Duplicate dialogue id across fixtures: {dialogue_id}")
            seen.add(dialogue_id)
            errors = validate_duplex_dialogue(dialogue)
            if errors:
                raise ValueError(
                    f"Invalid dialogue {dialogue_id} in {path}: {'; '.join(errors)}"
                )
            dialogue["_source_path"] = str(path)
            dialogues.append(dialogue)
    return dialogues


def load_mild_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        mild = json.load(handle)
    if not isinstance(mild, dict):
        raise ValueError(f"Mild emotion map must be a JSON object: {path}")
    result = dict(EMOTION_PRESETS)
    result.update({str(key): str(value) for key, value in mild.items()})
    return result


def make_backend(args: argparse.Namespace, name: str) -> Any:
    if name == "qwen3":
        return Qwen3TTS(
            model_id=args.qwen_model,
            device=args.device,
            dtype_str=args.dtype,
            attn_impl=args.attn_impl,
            speaker_user=args.qwen_speaker_user,
            speaker_moshi=args.qwen_speaker_moshi,
            language="Japanese",
            instruct_user=None,
            instruct_moshi=None,
        )
    if name == "moss-ttsd":
        ref_paths, ref_texts = load_reference_manifest(args.refs_json)
        return MossTTSD(
            model_name=args.moss_model,
            codec_model_name=args.moss_codec_model,
            ref_audio_paths=ref_paths,
            ref_texts=ref_texts,
            device=args.device,
            dtype=args.dtype,
        )
    if name == "cosyvoice2":
        return CosyVoice2TTS(
            model_id=args.cosyvoice_model,
            refs_json=args.refs_json,
            device=args.device,
        )
    if name == "kokoro":
        return KokoroTTS(
            device=args.device,
            voice_moshi=args.kokoro_voice_moshi,
            voice_user=args.kokoro_voice_user,
        )
    raise ValueError(f"Unsupported backend: {name}")


def conditions_for_backend(name: str, requested: Sequence[str]) -> list[str]:
    return list(requested) if BACKEND_EMOTION_SUPPORT[name] else ["n/a"]


def to_dialogue(
    raw: dict[str, Any],
    condition: str,
    emotion_map: dict[str, str],
) -> Dialogue:
    turns: list[DialogueTurn] = []
    for raw_turn in raw["turns"]:
        if raw_turn["speaker"] == "silence":
            turns.append(
                DialogueTurn(
                    speaker="silence",
                    duration_sec=float(raw_turn.get("duration_sec", 2.0)),
                    note=str(raw_turn.get("note", "")) or None,
                )
            )
            continue
        emotion = str(raw_turn.get("emotion", "")) or None
        instruct = (
            resolve_emotion(emotion, emotion_map)
            if condition in {"on", "mild"}
            else None
        )
        turns.append(
            DialogueTurn(
                speaker=str(raw_turn["speaker"]),
                text=str(raw_turn["text"]),
                emotion=emotion,
                instruct=instruct,
                timing=str(raw_turn.get("timing") or "sequential"),
                start_after_previous_start_sec=raw_turn.get(
                    "start_after_previous_start_sec"
                ),
                truncate_previous_after_sec=raw_turn.get(
                    "truncate_previous_after_sec"
                ),
                gain=float(raw_turn.get("gain", 1.0)),
                voice_role=str(raw_turn.get("voice_role", "")) or None,
                event=str(raw_turn.get("event", "")) or None,
            )
        )
    return Dialogue(
        id=safe_stem(str(raw["id"]), "dialogue"),
        category=str(raw["category"]),
        risk_level=str(raw["risk_level"]),
        title=str(raw["title"]),
        turns=turns,
        duplex_task=str(raw.get("duplex_task") or "") or None,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def render_variant(
    raw: dict[str, Any],
    backend_name: str,
    condition: str,
    backend: Any,
    emotion_map: dict[str, str],
    out_dir: Path,
    lead_in_sec: float,
    gap_sec: float,
) -> Path:
    dialogue = to_dialogue(raw, condition, emotion_map)
    role_override = backend_name in {"moss-ttsd", "cosyvoice2", "kokoro"}
    segments, silences = build_segments(
        dialogue,
        backend,
        lead_in_sec,
        gap_sec,
        user_speaker_override="user" if role_override else None,
        other_speaker_override="other" if role_override else None,
        background_speaker_override="background" if role_override else None,
    )
    if not segments or backend.sample_rate <= 0:
        raise RuntimeError(f"No audio rendered for {dialogue.id}/{backend_name}/{condition}")

    stereo = render_stereo(segments, backend.sample_rate)
    dialogue_dir = out_dir / dialogue.id
    condition_stem = "na" if condition == "n/a" else condition
    wav_path = dialogue_dir / f"{backend_name}__{condition_stem}.wav"
    json_path = wav_path.with_suffix(".json")
    write_wav(wav_path, stereo, backend.sample_rate)
    write_json(
        json_path,
        {
            "dialogue_id": dialogue.id,
            "title": dialogue.title,
            "category": dialogue.category,
            "risk_level": dialogue.risk_level,
            "source_path": raw["_source_path"],
            "backend": backend_name,
            "emotion_condition": condition,
            "supports_emotion_instruct": BACKEND_EMOTION_SUPPORT[backend_name],
            "sample_rate": backend.sample_rate,
            "duration_sec": round(stereo.shape[-1] / backend.sample_rate, 4),
            "left_channel": "moshi",
            "right_channel": "user",
            "turns": [asdict(turn) for turn in dialogue.turns],
            "instruct_used": [
                {
                    "turn_index": index,
                    "speaker": turn.speaker,
                    "emotion": turn.emotion,
                    "instruct": turn.instruct,
                }
                for index, turn in enumerate(dialogue.turns)
                if turn.speaker != "silence"
            ],
            "silences": silences,
        },
    )
    return wav_path


def collect_sidecars(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["_json_path"] = path
        row["_wav_path"] = path.with_suffix(".wav")
        grouped.setdefault(str(row["dialogue_id"]), []).append(row)
    return grouped


def write_indexes(out_dir: Path) -> None:
    grouped = collect_sidecars(out_dir)
    markdown = [
        "# TTS comparison listening index",
        "",
        "Stereo format: left channel = moshi, right channel = user.",
        "",
        "| Backend | Emotion variants | Voice mode |",
        "|---|---|---|",
        "| qwen3 | on, off, mild | fixed Qwen3 presets |",
        "| moss-ttsd | n/a | shared Qwen3 clone references |",
        "| cosyvoice2 | on, off, mild | shared Qwen3 clone references |",
        "| kokoro | n/a | fixed Japanese voices |",
        "",
    ]
    html_lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>TTS comparison</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.4rem}"
        "audio{width:520px}</style></head><body>",
        "<h1>TTS comparison listening index</h1>",
        "<p>Stereo: left = moshi, right = user.</p>",
    ]
    for dialogue_id, rows in sorted(grouped.items()):
        first = rows[0]
        markdown.extend([f"## {dialogue_id}", "", str(first["title"]), ""])
        html_lines.extend(
            [
                f"<h2>{html.escape(dialogue_id)}</h2>",
                f"<p>{html.escape(str(first['title']))}</p>",
                "<table><tr><th>Backend</th><th>Emotion</th><th>Audio</th><th>Metadata</th></tr>",
            ]
        )
        for row in sorted(rows, key=lambda item: (item["backend"], item["emotion_condition"])):
            wav_rel = Path(row["_wav_path"]).relative_to(out_dir).as_posix()
            json_rel = Path(row["_json_path"]).relative_to(out_dir).as_posix()
            label = f"{row['backend']} / {row['emotion_condition']}"
            markdown.append(f"- [{label}]({wav_rel}) ([json]({json_rel}))")
            html_lines.append(
                "<tr>"
                f"<td>{html.escape(str(row['backend']))}</td>"
                f"<td>{html.escape(str(row['emotion_condition']))}</td>"
                f"<td><audio controls preload='none' src='{html.escape(wav_rel)}'></audio></td>"
                f"<td><a href='{html.escape(json_rel)}'>json</a></td>"
                "</tr>"
            )
        markdown.append("")
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    (out_dir / "INDEX.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (out_dir / "INDEX.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dialogues-jsonl",
        nargs="+",
        type=Path,
        default=list(DEFAULT_DIALOGUES),
    )
    parser.add_argument("--backends", default=",".join(SUPPORTED_BACKENDS))
    parser.add_argument("--emotion-conditions", default="on,off,mild")
    parser.add_argument("--refs-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--mild-emotion-map",
        type=Path,
        default=Path("configs/emotion_presets_mild.json"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--attn-impl", default="default")
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    parser.add_argument("--qwen-speaker-user", default="Ono_Anna")
    parser.add_argument("--qwen-speaker-moshi", default="Serena")
    parser.add_argument("--moss-model", default="OpenMOSS-Team/MOSS-TTSD-v1.0")
    parser.add_argument(
        "--moss-codec-model",
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
    )
    parser.add_argument(
        "--cosyvoice-model",
        default="FunAudioLLM/CosyVoice2-0.5B",
    )
    parser.add_argument("--kokoro-voice-moshi", default="jf_alpha")
    parser.add_argument("--kokoro-voice-user", default="jm_kumo")
    parser.add_argument("--lead-in-sec", type=float, default=0.3)
    parser.add_argument("--gap-sec", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    backends = csv_choices(args.backends, SUPPORTED_BACKENDS, "--backends")
    conditions = csv_choices(
        args.emotion_conditions, SUPPORTED_CONDITIONS, "--emotion-conditions"
    )
    dialogues = load_and_validate_dialogues(args.dialogues_jsonl)
    mild_map = load_mild_map(args.mild_emotion_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for backend_name in backends:
        backend = make_backend(args, backend_name)
        backend.load()
        for raw in dialogues:
            for condition in conditions_for_backend(backend_name, conditions):
                logger.info(
                    "Rendering dialogue=%s backend=%s emotion=%s",
                    raw["id"],
                    backend_name,
                    condition,
                )
                render_variant(
                    raw,
                    backend_name,
                    condition,
                    backend,
                    mild_map if condition == "mild" else dict(EMOTION_PRESETS),
                    args.out_dir,
                    args.lead_in_sec,
                    args.gap_sec,
                )
        del backend
    write_indexes(args.out_dir)
    print(args.out_dir / "INDEX.md")


if __name__ == "__main__":
    main()
