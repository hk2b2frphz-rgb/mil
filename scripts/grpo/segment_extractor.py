"""Extract interactivity event segments from stereo dialogue WAVs.

Implements the segment extraction criteria from the Kyutai paper
"Multi-Faceted Interactivity Alignment" (Section 3.2) exactly:

  D_pause  — User utterance ≥τ_min with internal pause, Y silent
  D_turn   — X→Y consecutive pair, both ≥τ_min, gap ≤0.4s
  D_bc     — User utterance ≥τ_min, Y produces only short (≤1s) utterances
  D_int    — X→Y→X→Y 4-utterance sequence (interruption pattern)

Usage:
    python scripts/grpo/segment_extractor.py \
        --manifest data/runs/qwen_dialogues_1000/training_set/synthetic_moshi_train.jsonl \
        --out-dir data/runs/qwen_dialogues_1000/grpo_segments \
        --max-per-axis 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 24_000


# ── Audio I/O ───────────────────────────────────────────────────────

def _load_audio_channel(wav_path: Path, channel: int) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import torchaudio, torch
        t = torch.from_numpy(data.T)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        data = t.numpy().T
    if data.shape[1] < 2:
        raise ValueError(f"Expected stereo, got {data.shape[1]} channels: {wav_path}")
    return data[:, channel]


# ── VAD ─────────────────────────────────────────────────────────────

_vad_model_cache: dict[str, Any] = {}


def _get_vad_model():
    if "model" not in _vad_model_cache:
        import torch
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        _vad_model_cache["model"] = model
        _vad_model_cache["utils"] = utils
    return _vad_model_cache["model"], _vad_model_cache["utils"]


def _run_vad(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[dict[str, float]]:
    """Return utterances as [{start, end, dur}, ...] via Silero VAD."""
    import torch
    model, utils = _get_vad_model()
    get_speech_timestamps = utils[0]
    tensor = torch.from_numpy(audio)
    stamps = get_speech_timestamps(tensor, model, sampling_rate=sr)
    result = []
    for s in stamps:
        start = s["start"] / sr
        end = s["end"] / sr
        result.append({"start": start, "end": end, "dur": end - start})
    return result


def _has_internal_pause(
    utterance: dict[str, float],
    all_utterances: list[dict[str, float]],
    speaker_utterances: list[dict[str, float]],
    min_pause_sec: float = 0.3,
) -> bool:
    """Check if a speaker's utterance span contains an internal pause.

    An 'internal pause' means the speaker has multiple sub-utterances within
    the span of what appears as one turn, separated by short silences.
    """
    u_start, u_end = utterance["start"], utterance["end"]
    sub_utts = [
        u for u in speaker_utterances
        if u["start"] >= u_start and u["end"] <= u_end and u is not utterance
    ]
    if not sub_utts:
        return False
    all_in_span = sorted(
        [utterance] + sub_utts, key=lambda x: x["start"],
    )
    for i in range(len(all_in_span) - 1):
        gap = all_in_span[i + 1]["start"] - all_in_span[i]["end"]
        if gap >= min_pause_sec:
            return True
    return False


def _merge_close_utterances(
    utts: list[dict[str, float]],
    max_gap: float = 0.3,
) -> list[dict[str, float]]:
    """Merge utterances separated by gaps < max_gap into single spans."""
    if not utts:
        return []
    sorted_utts = sorted(utts, key=lambda u: u["start"])
    merged = [dict(sorted_utts[0])]
    for u in sorted_utts[1:]:
        if u["start"] - merged[-1]["end"] < max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], u["end"])
            merged[-1]["dur"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(dict(u))
    return merged


# ── Segment extraction per axis (Kyutai Section 3.2) ────────────────

def _extract_pause_segments(
    user_utts: list[dict[str, float]],
    moshi_utts: list[dict[str, float]],
    wav_path: str,
    total_dur: float,
    tau_min: float = 4.0,
) -> list[dict[str, Any]]:
    """D_pause: User utterance ≥τ_min with internal pause, Y produces no speech."""
    user_merged = _merge_close_utterances(user_utts, max_gap=0.15)
    segments = []
    for u in user_merged:
        if u["dur"] < tau_min:
            continue
        has_pause = _has_internal_pause(u, user_utts, user_utts, min_pause_sec=0.3)
        if not has_pause:
            sub_utts = [
                s for s in user_utts
                if s["start"] >= u["start"] and s["end"] <= u["end"]
            ]
            if len(sub_utts) < 2:
                continue
            sub_sorted = sorted(sub_utts, key=lambda x: x["start"])
            has_pause = any(
                sub_sorted[i + 1]["start"] - sub_sorted[i]["end"] >= 0.3
                for i in range(len(sub_sorted) - 1)
            )
        if not has_pause:
            continue
        moshi_in_span = [
            m for m in moshi_utts
            if m["start"] < u["end"] and m["end"] > u["start"]
        ]
        if moshi_in_span:
            continue
        pause_times = []
        sub_in = sorted(
            [s for s in user_utts if s["start"] >= u["start"] and s["end"] <= u["end"]],
            key=lambda x: x["start"],
        )
        for i in range(len(sub_in) - 1):
            gap_start = sub_in[i]["end"]
            gap_end = sub_in[i + 1]["start"]
            if gap_end - gap_start >= 0.3:
                pause_times.append({"start": gap_start, "end": gap_end})

        segments.append({
            "wav_path": wav_path,
            "segment_start_sec": u["start"],
            "segment_end_sec": u["end"],
            "axis": "pause",
            "internal_pauses": pause_times,
        })
    return segments


def _extract_turn_taking_segments(
    user_utts: list[dict[str, float]],
    moshi_utts: list[dict[str, float]],
    wav_path: str,
    total_dur: float,
    tau_min: float = 5.0,
    max_gap: float = 0.4,
) -> list[dict[str, Any]]:
    """D_turn: X→Y consecutive pair, both ≥τ_min, gap ≤ max_gap."""
    user_merged = _merge_close_utterances(user_utts, max_gap=0.3)
    moshi_merged = _merge_close_utterances(moshi_utts, max_gap=0.3)
    segments = []
    for u in user_merged:
        if u["dur"] < tau_min:
            continue
        for m in moshi_merged:
            if m["dur"] < tau_min:
                continue
            gap = m["start"] - u["end"]
            if 0 <= gap <= max_gap:
                segments.append({
                    "wav_path": wav_path,
                    "segment_start_sec": u["start"],
                    "segment_end_sec": m["end"],
                    "axis": "turn_taking",
                    "user_turn_end_sec": u["end"],
                })
                break
    return segments


def _extract_backchannel_segments(
    user_utts: list[dict[str, float]],
    moshi_utts: list[dict[str, float]],
    wav_path: str,
    total_dur: float,
    tau_min: float = 5.0,
    max_bc_dur: float = 1.0,
) -> list[dict[str, Any]]:
    """D_bc: User utterance ≥τ_min, Y produces only short (≤1s) utterances."""
    user_merged = _merge_close_utterances(user_utts, max_gap=0.3)
    segments = []
    for u in user_merged:
        if u["dur"] < tau_min:
            continue
        moshi_during = [
            m for m in moshi_utts
            if m["start"] >= u["start"] and m["end"] <= u["end"]
        ]
        if not moshi_during:
            continue
        if any(m["dur"] > max_bc_dur for m in moshi_during):
            continue
        gt_bc_times = [m["start"] for m in moshi_during]
        segments.append({
            "wav_path": wav_path,
            "segment_start_sec": u["start"],
            "segment_end_sec": u["end"],
            "axis": "backchannel",
            "gt_backchannel_times": gt_bc_times,
        })
    return segments


def _extract_interruption_segments(
    user_utts: list[dict[str, float]],
    moshi_utts: list[dict[str, float]],
    wav_path: str,
    total_dur: float,
    tau_min: float = 3.0,
) -> list[dict[str, Any]]:
    """D_int: X→Y→X→Y pattern where X interrupts Y."""
    all_utts = []
    for u in user_utts:
        all_utts.append({"start": u["start"], "end": u["end"], "speaker": "X"})
    for m in moshi_utts:
        all_utts.append({"start": m["start"], "end": m["end"], "speaker": "Y"})
    all_utts.sort(key=lambda x: x["start"])

    segments = []
    for i in range(len(all_utts) - 3):
        u0, u1, u2, u3 = all_utts[i], all_utts[i + 1], all_utts[i + 2], all_utts[i + 3]
        if (
            u0["speaker"] == "X"
            and u1["speaker"] == "Y"
            and u2["speaker"] == "X"
            and u3["speaker"] == "Y"
        ):
            if u2["start"] < u1["end"]:
                if all(
                    u["end"] - u["start"] >= tau_min
                    for u in [u0, u1, u2, u3]
                ):
                    segments.append({
                        "wav_path": wav_path,
                        "segment_start_sec": u0["start"],
                        "segment_end_sec": u3["end"],
                        "axis": "interruption",
                        "interruption_start_sec": u2["start"],
                        "interrupted_utterance_end_sec": u1["end"],
                    })
    return segments


# ── Sidecar JSON for user text ──────────────────────────────────────

def _load_sidecar_user_texts(
    wav_path: Path,
) -> list[dict[str, Any]]:
    """Load user turn texts + alignments from the sidecar JSON.

    Sidecar format (from generate_synthetic_moshi_training_data.py):
      { "alignments": [...], "metadata": { "dialogue": { "turns": [...] } } }
    Returns list of {"text": str, "start": float, "end": float, "speaker": str}.
    """
    json_path = wav_path.with_suffix(".json")
    if not json_path.is_file():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    alignments = data.get("alignments", [])
    dialogue = data.get("metadata", {}).get("dialogue", {})
    turns = dialogue.get("turns", [])

    result: list[dict[str, Any]] = []
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        if not text:
            continue
        align = alignments[i] if i < len(alignments) else {}
        result.append({
            "text": text,
            "speaker": speaker,
            "start": align.get("start_sec", 0.0),
            "end": align.get("end_sec", 0.0),
        })
    return result


def _find_user_text_for_segment(
    sidecar_turns: list[dict[str, Any]],
    seg_start: float,
    seg_end: float,
) -> str:
    """Find user turn text(s) overlapping with the segment time range."""
    texts = []
    for turn in sidecar_turns:
        if turn["speaker"] not in ("user", "User"):
            continue
        if turn["end"] < seg_start or turn["start"] > seg_end:
            continue
        texts.append(turn["text"])
    return " ".join(texts)


# ── Main extraction pipeline ────────────────────────────────────────

def extract_segments(wav_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract all axis segments from one stereo WAV.

    Channel 0 = moshi (Y), Channel 1 = user (X).
    Attaches user_text from sidecar JSON for LLM judge use.
    """
    moshi_audio = _load_audio_channel(wav_path, 0)
    user_audio = _load_audio_channel(wav_path, 1)
    total_dur = len(user_audio) / SAMPLE_RATE

    moshi_utts = _run_vad(moshi_audio)
    user_utts = _run_vad(user_audio)
    wp = str(wav_path)

    sidecar_turns = _load_sidecar_user_texts(wav_path)

    result = {
        "pause": _extract_pause_segments(user_utts, moshi_utts, wp, total_dur),
        "turn_taking": _extract_turn_taking_segments(user_utts, moshi_utts, wp, total_dur),
        "backchannel": _extract_backchannel_segments(user_utts, moshi_utts, wp, total_dur),
        "interruption": _extract_interruption_segments(user_utts, moshi_utts, wp, total_dur),
    }

    for axis, items in result.items():
        for item in items:
            item["user_text"] = _find_user_text_for_segment(
                sidecar_turns,
                item["segment_start_sec"],
                item["segment_end_sec"],
            )

    return result


def extract_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    max_per_axis: int = 2000,
) -> dict[str, int]:
    """Walk a training manifest JSONL, extract segments, write per-axis JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = ["pause", "turn_taking", "backchannel", "interruption"]
    counts: dict[str, int] = {a: 0 for a in axes}
    buffers: dict[str, list[dict[str, Any]]] = {a: [] for a in axes}

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line.strip())
            wav = Path(row.get("audio_path", row.get("path", "")))
            if not wav.is_file():
                wav = manifest_path.parent / wav
            if not wav.is_file():
                continue
            try:
                segs = extract_segments(wav)
            except Exception as exc:
                print(f"WARN [{line_no}]: skipping {wav}: {exc}")
                continue
            for axis, items in segs.items():
                for item in items:
                    if counts[axis] >= max_per_axis:
                        break
                    buffers[axis].append(item)
                    counts[axis] += 1

            if line_no % 50 == 0:
                print(
                    f"  [{line_no}] "
                    + " ".join(f"{a}={counts[a]}" for a in axes)
                )

            if all(counts[a] >= max_per_axis for a in axes):
                break

    for axis, items in buffers.items():
        out_path = out_dir / f"{axis}_segments.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{axis}: {len(items)} segments -> {out_path}")

    summary = {
        "total_segments": sum(counts.values()),
        "per_axis": counts,
        "manifest": str(manifest_path),
    }
    (out_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Training manifest JSONL (synthetic_moshi_train.jsonl)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output dir for per-axis segment JSONLs")
    parser.add_argument("--max-per-axis", type=int, default=2000,
                        help="Max segments per axis (paper: 2000)")
    args = parser.parse_args()
    extract_from_manifest(args.manifest, args.out_dir, args.max_per_axis)


if __name__ == "__main__":
    main()
