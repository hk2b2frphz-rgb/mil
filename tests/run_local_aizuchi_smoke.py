#!/usr/bin/env python3
"""相槌(あいづち)台本のローカル・スモークテスト。

GPU も TTS モデルも使わず、FakeTTS で
  load_dialogues_from_jsonl -> validate_duplex_dialogue -> build_segments -> render_stereo
までを通し、相槌が前の発話に正しくオーバーラップしているかを確認する。

実行:
  python tests/run_local_aizuchi_smoke.py
  python tests/run_local_aizuchi_smoke.py --write-wav out_dir   # 無音/ノイズ placeholder WAV も書く
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_qwen3_tts_data import (  # noqa: E402
    Dialogue,
    DialogueTurn,
    build_segments,
    load_dialogues_from_jsonl,
    optional_float,
    render_stereo,
    validate_duplex_dialogue,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "aizuchi_dialogues.jsonl"
SAMPLE_RATE = 24000


class FakeTTS:
    """テキスト長に比例した長さのモノラル正弦波を返すだけのダミー TTS。"""

    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE

    def synthesize(self, text, speaker_role, instruct=None, speaker_override=None):
        dur = max(0.4, 0.18 * len(text))  # おおよそ 1 文字 0.18 秒
        n = int(round(dur * self.sample_rate))
        t = np.arange(n) / self.sample_rate
        freq = 180.0 if speaker_role == "moshi" else 240.0
        return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def to_dialogue(tmpl: dict) -> Dialogue:
    turns: list[DialogueTurn] = []
    for t in tmpl["turns"]:
        if t["speaker"] == "silence":
            turns.append(DialogueTurn(speaker="silence", duration_sec=float(t.get("duration_sec", 2.0)), note=t.get("note")))
            continue
        turns.append(DialogueTurn(
            speaker=t["speaker"],
            text=t["text"],
            emotion=t.get("emotion"),
            timing=str(t.get("timing") or "sequential"),
            start_after_previous_start_sec=optional_float(t.get("start_after_previous_start_sec")),
            truncate_previous_after_sec=optional_float(t.get("truncate_previous_after_sec")),
            gain=max(0.0, float(optional_float(t.get("gain"), 1.0) or 0.0)),
            voice_role=str(t.get("voice_role") or "") or None,
            event=str(t.get("event") or "") or None,
        ))
    return Dialogue(
        id=tmpl["id"], category=tmpl["category"], risk_level=tmpl["risk_level"],
        title=tmpl["title"], turns=turns, duplex_task=str(tmpl.get("duplex_task") or "") or None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-wav", type=Path, default=None, help="placeholder WAV 出力先")
    args = ap.parse_args()

    templates = load_dialogues_from_jsonl(FIXTURE)
    print(f"loaded {len(templates)} dialogues from {FIXTURE.name}\n")

    tts = FakeTTS()
    failures = 0

    for tmpl in templates:
        errors = validate_duplex_dialogue(tmpl)
        dlg = to_dialogue(tmpl)
        segments, silences = build_segments(dlg, tts, lead_in_sec=0.3, gap_sec=0.2)
        stereo = render_stereo(segments, tts.sample_rate)
        dur = stereo.shape[-1] / tts.sample_rate

        status = "OK " if not errors else "FAIL"
        if errors:
            failures += 1
        print(f"[{status}] {dlg.id}  task={dlg.duplex_task}  dur={dur:.2f}s  segs={len(segments)}")
        if errors:
            for e in errors:
                print(f"        validation: {e}")

        # 相槌(overlap)の検出を表示
        for i, seg in enumerate(segments):
            tag = ""
            if seg.event in ("model_backchannel", "user_backchannel"):
                # 直前 seg との時間的重なりを確認
                prev = segments[i - 1] if i > 0 else None
                overlap = prev is not None and seg.start_sec < prev.end_sec
                tag = f"  <-- 相槌 event={seg.event} overlap_prev={overlap}"
            print(f"        seg{i} {seg.label:13s} [{seg.start_sec:5.2f}, {seg.end_sec:5.2f}] {seg.text[:18]!r}{tag}")
        print()

        if args.write_wav:
            try:
                import sphn  # type: ignore
                args.write_wav.mkdir(parents=True, exist_ok=True)
                out = args.write_wav / f"{dlg.id}.wav"
                sphn.write_wav(str(out), stereo.astype(np.float32), tts.sample_rate)
                print(f"        wrote {out}")
            except ImportError:
                print("        (sphn 未インストール: WAV 書き出しスキップ)")

    print("=" * 60)
    if failures:
        print(f"RESULT: {failures} dialogue(s) FAILED validation")
        return 1
    print(f"RESULT: all {len(templates)} dialogues passed validation + stereo render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
