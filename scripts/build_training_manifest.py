#!/usr/bin/env python3
"""training_set/data_stereo/ を走査して synthetic_moshi_train.jsonl を作り直す。

必要になる理由:
    inject_inner_thoughts.py は --out-dir にタグを挿せたサンプルだけを書く。
    さらに --pad-lead-in-sec を使うと音声の尺が変わる。つまり元の manifest は
    「もう存在しないファイル」を指し、「実際とは違う尺」を載せた状態になる。
    prepare_nu_fullft_dataset.py は manifest を入口にするので、タグ入りの
    コーパスにはそれに対応した manifest が要る。

    尺は wav から実測する。元の manifest の値を足し引きして辻褄を合わせる
    より、実ファイルを読む方が確実で、パディングの有無に依存しない。

使い方:
    uv run python scripts/build_training_manifest.py \
        --training-set data/runs/<run>/tts/.../shard_000_conditioned/training_set

    既定では data_stereo/ の wav を並べ、sidecar JSON が無いものは落とす
    （prepare_nu_fullft_dataset.py がどのみち両方を要求するため）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild synthetic_moshi_train.jsonl from the wavs on disk."
    )
    parser.add_argument("--training-set", required=True, type=Path,
                        help="data_stereo/ を含むディレクトリ")
    parser.add_argument("--audio-subdir", default="data_stereo",
                        help="wav と sidecar JSON があるサブディレクトリ")
    parser.add_argument("--out-name", default="synthetic_moshi_train.jsonl",
                        help="書き出す manifest のファイル名")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def wav_duration_sec(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def main() -> int:
    args = parse_args()
    audio_dir = args.training_set / args.audio_subdir
    if not audio_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {audio_dir}")

    rows: list[dict[str, object]] = []
    skipped_no_json = 0
    for wav_path in sorted(audio_dir.glob("*.wav")):
        if not wav_path.with_suffix(".json").is_file():
            skipped_no_json += 1
            continue
        rows.append({
            "path": f"{args.audio_subdir}/{wav_path.name}",
            "duration": round(wav_duration_sec(wav_path), 4),
        })

    total = sum(float(row["duration"]) for row in rows)
    print(f"samples          {len(rows)}")
    print(f"skipped_no_json  {skipped_no_json}")
    print(f"total_hours      {total / 3600:.2f}")
    if rows:
        durations = sorted(float(row["duration"]) for row in rows)
        p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
        print(f"duration p95/max {p95:.1f} / {durations[-1]:.1f} sec")

    if args.dry_run:
        return 0
    out_path = args.training_set / args.out_name
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote            {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
