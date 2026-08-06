#!/usr/bin/env python3
"""
model_id=gold の「推論」。モデルを動かさず、実際の相談員の応答を出力として置く。

eval/run_full_duplex_bench.py が書くのと**同じ形状**の実行ディレクトリを作る。
そうすることで、後段の eval/evaluate_real_response.py は gold を特別扱いせず、
モデルと同じコードで応答速度・応答率・音声品質を測れる。gold のスコアが
そのケースの実質的な上限になる。

    <out-dir>/real_response/<case_id>/seed_0/
      metadata.json      データセットからのコピー
      output.wav         入力長ぶんの無音 + 実際の相談員の応答音声
      output.json        時刻つきテキスト(アノテーションの書き起こし)
      output.meta.json   タイミング

音声は元の 1ch 録音から切り出すので、相談員の応答に User の声が重なって
いれば残る。gold の音声品質はその条件込みの値であり、「人間の声だから満点」
という数字にはならない。

使い方:
    uv run python eval/run_gold_reference.py \\
        --dataset-dir data/eval_sets/real_response \\
        --out-dir eval_runs/real_response/gold/inference
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000  # Moshi 側の出力と揃える(比較時に再サンプルしないで済む)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="gold")
    parser.add_argument("--seeds", default="0",
                        help="モデル側と行数を揃えるためのシード列。"
                             "gold は決定的なので中身は同じになる")
    parser.add_argument("--tail-sec", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_reply_audio(
    metadata: dict, sample_rate: int
) -> tuple[np.ndarray | None, float | None, float | None]:
    """相談員の応答音声を元録音から切り出して返す。

    3 つめの戻り値は、切り出した音声の先頭が User の発話終了から何秒後かを
    示す。切り出しには語頭が欠けないようマージンを付けているので、その分だけ
    先頭に無音が入る。配置にはこの値を使うこと。annotated な latency をその
    まま使うと、マージンぶん(0.08 秒)遅く鳴り始め、測った応答速度が
    アノテーションとずれる。
    """
    reference = metadata.get("human_reference") or {}
    span = reference.get("audio")
    if not span:
        return None, None, None
    wav_path = Path((metadata.get("source") or {}).get("wav", ""))
    if not wav_path.is_file():
        raise SystemExit(
            f"元の録音が見つかりません: {wav_path}\n"
            "  データセットを作った時とパスが変わっている場合は作り直してください。"
        )
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    lo = max(0, int(span["start_sec"] * sr))
    hi = min(audio.size, int(span["end_sec"] * sr))
    if hi <= lo:
        return None, None, None
    clip = audio[lo:hi]
    if sr != sample_rate:
        import librosa
        clip = librosa.resample(clip, orig_sr=sr, target_sr=sample_rate)
    user_end = float((metadata.get("source") or {}).get("user_end_sec", 0.0))
    clip_offset = lo / sr - user_end
    return (
        clip.astype(np.float32),
        reference.get("response_latency_sec"),
        clip_offset,
    )


def load_backchannel_audio(
    metadata: dict, sample_rate: int
) -> list[tuple[float, np.ndarray]]:
    """User の発話中に相談員が打った相槌を、元録音から切り出して返す。

    返すのは (output.wav の時計での開始秒, 波形) の列。これを置かないと gold は
    相槌軸で 0 点になり、実際には打っている相談員を「打っていない」と評価する
    ことになる。
    """
    items = metadata.get("backchannel_gt") or []
    if not items:
        return []
    wav_path = Path((metadata.get("source") or {}).get("wav", ""))
    if not wav_path.is_file():
        raise SystemExit(
            f"元の録音が見つかりません: {wav_path}\n"
            "  データセットを作った時とパスが変わっている場合は作り直してください。"
        )
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    out: list[tuple[float, np.ndarray]] = []
    for item in items:
        lo = max(0, int(float(item["abs_start_sec"]) * sr))
        hi = min(audio.size, int(float(item["abs_end_sec"]) * sr))
        if hi <= lo:
            continue
        clip = audio[lo:hi]
        if sr != sample_rate:
            import librosa
            clip = librosa.resample(clip, orig_sr=sr, target_sr=sample_rate)
        out.append((float(item["start_sec"]), clip.astype(np.float32)))
    return out


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]

    run_config = {
        "model_id": args.model_id,
        "system": "gold",
        "note": (
            "実際の相談員の応答をそのまま出力として置いたもの。モデルは動かして"
            "いない。指標の実質的な上限として読む。"
        ),
        "dataset_dir": str(dataset_dir),
        "seeds": seeds,
        "tail_sec": args.tail_sec,
        "sample_rate": args.sample_rate,
        "protocol": manifest.get("protocol"),
        "selected_case_count": len(samples),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    written = 0
    silent = 0
    for sample in samples:
        source_dir = dataset_dir / sample["path"]
        metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        input_sec = float((metadata.get("source") or {}).get("input_duration_sec", 0.0))
        reply, latency, clip_offset = load_reply_audio(metadata, args.sample_rate)

        total_sec = input_sec + args.tail_sec
        if reply is not None and clip_offset is not None:
            total_sec = max(
                total_sec,
                input_sec + max(0.0, clip_offset) + len(reply) / args.sample_rate,
            )
        aligned = np.zeros(int(total_sec * args.sample_rate) + 1, dtype=np.float32)

        # User の発話中の相槌。入力再生と同じ時計なので、そのまま置く。
        for bc_start, bc_audio in load_backchannel_audio(metadata, args.sample_rate):
            lo = max(0, int(bc_start * args.sample_rate))
            hi = min(aligned.size, lo + bc_audio.size)
            if hi > lo:
                aligned[lo:hi] = bc_audio[: hi - lo]

        start_sec = None
        if reply is not None:
            # 相談員が食い気味に応答した場合(latency < 0)、モデル側の出力は
            # 入力終了より前には置けない。0 で止めて、生の値は
            # annotated_latency_sec として別に記録する。
            start_sec = input_sec + max(0.0, clip_offset or 0.0)
            lo = int(start_sec * args.sample_rate)
            hi = min(aligned.size, lo + reply.size)
            if hi > lo:
                aligned[lo:hi] = reply[: hi - lo]
        else:
            silent += 1

        for seed in seeds:
            trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            output_wav = trial_dir / "output.wav"
            if output_wav.exists() and not args.overwrite:
                continue
            shutil.copy2(source_dir / "metadata.json", trial_dir / "metadata.json")
            shutil.copy2(source_dir / "input.wav", trial_dir / "input.wav")
            sf.write(output_wav, aligned, args.sample_rate)

            segments = (metadata.get("human_reference") or {}).get("segments") or []
            backchannel_chunks = [
                {
                    "timestamp": [
                        round(float(item["start_sec"]), 4),
                        round(float(item["end_sec"]), 4),
                    ],
                    "text": item.get("text", ""),
                }
                for item in (metadata.get("backchannel_gt") or [])
            ]
            (trial_dir / "output.json").write_text(
                json.dumps(
                    {
                        "text": "".join(s.get("text", "") for s in segments),
                        "chunks": backchannel_chunks + [
                            {
                                "timestamp": [
                                    round(input_sec + max(0.0, s["start_sec"]
                                          - metadata["source"]["user_end_sec"]), 4),
                                    round(input_sec + max(0.0, s["end_sec"]
                                          - metadata["source"]["user_end_sec"]), 4),
                                ],
                                "text": s.get("text", ""),
                            }
                            for s in segments
                        ],
                    },
                    ensure_ascii=False, indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            (trial_dir / "output.meta.json").write_text(
                json.dumps(
                    {
                        "model_id": args.model_id,
                        "task": sample["task"],
                        "case_id": sample["id"],
                        "seed": seed,
                        "variant": "input",
                        "sample_rate": args.sample_rate,
                        "input_duration_sec": round(input_sec, 4),
                        "audible_response_start_sec": (
                            round(start_sec, 4) if start_sec is not None else None
                        ),
                        "annotated_latency_sec": latency,
                        "language": "ja",
                        "expected_behavior": metadata.get("expected_behavior"),
                    },
                    ensure_ascii=False, indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            written += 1

    print(f"[gold] {written} 件を書き出しました -> {out_dir}")
    if silent:
        print(
            f"[gold] うち {silent} ケースは相談員も応答していません"
            "(応答率がその分下がります)。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
