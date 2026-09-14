#!/usr/bin/env python3
"""KABURI が置いた相槌の位置に、作り置きした相槌の音を差し込む。

バンク方式の最小の実験。KABURI に「いつ相槌を打つか」だけをやらせ、「どんな音
で打つか」は Qwen3-TTS で先に大量に作っておいたものから引いてくる。

  KABURI  : 配置(間・かぶり) -- gap model が決める。ここは触らない
  バンク  : 音 -- scripts/qwen3_tts_diversity_probe.py が作った wav 群
  ここ    : 置き換え -- 左チャンネル(moshi)の相槌の区間を差し替える

合成し直すのではなく、出来上がったステレオ WAV の該当区間だけを書き換える。
配置も相手の発話も 1 サンプルも動かないので、差し替え前後は直接 A/B できる。

  uv run python scripts/splice_aizuchi_bank.py \\
      --training-dir data/runs/smoke/<kaburi_run>/shard_000/training_set \\
      --bank-dir data/runs/diversity/<sweep_run> \\
      --out-dir data/runs/smoke/<kaburi_run>_bank/shard_000/training_set

バンクは 1 語しか持っていなくてよい。実録音 116 件では「うん」だけで 28 件、
上位 4 語で 76% を占めていて、いま合成に使っている語彙とはほとんど重ならない。
差し替え後の対話は「聞き手はうんとしか言わない」ものになるが、頻度の実態には
むしろ近い。

差し替え先の選び方は尺で決める。KABURI が空けた区間の長さに近い順に並べ、
上位 --match-top-k 本から 1 本引く(毎回いちばん近いものを取ると同じ音ばかりが
並ぶため)。尺が合わないバンクしか無ければその旨を数え、無理には詰めない。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alignment_words import (  # noqa: E402
    get_segmenter_name,
    split_utterance_alignments,
)
from scripts.qwen3_tts_diversity_probe import (  # noqa: E402
    measure,
    read_wav,
    split_outliers,
)

# 左が moshi(聞き手)、右が user。KABURI の出力もこの並び。
MAIN_LABEL = "SPEAKER_MAIN"
LEFT_CHANNEL = 0
# 継ぎ目のプチッを消すだけの長さ。相槌の立ち上がりを鈍らせない範囲で。
FADE_SEC = 0.005
# 相槌とみなす文字数の上限。「そうなんですね」で 7 文字なので、既定はそれより
# 短いものだけ。--aizuchi-max-chars で動かせる。
DEFAULT_MAX_CHARS = 6


def load_bank(
    bank_dir: Path,
    text: str | None,
    temperature: float | None,
    drop_outliers: bool,
) -> list[dict[str, Any]]:
    """バンクの wav を読んで [{signal, sample_rate, speech_sec, ...}] にする。

    区間の頭と尻の無音は落としておく。KABURI が空けた位置に「音が鳴り始める
    瞬間」を合わせたいので、ファイル先頭ではなく発話開始を基準にする。
    """
    samples_path = bank_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise SystemExit(f"バンクに samples.jsonl がありません: {samples_path}")

    entries: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if text is not None and entry.get("text") != text:
                continue
            if temperature is not None and entry.get("temperature") != temperature:
                continue
            entries.append(entry)
    if not entries:
        raise SystemExit(
            f"条件に合うバンクの音がありません (text={text!r}, temperature={temperature!r})"
        )

    if drop_outliers:
        durations = [float(e["measure"]["speech_sec"]) for e in entries]
        _kept, long_tail, short_tail = split_outliers(durations)
        dropped = set(long_tail) | set(short_tail)
        before = len(entries)
        entries = [
            e for e in entries if float(e["measure"]["speech_sec"]) not in dropped
        ]
        if before != len(entries):
            print(f"[bank] 尺の外れを {before - len(entries)} 本除外しました")

    clips: list[dict[str, Any]] = []
    for entry in entries:
        signal, sample_rate = read_wav(bank_dir / entry["wav"])
        stats = measure(signal, sample_rate)
        start = int(round(stats["lead_silence_sec"] * sample_rate))
        end = signal.size - int(round(stats["trail_silence_sec"] * sample_rate))
        trimmed = signal[start:max(start + 1, end)]
        if trimmed.size == 0:
            continue
        clips.append(
            {
                "wav": entry["wav"],
                "text": entry["text"],
                "temperature": entry.get("temperature"),
                "signal": trimmed,
                "sample_rate": sample_rate,
                "speech_sec": trimmed.size / sample_rate,
            }
        )
    if not clips:
        raise SystemExit("バンクの音を 1 本も読めませんでした")
    print(
        f"[bank] {len(clips)} 本 / 尺 "
        f"{min(c['speech_sec'] for c in clips):.2f}-{max(c['speech_sec'] for c in clips):.2f}s"
    )
    return clips


def is_backchannel(text: str, max_chars: int, vocab: set[str] | None) -> bool:
    stripped = text.strip().strip("。、．，!?！？ 　")
    if not stripped:
        return False
    if vocab is not None:
        return stripped in vocab
    return len(stripped) <= max_chars


def backchannel_gaps(
    placements: Sequence[dict[str, Any]], max_chars: int, vocab: set[str] | None
) -> list[float]:
    """KABURI の配置から、相槌ごとの「相手の発話の終わりからの差」を取り出す。

    上流の配置式は start_i = max(0, prev_any_end + gap_i)。相槌について意味が
    あるのは「相手が言い終えてから何秒後に(あるいは何秒前に)反応したか」なので、
    直前に始まっている相手の発話の終わりを基準に測る。負なら食い込み。

    この差だけを既存の音声へ移せば、KABURI の音は 1 サンプルも要らない --
    使うのはタイミングだけ、という元の狙いそのもの。
    """
    rows = sorted(placements, key=lambda row: row["start_sec"])
    gaps: list[float] = []
    for row in rows:
        if row["label"] != MAIN_LABEL or not is_backchannel(
            str(row["text"]), max_chars, vocab
        ):
            continue
        start = float(row["start_sec"])
        prior = [
            float(other["end_sec"])
            for other in rows
            if other["label"] != MAIN_LABEL and float(other["start_sec"]) < start
        ]
        gaps.append(round(start - (max(prior) if prior else 0.0), 4))
    return gaps


def retime_backchannels(
    rows: Sequence[Sequence[Any]],
    gaps: Sequence[float],
    max_chars: int,
    vocab: set[str] | None,
    duration_sec: float,
) -> dict[int, float]:
    """既存の音声の相槌に KABURI の差を当てはめ、{行番号: 新しい開始秒} を返す。

    対応は「相槌の出てくる順番」で取る。時刻順に並べた k 番目の相槌に k 番目の
    差を当てる。本数が合わないときは何も返さない -- 途中でずれた対応を黙って
    通すより、その対話を飛ばした方がよい。
    """
    indexed = sorted(range(len(rows)), key=lambda i: rows[i][1][0])
    targets = [
        i
        for i in indexed
        if rows[i][2] == MAIN_LABEL and is_backchannel(str(rows[i][0]), max_chars, vocab)
    ]
    if len(targets) != len(gaps):
        return {}

    new_starts: dict[int, float] = {}
    for position, row_index in enumerate(targets):
        start = float(rows[row_index][1][0])
        prior = [
            float(rows[i][1][1])
            for i in indexed
            if rows[i][2] != MAIN_LABEL and float(rows[i][1][0]) < start
        ]
        base = max(prior) if prior else 0.0
        moved = base + float(gaps[position])
        new_starts[row_index] = min(max(0.0, moved), max(0.0, duration_sec - 0.01))
    return new_starts


def pick_clip(
    clips: Sequence[dict[str, Any]],
    target_sec: float,
    top_k: int,
    rng: random.Random,
) -> dict[str, Any]:
    """尺の近い順に top_k 本を出し、その中から引く。"""
    ordered = sorted(clips, key=lambda clip: abs(clip["speech_sec"] - target_sec))
    return rng.choice(ordered[: max(1, top_k)])


def fade(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    n = min(int(round(FADE_SEC * sample_rate)), signal.size // 2)
    if n <= 0:
        return signal
    ramp = np.linspace(0.0, 1.0, n)
    out = signal.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def splice_dialogue(
    stereo: np.ndarray,
    sample_rate: int,
    rows: Sequence[Sequence[Any]],
    clips: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    vocab: set[str] | None,
    new_starts: dict[int, float] | None = None,
) -> tuple[np.ndarray, list[list[Any]], list[dict[str, Any]]]:
    """左チャンネルの相槌区間を差し替え、新しい波形と alignments を返す。

    new_starts が与えられたら、消す位置(元の相槌があったところ)と置く位置
    (KABURI の差を当てはめたところ)を分ける。
    """
    out = stereo.copy()
    new_rows: list[list[Any]] = []
    swaps: list[dict[str, Any]] = []
    new_starts = new_starts or {}

    # 消すのが先。移動させると、後の相槌の元位置に先の相槌を書いたあとで
    # 消してしまうことがある。
    for row_index, (text, (start, end), label) in enumerate(rows):
        if label == MAIN_LABEL and is_backchannel(
            str(text), args.aizuchi_max_chars, vocab
        ):
            erase_from = min(out.shape[-1], int(round(float(start) * sample_rate)))
            erase_to = min(out.shape[-1], int(round(float(end) * sample_rate)))
            out[LEFT_CHANNEL, erase_from:erase_to] = 0.0

    for row_index, (text, (start, end), label) in enumerate(rows):
        if label != MAIN_LABEL or not is_backchannel(
            str(text), args.aizuchi_max_chars, vocab
        ):
            new_rows.append([text, [start, end], label])
            continue

        slot_sec = float(end) - float(start)
        moved_to = new_starts.get(row_index)
        place_at = float(start) if moved_to is None else float(moved_to)
        clip = pick_clip(clips, slot_sec, args.match_top_k, rng)
        signal = clip["signal"]
        if clip["sample_rate"] != sample_rate:
            # バンクと対話のサンプリングレートが違うなら線形で合わせる。どちらも
            # 24kHz の想定だが、黙って音程がずれるよりは伸縮したと言う方がよい。
            n = int(round(signal.size * sample_rate / clip["sample_rate"]))
            signal = np.interp(
                np.linspace(0.0, signal.size - 1, max(1, n)),
                np.arange(signal.size),
                signal,
            )

        slot_begin = int(round(place_at * sample_rate))
        if slot_begin >= out.shape[-1]:
            new_rows.append([text, [start, end], label])
            continue
        # 音量合わせの基準は元の相槌。もう消してあるので、消す前に取った
        # チャンネル全体の控えから読む。
        original_begin = int(round(float(start) * sample_rate))
        original = stereo[
            LEFT_CHANNEL,
            min(stereo.shape[-1], original_begin) : min(
                stereo.shape[-1], int(round(float(end) * sample_rate))
            ),
        ]

        placed = fade(signal.astype(np.float64), sample_rate)
        if args.gain == "match" and original.size:
            original_rms = float(np.sqrt(np.mean(np.square(original))))
            placed_rms = float(np.sqrt(np.mean(np.square(placed))))
            if placed_rms > 0 and original_rms > 0:
                placed = placed * (original_rms / placed_rms)
        peak = float(np.max(np.abs(placed))) if placed.size else 0.0
        if peak > 1.0:
            placed = placed / peak

        # 消すのは上で済ませてある。ここは置くだけ。
        stop = min(out.shape[-1], slot_begin + placed.size)
        written = stop - slot_begin
        # 相手の発話に食い込むのは相槌として自然なので、区間をはみ出しても
        # 切らずにそのまま置く。切るのは対話の末尾に当たったときだけ。
        out[LEFT_CHANNEL, slot_begin:stop] += placed[:written]

        placed_start = slot_begin / sample_rate
        placed_end = round(placed_start + written / sample_rate, 4)
        new_rows.append([clip["text"], [round(placed_start, 4), placed_end], label])
        swaps.append(
            {
                "original_text": text,
                "bank_text": clip["text"],
                "bank_wav": clip["wav"],
                "bank_temperature": clip["temperature"],
                "slot_sec": round(slot_sec, 4),
                "placed_sec": round(written / sample_rate, 4),
                "length_error_sec": round(written / sample_rate - slot_sec, 4),
                "truncated": bool(written < placed.size),
                "original_start_sec": round(float(start), 4),
                "start_sec": round(placed_start, 4),
                "moved_sec": round(placed_start - float(start), 4),
            }
        )

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    new_rows.sort(key=lambda row: row[1][0])
    return out, new_rows, swaps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument(
        "--placement-dir",
        type=Path,
        default=None,
        help=(
            "generate_kaburi_tts_data.py --placement-only の出力。渡すと、相槌を"
            " KABURI が置いた間に合わせて動かしてから差し替える（KABURI の音は"
            "使わないので GPU 不要）"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bank-text", default="うん", help="空文字でバンク全体を使う")
    parser.add_argument("--bank-temperature", type=float, default=None)
    parser.add_argument("--keep-bank-outliers", action="store_true")
    parser.add_argument("--aizuchi-max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--aizuchi-vocab-file", type=Path, default=None, help="1 行 1 語。文字数判定の代わり"
    )
    parser.add_argument("--match-top-k", type=int, default=5)
    parser.add_argument("--gain", choices=("match", "none"), default="match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="対話数の上限(0 で全部)")
    args = parser.parse_args(argv)

    import torch
    import torchaudio

    json_paths = sorted(args.training_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"training_set に JSON がありません: {args.training_dir}")
    if args.limit > 0:
        json_paths = json_paths[: args.limit]

    vocab: set[str] | None = None
    if args.aizuchi_vocab_file is not None:
        vocab = {
            line.strip()
            for line in args.aizuchi_vocab_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    clips = load_bank(
        args.bank_dir,
        args.bank_text or None,
        args.bank_temperature,
        not args.keep_bank_outliers,
    )
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "bank_swaps.jsonl"
    report_path.unlink(missing_ok=True)

    total_swaps = 0
    retimed_total = 0
    errors: list[float] = []
    moves: list[float] = []
    for json_path in json_paths:
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.is_file():
            print(f"[skip] wav がありません: {wav_path}")
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = payload.get("alignments_utterance") or []
        if not rows:
            print(f"[skip] alignments_utterance が空です: {json_path.name}")
            continue

        wav, sample_rate = torchaudio.load(str(wav_path))
        stereo = wav.numpy().astype(np.float64)

        new_starts: dict[int, float] = {}
        if args.placement_dir is not None:
            placement_path = args.placement_dir / f"{json_path.stem}.placement.json"
            if not placement_path.is_file():
                print(f"[skip] 配置がありません: {placement_path.name}")
                continue
            placement = json.loads(placement_path.read_text(encoding="utf-8"))
            gaps = backchannel_gaps(
                placement.get("placements", []), args.aizuchi_max_chars, vocab
            )
            duration = stereo.shape[-1] / sample_rate
            new_starts = retime_backchannels(
                rows, gaps, args.aizuchi_max_chars, vocab, duration
            )
            if not new_starts and gaps:
                # 本数が合わない = 対応が取れない。黙ってずらすより飛ばす。
                print(
                    f"[skip] 相槌の本数が配置と合いません: {json_path.name} "
                    f"(KABURI {len(gaps)} 箇所)"
                )
                continue
            retimed_total += len(new_starts)

        spliced, new_rows, swaps = splice_dialogue(
            stereo, sample_rate, rows, clips, args, rng, vocab, new_starts
        )
        if not swaps:
            print(f"[skip] 相槌が見つかりません: {json_path.name}")

        alignments, _stats = split_utterance_alignments(new_rows)
        payload["alignments"] = alignments
        payload["alignments_utterance"] = new_rows
        metadata = payload.setdefault("metadata", {})
        metadata["alignments_word_split"] = {
            "version": 1,
            "method": "char-proportional",
            "segmenter": get_segmenter_name(),
        }
        # 置き換えたのは音と alignments だけ。metadata.dialogue.turns は
        # 「何を言わせるつもりだったか」の記録としてそのまま残してある。
        metadata["aizuchi_bank"] = {
            "source": str(args.bank_dir),
            "bank_text": args.bank_text or "(all)",
            "bank_temperature": args.bank_temperature,
            "n_clips": len(clips),
            "match": "duration",
            "match_top_k": args.match_top_k,
            "gain": args.gain,
            "seed": args.seed,
            "n_swapped": len(swaps),
            "n_retimed": len(new_starts),
            "placement_dir": str(args.placement_dir) if args.placement_dir else None,
            "turns_text_rewritten": False,
        }

        out_wav = args.out_dir / wav_path.name
        torchaudio.save(
            str(out_wav),
            torch.from_numpy(spliced).to(torch.float32),
            sample_rate,
            channels_first=True,
        )
        (args.out_dir / json_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        with report_path.open("a", encoding="utf-8") as handle:
            for swap in swaps:
                handle.write(
                    json.dumps({"stem": json_path.stem, **swap}, ensure_ascii=False) + "\n"
                )
        total_swaps += len(swaps)
        errors.extend(swap["length_error_sec"] for swap in swaps)
        moves.extend(swap["moved_sec"] for swap in swaps if swap["moved_sec"])
        print(f"[ok] {json_path.stem}: {len(swaps)} 箇所を差し替え -> {out_wav.name}")

    print()
    print(f"対話 {len(json_paths)} 本 / 差し替え {total_swaps} 箇所")
    if errors:
        absolute = [abs(value) for value in errors]
        print(
            f"尺のずれ: 中央値 {statistics.median(absolute):.3f}s / "
            f"最大 {max(absolute):.3f}s / 平均 {statistics.fmean(errors):+.3f}s"
        )
        over = sum(1 for value in errors if value > 0)
        print(f"  うち区間より長くなった: {over}/{len(errors)} (相手に食い込む方向)")
    if moves:
        absolute = [abs(value) for value in moves]
        earlier = sum(1 for value in moves if value < 0)
        print(
            f"KABURI の間へ移動: {retimed_total} 箇所 / 中央値 "
            f"{statistics.median(absolute):.3f}s / 最大 {max(absolute):.3f}s"
        )
        print(f"  うち前倒し(食い込む方向): {earlier}/{len(moves)}")
    print(f"出力: {args.out_dir}")
    print(f"内訳: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
