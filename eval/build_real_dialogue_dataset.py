#!/usr/bin/env python3
"""
実対話(分離済み)から Full-Duplex-Bench 形状の評価データセットを組み立てる。

入力は scripts/separate_real_dialogue.py の出力ディレクトリ。人手で
  1) counselor_A.txt または counselor_B.txt を置いて相談員側を指定
  2) <話者>_segments_review.tsv の keep 列を 0 にして不要セグメントを間引く
を済ませてあることを前提にする。

出力は eval/build_full_duplex_ja_dataset.py と同じディレクトリ規約
(<task>/<id>/{input.wav, metadata.json, input.txt} + manifest.json)なので、
eval/run_full_duplex_bench.py と eval/evaluate_full_duplex_ja.py を無改造で
そのまま使える。

合成 FDB-JA との決定的な違いが 2 つある。

1. 開ループである
   入力に入るのはユーザー側の実音声だけで、相談員側はモデルが置き換わる。
   実際の相談員が何を言ったかはモデルに伝わらないので、文脈区間でモデルは
   自分で相談員役を喋りながら進む。評価点に着く頃には会話履歴が実際の対話と
   別物になっている。したがって重畳を前提とする指標は「その瞬間モデルが実際に
   喋っていたか」(評価器の overlap_achieved)で条件付けて読む必要がある。

2. v1.5 プロトコルを名乗らない
   静的重畳プロトコルは統制条件を前提にしており実対話は満たさない。manifest の
   protocol.name を専用の名前にし、run_full_duplex_bench.py には
   --require-v15-profile を渡さない運用にする。合成 FDB-JA のスコアと同じ表に
   並べてはいけない。

metadata には評価器が使う task / user_segments / events に加えて、
  - human_reference : その窓で相談員が実際に返した発話(LLM-as-a-judge の参照)
  - backchannel_gt  : その窓で相談員が実際に打った相槌(時刻・種類つき)
を入れる。前者は eval/pack_real_dialogue_judge_input.py が、後者は
eval/evaluate_real_backchannel.py が読む。

使い方:
    uv run python eval/build_real_dialogue_dataset.py \\
        --analysis-dir data/real_dialogue/pilot/対話1 \\
        --out-dir data/real_dialogue_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_NAME = "Real dialogue replay (open-loop), FDB-JA metric definitions"

TASKS = (
    "backchannel",
    "pause_handling",
    "smooth_turn_taking",
    "user_backchannel",
    "user_interruption",
)


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str = ""
    overlap_sec: float = 0.0
    is_aizuchi: bool = False
    aizuchi_labels: list[str] = field(default_factory=list)
    index: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--analysis-dir", type=Path, nargs="+", required=True,
                        help="separate_real_dialogue.py の出力ディレクトリ(複数可)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="出力 WAV のサンプリングレート(Moshi は 24000)")
    parser.add_argument("--context-sec", type=float, default=30.0,
                        help="評価点の手前に付ける文脈の長さ。0 で文脈なし")
    parser.add_argument("--tail-sec", type=float, default=0.0,
                        help="評価点の後ろに残す無音の長さ(応答を待つ時間は "
                             "run_full_duplex_bench.py の --tail-sec が別途足す)")
    parser.add_argument("--tasks", default="all",
                        help=f"カンマ区切り、または all。既知: {','.join(TASKS)}")
    parser.add_argument("--max-cases-per-task", type=int, default=None)
    parser.add_argument("--pause-min-sec", type=float, default=0.8,
                        help="ユーザーターン内のこれ以上の無音を pause とみなす")
    parser.add_argument("--backchannel-min-turn-sec", type=float, default=6.0,
                        help="backchannel タスクに使うユーザー長発話の下限")
    parser.add_argument("--turn-merge-gap-sec", type=float, default=0.4,
                        help="同一話者の連続セグメントをこの間隔未満なら 1 発話に結合")
    parser.add_argument("--no-review-filter", action="store_true",
                        help="review TSV の keep 列を無視してすべて使う")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_counselor(analysis_dir: Path) -> str:
    """counselor_<話者>.txt から相談員側の話者ラベルを決める。

    取り違えると入力側と参照側が丸ごと入れ替わり、指標は普通に出るのに結果は
    全部無意味になる。静かに通してはいけないので、0 個でも 2 個でも落とす。
    """
    markers = sorted(p.name for p in analysis_dir.glob("counselor_*.txt"))
    if not markers:
        raise SystemExit(
            f"{analysis_dir}: counselor_A.txt / counselor_B.txt が見つかりません。\n"
            "相談員側の話者を決めて、その名前の空ファイルを置いてください。"
        )
    if len(markers) > 1:
        raise SystemExit(
            f"{analysis_dir}: 相談員マーカーが複数あります: {markers}\n"
            "どちらか一方だけを残してください。"
        )
    return markers[0][len("counselor_"): -len(".txt")]


def load_timeline(analysis_dir: Path) -> list[Turn]:
    path = analysis_dir / "timeline_separated.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"{path} がありません。先に scripts/separate_real_dialogue.py を実行してください。"
        )
    turns: list[Turn] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns.append(
                Turn(
                    speaker=str(row["speaker"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    text=str(row.get("text", "")),
                    overlap_sec=float(row.get("overlap_sec", 0.0)),
                    is_aizuchi=bool(row.get("is_aizuchi", False)),
                    aizuchi_labels=list(row.get("aizuchi_labels", [])),
                    index=index,
                )
            )
    turns.sort(key=lambda t: t.start)
    return turns


def load_keep_filter(analysis_dir: Path, speaker: str) -> set[int] | None:
    """review TSV の keep 列から、残すセグメントの index 集合を作る。

    TSV が無ければ None(=全採用)。人手で間引く前に流してしまった場合に
    黙って全件通ると気づけないので、呼び出し側で警告する。
    """
    path = analysis_dir / f"{speaker}_segments_review.tsv"
    if not path.is_file():
        return None
    keep: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row.get("keep", "1")).strip() in {"1", "true", "True"}:
                try:
                    keep.add(int(row["index"]))
                except (KeyError, ValueError):
                    continue
    return keep


def merge_turns(turns: list[Turn], gap: float) -> list[Turn]:
    """同一話者の連続セグメントを 1 発話へ結合する。

    ダイアライゼーションは息継ぎや短い無音で切るので、1 発話が細かく割れる。
    評価点の切り出しは発話単位で行いたいのでここで粗くする。相槌は結合すると
    消えてしまうため、相槌セグメントは結合対象から外す。
    """
    merged: list[Turn] = []
    for turn in turns:
        if (
            merged
            and not turn.is_aizuchi
            and not merged[-1].is_aizuchi
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end < gap
        ):
            last = merged[-1]
            last.end = turn.end
            last.text = (last.text + turn.text).strip()
            last.overlap_sec += turn.overlap_sec
            continue
        merged.append(
            Turn(
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                overlap_sec=turn.overlap_sec,
                is_aizuchi=turn.is_aizuchi,
                aizuchi_labels=list(turn.aizuchi_labels),
                index=turn.index,
            )
        )
    return merged


def overlaps(a: Turn, b: Turn) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def resample_linear(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = pcm.size / source_rate
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.arange(pcm.size, dtype=np.float64) / source_rate
    target_x = np.arange(target_len, dtype=np.float64) / target_rate
    return np.interp(target_x, source_x, pcm).astype(np.float32)


def detect_cases(
    turns: list[Turn],
    user: str,
    counselor: str,
    keep: set[int] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """タイムラインを走査して、タスクごとの評価点を拾う。

    ここで決めるのは「窓の範囲」と「評価点の意味づけ」だけ。音声の切り出しと
    metadata 整形は呼び出し側で行う。
    """
    cases: list[dict[str, Any]] = []
    user_turns = [t for t in turns if t.speaker == user]
    counselor_turns = [t for t in turns if t.speaker == counselor]

    def kept(turn: Turn) -> bool:
        return keep is None or turn.index in keep

    # --- backchannel: ユーザーの長い発話。相談員が実際に打った相槌が正解になる ---
    for turn in user_turns:
        if turn.is_aizuchi or turn.duration < args.backchannel_min_turn_sec:
            continue
        if not kept(turn):
            continue
        gt = [
            c for c in counselor_turns
            if c.is_aizuchi and overlaps(c, turn) > 0.0
        ]
        cases.append(
            {
                "task": "backchannel",
                "eval_start": turn.start,
                "eval_end": turn.end,
                "user_turns": [turn],
                "backchannel_gt": gt,
                "expected_behavior":
                    "ユーザーが話し続けている間、話を奪わず自然な相槌で受ける。",
            }
        )

    # --- pause_handling: ユーザー発話の途中に空く無音 ---
    for prev, cur in zip(user_turns, user_turns[1:]):
        gap = cur.start - prev.end
        if gap < args.pause_min_sec:
            continue
        if prev.is_aizuchi or cur.is_aizuchi:
            continue
        if not (kept(prev) and kept(cur)):
            continue
        # 無音の間に相談員が喋っていたら、それは間ではなくターン交替。
        if any(c.start < cur.start and c.end > prev.end for c in counselor_turns):
            continue
        cases.append(
            {
                "task": "pause_handling",
                "eval_start": prev.start,
                "eval_end": cur.end,
                "user_turns": [prev, cur],
                "events": {"pause": [[prev.end, cur.start]]},
                "expected_behavior":
                    "途中の沈黙では長い返答を始めず、発話の続きまで待つ。短い相槌は許容する。",
            }
        )

    # --- smooth_turn_taking: 重畳なしでユーザーのターンが終わった直後 ---
    for turn in user_turns:
        if turn.is_aizuchi or not kept(turn):
            continue
        following = [c for c in counselor_turns if c.start >= turn.end]
        if not following:
            continue
        nxt = min(following, key=lambda c: c.start)
        if any(overlaps(c, turn) > 0.0 for c in counselor_turns):
            continue
        cases.append(
            {
                "task": "smooth_turn_taking",
                "eval_start": turn.start,
                "eval_end": turn.end,
                "user_turns": [turn],
                "human_response": nxt,
                "expected_behavior":
                    "ユーザーが話し終えたら、過度な間を置かずに応答を始める。",
            }
        )

    # --- user_backchannel / user_interruption: ユーザーが相談員に重ねた区間 ---
    for turn in user_turns:
        if not kept(turn):
            continue
        host = next((c for c in counselor_turns if overlaps(c, turn) > 0.0), None)
        if host is None:
            continue
        task = "user_backchannel" if turn.is_aizuchi else "user_interruption"
        event_key = "overlap" if turn.is_aizuchi else "interruption"
        cases.append(
            {
                "task": task,
                "eval_start": turn.start,
                "eval_end": turn.end,
                "user_turns": [turn],
                "events": {event_key: [[turn.start, turn.end]]},
                "expected_behavior": (
                    "ユーザーの相槌は話の続行を促す合図なので、話し続けてよい。"
                    if turn.is_aizuchi
                    else "ユーザーが割り込んだら速やかに話すのをやめ、相手に譲る。"
                ),
            }
        )

    cases.sort(key=lambda c: (c["task"], c["eval_start"]))
    return cases


def build_case_metadata(
    case: dict[str, Any],
    case_id: str,
    turns: list[Turn],
    counselor: str,
    window_start: float,
    window_end: float,
    sample_rate: int,
    duration_sec: float,
) -> dict[str, Any]:
    """評価器が読む metadata を組み立てる。時刻はすべて窓の先頭からの相対秒。"""

    def rel(value: float) -> float:
        return round(value - window_start, 4)

    user_segments = [
        {
            "start_sec": rel(turn.start),
            "end_sec": rel(turn.end),
            "text": turn.text,
            "kind": "speech",
            "is_aizuchi": turn.is_aizuchi,
        }
        for turn in case["user_turns"]
    ]

    events = {
        name: [[rel(lo), rel(hi)] for lo, hi in intervals]
        for name, intervals in (case.get("events") or {}).items()
    }

    # 相談員がこの窓で実際に返した発話。LLM-as-a-judge の参照であり、
    # 「人間ならこう返した」の唯一の根拠になる。
    human_segments = [
        {
            "start_sec": rel(turn.start),
            "end_sec": rel(turn.end),
            "text": turn.text,
            "is_aizuchi": turn.is_aizuchi,
            "aizuchi_labels": turn.aizuchi_labels,
        }
        for turn in turns
        if turn.speaker == counselor and turn.end > window_start and turn.start < window_end
    ]
    response = case.get("human_response")
    human_reference = {
        "segments": human_segments,
        "response_text": response.text if response is not None else "",
        "response_latency_sec": (
            round(response.start - case["eval_end"], 4) if response is not None else None
        ),
    }

    backchannel_gt = [
        {
            "start_sec": rel(turn.start),
            "end_sec": rel(turn.end),
            "labels": turn.aizuchi_labels,
            "text": turn.text,
        }
        for turn in case.get("backchannel_gt", [])
    ]

    return {
        "id": case_id,
        "task": case["task"],
        "category": "real_dialogue",
        "risk_level": "unknown",
        "language": "ja",
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 4),
        "expected_behavior": case.get("expected_behavior"),
        "avoid_behavior": [],
        "user_segments": user_segments,
        "events": events,
        "paired": False,
        "opening_greeting": None,
        "source": {
            "window_start_sec": round(window_start, 4),
            "window_end_sec": round(window_end, 4),
            "eval_start_sec": rel(case["eval_start"]),
            "eval_end_sec": rel(case["eval_end"]),
        },
        "human_reference": human_reference,
        "backchannel_gt": backchannel_gt,
    }


def process_analysis_dir(
    analysis_dir: Path, out_dir: Path, args: argparse.Namespace,
    selected_tasks: set[str] | None, counters: dict[str, int],
) -> list[dict[str, Any]]:
    counselor = resolve_counselor(analysis_dir)
    turns_raw = load_timeline(analysis_dir)
    speakers = sorted({t.speaker for t in turns_raw})
    if counselor not in speakers:
        raise SystemExit(
            f"{analysis_dir}: counselor_{counselor}.txt が指す話者が timeline にいません: {speakers}"
        )
    if len(speakers) != 2:
        raise SystemExit(f"{analysis_dir}: 2 話者を想定していますが {speakers} でした。")
    user = next(sp for sp in speakers if sp != counselor)
    print(f"[real-data] {analysis_dir.name}: counselor={counselor} user={user}")

    keep = None if args.no_review_filter else load_keep_filter(analysis_dir, user)
    if keep is None and not args.no_review_filter:
        print(
            f"[real-data] WARNING: {user}_segments_review.tsv がないため全セグメントを使います。"
            " 人手の間引きを反映したい場合は separate_real_dialogue.py の出力を編集してください。"
        )

    turns = merge_turns(turns_raw, args.turn_merge_gap_sec)
    user_wav = analysis_dir / "separated" / f"{user}.wav"
    if not user_wav.is_file():
        raise SystemExit(f"{user_wav} がありません。分離ステージが未完了です。")
    audio, source_rate = sf.read(user_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    cases = detect_cases(turns, user, counselor, keep, args)
    manifest_rows: list[dict[str, Any]] = []
    for case in cases:
        task = case["task"]
        if selected_tasks is not None and task not in selected_tasks:
            continue
        if (
            args.max_cases_per_task is not None
            and counters.get(task, 0) >= args.max_cases_per_task
        ):
            continue

        window_start = max(0.0, case["eval_start"] - args.context_sec)
        window_end = min(audio.size / source_rate, case["eval_end"] + args.tail_sec)
        if window_end <= window_start:
            continue

        lo = max(0, int(round(window_start * source_rate)))
        hi = min(audio.size, int(round(window_end * source_rate)))
        pcm = resample_linear(audio[lo:hi], source_rate, args.sample_rate)
        if pcm.size == 0:
            continue

        counters[task] = counters.get(task, 0) + 1
        case_id = f"{analysis_dir.name}_{task}_{counters[task]:03d}"
        metadata = build_case_metadata(
            case, case_id, turns, counselor, window_start, window_end,
            args.sample_rate, pcm.size / args.sample_rate,
        )
        metadata["source"]["analysis_dir"] = str(analysis_dir)
        metadata["source"]["counselor_speaker"] = counselor
        metadata["source"]["user_speaker"] = user

        sample_dir = out_dir / task / case_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sf.write(sample_dir / "input.wav", pcm, args.sample_rate)
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (sample_dir / "input.txt").write_text(
            "".join(seg["text"] for seg in metadata["user_segments"]) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "id": case_id,
                "task": task,
                "path": str(sample_dir.relative_to(out_dir)),
                "paired": False,
            }
        )
    return manifest_rows


def main() -> int:
    args = parse_args()
    selected_tasks = None if args.tasks == "all" else {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }
    if selected_tasks:
        unknown = selected_tasks - set(TASKS)
        if unknown:
            raise SystemExit(f"未知のタスク: {sorted(unknown)}")

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} は既に存在します。--overwrite を付けてください。")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    sources: list[str] = []
    for analysis_dir in args.analysis_dir:
        analysis_dir = analysis_dir.resolve()
        if not analysis_dir.is_dir():
            raise SystemExit(f"ディレクトリがありません: {analysis_dir}")
        samples.extend(
            process_analysis_dir(analysis_dir, out_dir, args, selected_tasks, counters)
        )
        sources.append(str(analysis_dir))

    manifest = {
        "profile": "Real dialogue replay (Japanese)",
        "language": "ja",
        "protocol": {
            "name": PROTOCOL_NAME,
            "paired_clean_input": False,
            "open_loop": True,
            "note": (
                "入力はユーザー側の実音声のみ。相談員側はモデルが生成するため、"
                "重畳前提の指標は評価器の overlap_achieved で条件付けて読むこと。"
                "合成 Full-Duplex-Bench-JA のスコアと同じ表に並べてはいけない。"
            ),
        },
        "dataset_config": {
            "sample_rate": args.sample_rate,
            "context_sec": args.context_sec,
            "tail_sec": args.tail_sec,
            "pause_min_sec": args.pause_min_sec,
            "backchannel_min_turn_sec": args.backchannel_min_turn_sec,
            "turn_merge_gap_sec": args.turn_merge_gap_sec,
            "review_filter": not args.no_review_filter,
        },
        "sources": sources,
        "count": len(samples),
        "counts_by_task": dict(sorted(counters.items())),
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[real-data] wrote {len(samples)} cases -> {out_dir}")
    for task, count in sorted(counters.items()):
        print(f"[real-data]   {task}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
