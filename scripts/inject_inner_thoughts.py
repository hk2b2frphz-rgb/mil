#!/usr/bin/env python3
"""レンダリング済みサンプルの moshi テキストストリームに「内心」を差し込む。

狙い:
    相づちだけのコーパスは moshi 側テキストが内容非依存（`はい。` は相手が
    何を話していても正解）なため、ユーザーチャンネルを聴かなくても損失が
    下がる。結果としてモデルは条件付けを学ばず、入力に関係なく相づちを
    出し続ける無条件生成器に縮退する。

    そこで相づちの直前に「相手をどう受け止めたか」を書いた読み上げない
    テキストを置き、相づちに到達する前に必ずユーザー発話へ依存した予測を
    通す。テキストストリームが内容依存になるので、聴かなければ当たらない。

音声は一切変更しない:
    Moshi のテキストストリームと音声ストリームは別系統でトークン化され
    (`tools.tokenize_text --word_transcript_dir` と `tools.tokenize_audio`)、
    `tools.prepare_dataset` で後から突き合わされる。よって sidecar JSON の
    alignments にだけ手を入れれば、その区間の音声は無音のままで、内心は
    読み上げられない。音声トークンも再利用できる。

1 エントリ = 1 単語:
    moshi-finetune の Interleaver はエントリ開始フレームからトークンを
    詰めるため、発話単位のまま渡すとペーシングが壊れる
    （scripts/alignment_words.py の docstring を参照）。本スクリプトは発話
    単位リスト（`alignments_utterance`）に内心を挿してから
    `split_utterance_alignments` で単語単位に展開し直し、生成時と同じ形の
    `alignments` を書き戻す。

内心はユーザー発話に重なってよい:
    「聴きながら考える」ので、避けるべきなのは moshi 自身の発話との衝突だけ。
    ユーザーチャンネルとの重なりは意図した挙動。

使い方:
    uv run python scripts/inject_inner_thoughts.py \
        --data-dir data/runs/<run>/tts/merged/training_set/data_stereo \
        --thoughts thoughts.jsonl \
        --out-dir  data/runs/<run>/tts/merged/training_set/data_stereo_thought

    thoughts.jsonl の 1 行 = 1 対話:
        {"stem": "sample_001_xxx",
         "thoughts": [{"turn_index": 1, "text": "<仕事の話で声が沈んだ>"},
                      {"anchor_sec": 31.2, "text": "<言い淀んで止まった>"}]}

    turn_index は SPEAKER_MAIN 発話の 0 始まりの通し番号（固定の冒頭挨拶も
    1 つと数える）。anchor_sec を直接与えてもよい。両方あれば turn_index 優先。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_words import split_utterance_alignments  # noqa: E402

MOSHI_LABEL = "SPEAKER_MAIN"

# 日本語の発話速度 ~8 文字/秒。alignment_words.DEFAULT_MAX_WORD_CHARS が
# 「8 文字 ≒ 1 秒分（Mimi 12.5Hz）」を前提にしているのと同じ見積り。内心を
# 置くのに必要な無音の長さをこれで見る。
DEFAULT_CHARS_PER_SEC = 8.0

# 内心の終わりと相づちの始まりの間に空ける余白。ゼロにすると単語分割の丸めで
# 相づちの開始フレームに食い込むことがある。
DEFAULT_GAP_MARGIN_SEC = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject unspoken inner-thought text into moshi's text stream."
    )
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="sidecar JSON と WAV があるディレクトリ")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="書き出し先。--data-dir と同じなら破壊的更新")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--thoughts", type=Path,
                     help="stem ごとの内心を書いた JSONL")
    src.add_argument("--from-emotional-state", action="store_true",
                     help="配管確認用。metadata の emotional_state から一定の"
                          "内心を作る。対話内で不変なので学習用途には使わない")
    parser.add_argument("--chars-per-sec", type=float, default=DEFAULT_CHARS_PER_SEC)
    parser.add_argument("--gap-margin-sec", type=float, default=DEFAULT_GAP_MARGIN_SEC)
    parser.add_argument("--wav", choices=["symlink", "copy", "none"], default="symlink",
                        help="out-dir へ WAV をどう用意するか")
    parser.add_argument("--dry-run", action="store_true",
                        help="書き出さずに統計だけ出す")
    parser.add_argument("--limit", type=int, default=0,
                        help="先頭 N 件だけ処理する（0 で全件）")
    args = parser.parse_args()
    if args.chars_per_sec <= 0:
        parser.error("--chars-per-sec must be > 0")
    if args.gap_margin_sec < 0:
        parser.error("--gap-margin-sec must be >= 0")
    return args


def load_thoughts(path: Path) -> dict[str, list[dict[str, Any]]]:
    table: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from None
            stem = str(row.get("stem") or "").strip()
            if not stem:
                raise SystemExit(f"{path}:{lineno}: missing 'stem'")
            items = row.get("thoughts") or []
            if not isinstance(items, list):
                raise SystemExit(f"{path}:{lineno}: 'thoughts' must be a list")
            table[stem] = items
    return table


def utterance_alignments(payload: dict[str, Any]) -> list[list[Any]] | None:
    """発話単位の alignments を取り出す。無ければ None。"""
    raw = payload.get("alignments_utterance")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[list[Any]] = []
    for entry in raw:
        if (not isinstance(entry, (list, tuple)) or len(entry) < 3
                or not isinstance(entry[1], (list, tuple)) or len(entry[1]) < 2):
            return None
        try:
            span = [float(entry[1][0]), float(entry[1][1])]
        except (TypeError, ValueError):
            return None
        out.append([entry[0], span, entry[2]])
    return out


def moshi_turn_starts(entries: list[list[Any]]) -> list[float]:
    return [e[1][0] for e in entries if e[2] == MOSHI_LABEL]


def resolve_anchor(item: dict[str, Any], entries: list[list[Any]]) -> float | None:
    """内心を置く相づちの開始時刻を決める。"""
    if item.get("turn_index") is not None:
        starts = moshi_turn_starts(entries)
        try:
            idx = int(item["turn_index"])
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(starts):
            return None
        return starts[idx]
    if item.get("anchor_sec") is not None:
        try:
            return float(item["anchor_sec"])
        except (TypeError, ValueError):
            return None
    return None


def prev_moshi_end(entries: list[list[Any]], anchor: float) -> float:
    """anchor より前で最後に moshi が喋り終わった時刻。無ければ 0.0。"""
    ends = [e[1][1] for e in entries if e[2] == MOSHI_LABEL and e[1][1] <= anchor]
    return max(ends) if ends else 0.0


def inject_one(
    entries: list[list[Any]],
    items: list[dict[str, Any]],
    chars_per_sec: float,
    gap_margin_sec: float,
    stats: Counter,
) -> tuple[list[list[Any]], int]:
    """発話単位リストに内心を挿す。衝突するものは飛ばす。"""
    merged = [list(e) for e in entries]
    injected = 0
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            stats["skip_empty_text"] += 1
            continue
        anchor = resolve_anchor(item, entries)
        if anchor is None:
            stats["skip_no_anchor"] += 1
            continue

        # 直前の moshi 発話が終わってから相づちが始まるまでが置ける範囲。
        # ユーザー発話との重なりは許す（聴きながら考えるのが狙いなので）。
        window_end = anchor - gap_margin_sec
        window_start = prev_moshi_end(merged, anchor)
        available = window_end - window_start
        needed = len(text) / chars_per_sec
        if available < needed:
            stats["skip_gap_too_short"] += 1
            continue

        start = window_end - needed
        merged.append([text, [round(start, 4), round(window_end, 4)], MOSHI_LABEL])
        injected += 1
        stats["injected"] += 1

    merged.sort(key=lambda e: (e[1][0], e[1][1]))
    return merged, injected


def bootstrap_items(payload: dict[str, Any], entries: list[list[Any]]) -> list[dict[str, Any]]:
    """配管確認用。対話ごとに不変な内心を全 moshi ターンに置く。"""
    meta = payload.get("metadata") or {}
    dialogue = meta.get("dialogue") or {}
    state = str(dialogue.get("emotional_state") or "").strip()
    if not state:
        return []
    text = f"<{state}ように感じている>"
    # 冒頭挨拶（最初の moshi ターン）には内心を置かない。まだ何も聴いていない。
    return [{"turn_index": i, "text": text}
            for i in range(1, len(moshi_turn_starts(entries)))]


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"ERROR: --data-dir is not a directory: {args.data_dir}")
    thoughts = load_thoughts(args.thoughts) if args.thoughts else {}

    in_place = args.out_dir.resolve() == args.data_dir.resolve()
    if not args.dry_run and not in_place:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    json_paths = sorted(args.data_dir.glob("*.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]

    for json_path in json_paths:
        stats["samples"] += 1
        stem = json_path.stem
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["skip_unreadable"] += 1
            continue

        entries = utterance_alignments(payload)
        if entries is None:
            # 生成が古く alignments_utterance が無い。単語単位から復元すると
            # 分割済みの境界が混ざって二重分割になるため、触らず飛ばす。
            stats["skip_no_utterance_alignments"] += 1
            continue

        items = (bootstrap_items(payload, entries) if args.from_emotional_state
                 else thoughts.get(stem, []))
        if not items:
            stats["skip_no_thoughts"] += 1
            continue

        merged, injected = inject_one(entries, items, args.chars_per_sec,
                                      args.gap_margin_sec, stats)
        if injected == 0:
            stats["samples_unchanged"] += 1
            continue

        word_level, split_stats = split_utterance_alignments(merged)

        payload["alignments"] = word_level
        payload["alignments_utterance"] = merged
        meta = payload.setdefault("metadata", {})
        meta["inner_thoughts"] = {
            "version": 1,
            "injected": injected,
            "chars_per_sec": args.chars_per_sec,
            "gap_margin_sec": args.gap_margin_sec,
            "source": "emotional_state" if args.from_emotional_state else str(args.thoughts),
        }
        meta["alignments_word_split"] = dict(
            meta.get("alignments_word_split") or {}, words=split_stats["words"]
        )
        stats["samples_written"] += 1

        if args.dry_run:
            continue

        out_json = args.out_dir / json_path.name
        out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if not in_place and args.wav != "none":
            src_wav = json_path.with_suffix(".wav")
            dst_wav = args.out_dir / src_wav.name
            if src_wav.is_file() and not dst_wav.exists():
                if args.wav == "symlink":
                    dst_wav.symlink_to(src_wav.resolve())
                else:
                    shutil.copy2(src_wav, dst_wav)

    print("=== inject_inner_thoughts ===")
    for key in sorted(stats):
        print(f"{key:34s} {stats[key]}")
    if stats["skip_gap_too_short"]:
        print("\nNOTE: skip_gap_too_short は内心が無音に収まらなかった件数。"
              "\n      内心を短くするか --chars-per-sec を見直すこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
