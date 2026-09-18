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

音声は基本的に変更しない:
    Moshi のテキストストリームと音声ストリームは別系統でトークン化され
    (`tools.tokenize_text --word_transcript_dir` と `tools.tokenize_audio`)、
    `tools.prepare_dataset` で後から突き合わされる。よって sidecar JSON の
    alignments にだけ手を入れれば、その区間の音声は無音のままで、内心は
    読み上げられない。音声トークンも再利用できる。

    例外が --pad-lead-in-sec（--from-aizuchi-density 用）。相槌頻度のタグは
    対話ごとに値が変わってはいけないので置き場所は聞き手が最初に音を出す前の1箇所しか
    無いが、KABURI レンダラーは無音ターンを明示的なタイミングとして扱わず
    (`generate_kaburi_tts_data.py` の `load_dialogues` を参照。無音ターンは
    落として自前のタイミングモデルで間を作る)、対話生成側から無音の長さを
    保証する手段が無い。生まれつき無音が足りない対話ではタグが skip され、
    生成時に付けたはずの頻度がモデルに伝わらないまま学習データに混ざって
    しまう。--pad-lead-in-sec は音声の先頭に固定長の無音を足してから全ての
    タイムスタンプをその分だけ後ろにずらし、タグの置き場所を偶然の無音では
    なく確実な予約済みスロットにする。

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
import math
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
    src.add_argument("--from-aizuchi-density", action="store_true",
                     help="metadata.dialogue.aizuchi_frequency_label "
                          "（generate_synthetic_moshi_training_data.py が "
                          "aizuchi-only モードで記録する相槌頻度）から、対話の"
                          "聞き手の最初の発話の直前に読み上げないタグを1つだけ置く。"
                          "モデルに相槌頻度を条件付けさせるためのもの。その前に"
                          "無音が無い対話では置き場所が無く skip される"
                          "（統計の skip_gap_too_short / skip_no_anchor / "
                          "WARNING の density_tag_dropped を確認すること）。"
                          "--pad-lead-in-sec と併用すると置き場所を保証できる。"
                          "emotional_state と違い対話内で不変であることが狙い"
                          "なので学習用途に使ってよい")
    parser.add_argument("--pad-lead-in-sec", default="0",
                        help="auto を渡すと、置くのに要る最小の秒数を全対話から"
                             "measure してから使う（足す必要が無ければ 0 のまま"
                             "音声に触らない）。数値なら"
                             "0 より大きいと、音声の先頭にこの秒数の無音を足し"
                             "全タイムスタンプをその分だけ後ろへずらしてから"
                             "タグを挿す。KABURI レンダラーは無音ターンを"
                             "落として自前でタイミングを作るため、生成側からは"
                             "冒頭の無音の長さを保証できない。--from-aizuchi-"
                             "density のタグは対話に1個しか置き場所が無く、"
                             "そこが skip されると頻度がモデルに伝わらないまま"
                             "学習データに混ざるので、既定の 0（無音を足さず"
                             "元の音声のまま）ではなくこちらを使うことを推奨"
                             "する。--wav の指定は無視され、対話ごとに新しい"
                             "WAV を書き出す。soundfile が必要")
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
    args.pad_lead_in_auto = str(args.pad_lead_in_sec).strip().lower() == "auto"
    if args.pad_lead_in_auto:
        args.pad_lead_in_sec = 0.0
    else:
        try:
            args.pad_lead_in_sec = float(args.pad_lead_in_sec)
        except ValueError:
            parser.error("--pad-lead-in-sec must be a number or 'auto'")
        if args.pad_lead_in_sec < 0:
            parser.error("--pad-lead-in-sec must be >= 0")
    if args.pad_lead_in_auto and not args.from_aizuchi_density:
        parser.error("--pad-lead-in-sec auto は --from-aizuchi-density 専用")
    return args


def shift_alignments(entries: list[list[Any]], offset_sec: float) -> list[list[Any]]:
    """全エントリの [start, end] を offset_sec だけ後ろへずらす。"""
    return [
        [text, [round(start + offset_sec, 4), round(end + offset_sec, 4)], speaker]
        for text, (start, end), speaker in entries
    ]


def pad_lead_in_wav(src_wav: Path, dst_wav: Path, pad_sec: float) -> None:
    """WAV の先頭に pad_sec 秒の無音を足して dst_wav に書き出す。

    subtype を明示的に引き継ぐ。soundfile の既定は WAV なら PCM_16 なので、
    渡さないと 32bit float や 24bit PCM の音声が黙って 16bit に落ちる。
    """
    import numpy as np
    import soundfile as sf

    info = sf.info(str(src_wav))
    data, sample_rate = sf.read(str(src_wav), always_2d=True)
    pad_samples = int(round(pad_sec * sample_rate))
    silence = np.zeros((pad_samples, data.shape[1]), dtype=data.dtype)
    sf.write(
        str(dst_wav),
        np.concatenate([silence, data], axis=0),
        sample_rate,
        subtype=info.subtype,
    )


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


def aizuchi_tag_text(label: str) -> str:
    """頻度ラベルを、学習データまで字面が変わらないタグにする。

    prepare_nu_fullft_dataset.py は nu に渡す前にテキストを正規化する
    （NFKC、ASCII 記号の全角化、連続する句読点の圧縮、空白の除去）。ここを
    素通りしない字面を使うと、学習データに入る形と、推論時にこちらが与える形が
    食い違う。実際 "<相槌:density=0.75>" は ASCII コロンが全角に化けていた。
    デバッグが難しい壊れ方（タグは効かないが、どこも失敗しない）なので、
    正規化を通しても同じになる字面だけを使う:

    - ASCII コロン・空白を使わない（前者は全角化、後者は削除される）
    - 小数点を使わない。単語分割の切れ目が "." の直前に来ると、前後が数字で
      なくなって "。" に化ける。密度は 0-100 の整数にする
    - 短くする。単語分割で細切れになる数が減るうえ、置くのに要る無音も
      短くて済む（--pad-lead-in-sec がその分小さくできる）

    テストが実際の正規化関数に通して字面が変わらないことを確かめている。
    """
    if label.startswith("density="):
        try:
            percent = round(float(label.split("=", 1)[1]) * 100)
        except ValueError:
            percent = 0
        return f"<相槌{percent}>"
    # rule の preset 名（eager 等）と llm。どちらも ASCII 英字だけなので
    # 正規化を素通りする。
    return f"<相槌{label}>"


def density_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """相槌頻度の条件付けタグを聞き手の最初の発話の直前に1つだけ置く。

    emotional_state と違い、対話全体を通して値が変わらないことが目的そのもの
    （density_mixed で対話ごとに引いた頻度をモデルに教えるタグなので、対話の
    途中でぶれてはむしろ困る）。turn_index=0 は聞き手の最初の発話を指すので、
    bootstrap_items が「まだ何も聴いていない」ことを理由に避けている場所に、
    ここでは意図して置く。

    それが何になるかはレンダリング経路で変わる。聞き手側も合成する経路では
    冒頭の名乗りだが、バンク経路（run_bank_stereo_test.sh）は user 側だけを
    合成して聞き手の音をバンクから差し込むので、名乗りは音声に存在せず、
    最初の SPEAKER_MAIN は最初の相槌になる。どちらでも「聞き手が最初に音を
    出す前」であることは変わらないので、置き場所としては同じ意味を持つ。
    """
    meta = payload.get("metadata") or {}
    dialogue = meta.get("dialogue") or {}
    label = str(dialogue.get("aizuchi_frequency_label") or "").strip()
    if not label:
        return []
    return [{"turn_index": 0, "text": aizuchi_tag_text(label)}]


def required_pad_sec(
    json_paths: list[Path], chars_per_sec: float, gap_margin_sec: float
) -> tuple[float, int]:
    """density タグを全対話に置くのに要る最小の無音を測る。

    タグを置く窓は [直前の聞き手発話の終わり, 最初の聞き手発話 - margin]。
    turn_index=0 には直前の聞き手発話が無いので窓の始まりは常に 0.0 で、
    先頭に P 秒足すと窓は P だけ広がる。つまり 1 対話に要る P は

        needed + margin - anchor

    で、コーパス全体にはその最大値が要る。挨拶を残したコーパスでは聞き手が
    0.3 秒あたりで喋り出すので、既定の 0 では 1 件も置けない。それを 10000 件
    走らせてから WARNING で知るのは高くつくので、先に測る。

    JSON しか読まないので、音声を書き直すかどうかを決める前に済む。
    """
    required = 0.0
    measured = 0
    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries = utterance_alignments(payload)
        if entries is None:
            continue
        items = density_items(payload)
        if not items:
            continue
        anchor = resolve_anchor(items[0], entries)
        if anchor is None:
            continue
        measured += 1
        needed = len(str(items[0].get("text") or "")) / chars_per_sec
        short = needed + gap_margin_sec - (anchor - prev_moshi_end(entries, anchor))
        required = max(required, short)
    if required <= 0:
        return 0.0, measured
    # 測った値ちょうどだと丸めで際どい対話が落ちる。0.1 秒刻みで切り上げる。
    return math.ceil(required * 10.0) / 10.0, measured


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

    if args.pad_lead_in_auto:
        args.pad_lead_in_sec, measured = required_pad_sec(
            json_paths, args.chars_per_sec, args.gap_margin_sec
        )
        print(f"[pad] measured {measured} dialogue(s) -> "
              f"--pad-lead-in-sec {args.pad_lead_in_sec}")
        if args.pad_lead_in_sec > 0:
            print("[pad] 先頭に無音を足すので全 WAV を書き直します"
                  "（元の音声と同じだけディスクを使います）。")
        else:
            print("[pad] 足す必要なし。WAV は触りません。")

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

        if args.pad_lead_in_sec > 0:
            src_wav = json_path.with_suffix(".wav")
            if not src_wav.is_file():
                stats["skip_no_wav_to_pad"] += 1
                continue
            entries = shift_alignments(entries, args.pad_lead_in_sec)
            meta_in = payload.get("metadata")
            if isinstance(meta_in, dict) and "duration_sec" in meta_in:
                meta_in["duration_sec"] = round(
                    float(meta_in["duration_sec"]) + args.pad_lead_in_sec, 4
                )

        if args.from_emotional_state:
            items = bootstrap_items(payload, entries)
        elif args.from_aizuchi_density:
            items = density_items(payload)
            if items:
                stats["density_labeled"] += 1
        else:
            items = thoughts.get(stem, [])
        if not items:
            stats["skip_no_thoughts"] += 1
            continue

        merged, injected = inject_one(entries, items, args.chars_per_sec,
                                      args.gap_margin_sec, stats)
        if injected == 0:
            stats["samples_unchanged"] += 1
            if args.from_aizuchi_density:
                stats["density_tag_dropped"] += 1
            continue

        word_level, split_stats = split_utterance_alignments(merged)

        payload["alignments"] = word_level
        payload["alignments_utterance"] = merged
        meta = payload.setdefault("metadata", {})
        if args.from_emotional_state:
            source = "emotional_state"
        elif args.from_aizuchi_density:
            source = "aizuchi_density"
        else:
            source = str(args.thoughts)
        meta["inner_thoughts"] = {
            "version": 1,
            "injected": injected,
            "chars_per_sec": args.chars_per_sec,
            "gap_margin_sec": args.gap_margin_sec,
            "source": source,
            "pad_lead_in_sec": args.pad_lead_in_sec,
        }
        meta["alignments_word_split"] = dict(
            meta.get("alignments_word_split") or {}, words=split_stats["words"]
        )
        stats["samples_written"] += 1

        if args.dry_run:
            continue

        out_json = args.out_dir / json_path.name
        out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        src_wav = json_path.with_suffix(".wav")
        dst_wav = args.out_dir / src_wav.name
        if args.pad_lead_in_sec > 0:
            # --wav symlink/copy はどちらも元の音声をそのまま持ってくるだけ
            # なので、無音を足した新しい音声を書くこの経路には使えない。
            # soundfile は書き込み前に全体を読み切るので、in-place（src と
            # dst が同じパス）で上書きしても安全。
            pad_lead_in_wav(src_wav, dst_wav, args.pad_lead_in_sec)
        elif not in_place and args.wav != "none" and src_wav.is_file() and not dst_wav.exists():
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
    if args.from_aizuchi_density:
        dropped = stats["density_tag_dropped"]
        labeled = stats["density_labeled"]
        if dropped:
            fate = (
                "タグ無しの元ファイルがそのまま残ります（--out-dir が "
                "--data-dir と同じなので）。頻度がモデルに伝わらない対話が"
                "コーパスに混ざるということです"
                if in_place
                else "--out-dir に書き出されません。つまりこの分だけ"
                "コーパスが小さくなります（誤ったタグが付くよりは安全側ですが、"
                "黙って件数が減るので気付けるようにここで出しています）"
            )
            print(
                f"\nWARNING: density タグを置けなかった対話が {dropped}/{labeled} "
                f"件あります（聞き手が最初に音を出す前に十分な無音が無かった）。この分は"
                f"{fate}。--pad-lead-in-sec を（十分な秒数で）指定すれば確実に"
                f"置けます。指定済みで出ている場合はタグの長さに対して値が"
                f"短すぎるので、増やすか --chars-per-sec を上げてください。"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
