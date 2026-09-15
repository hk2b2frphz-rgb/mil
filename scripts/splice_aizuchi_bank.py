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


def _load_sibling(name: str):
    """scripts/<name>.py をファイルパスから読む。

    "from scripts.x import y" は使えない。このパイプラインは PYTHONPATH に
    kaburi-tts の側も載せて動くので、トップレベルの "scripts" がどちらの
    ディレクトリに解決されるかが環境で変わる（向こうにも scripts/ がある）。
    名前空間パッケージとして両方が混ざれば通るが、片方にでも __init__.py が
    あれば通常パッケージになって一方だけが見え、"No module named
    scripts.alignment_words" になる。パスで指せばその曖昧さが無い。
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_mil_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} を読み込めません")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_alignment_words = _load_sibling("alignment_words")
get_segmenter_name = _alignment_words.get_segmenter_name
split_utterance_alignments = _alignment_words.split_utterance_alignments

_probe = _load_sibling("qwen3_tts_diversity_probe")
measure = _probe.measure
read_wav = _probe.read_wav
split_outliers = _probe.split_outliers

# 左が moshi(聞き手)、右が user。KABURI の出力もこの並び。
MAIN_LABEL = "SPEAKER_MAIN"
LEFT_CHANNEL = 0
# 継ぎ目のプチッを消すだけの長さ。相槌の立ち上がりを鈍らせない範囲で。
FADE_SEC = 0.005
# 相槌とみなす文字数の上限。--aizuchi-vocab-file を渡さなかったときだけ使う
# 目安で、本来は語で決めるべきもの。「なるほど」は 4 文字だが うん では
# 代用できない（評価であって継続の相槌ではない）ので、長さで切ると必ず
# 取りこぼすか取りすぎる。
DEFAULT_MAX_CHARS = 6
# 「置き換えなかった短い発話」として報告する上限。語彙を広げる判断材料。
SHORT_UTTERANCE_CHARS = 12


def stereo_dir(training_dir: Path) -> Path:
    """training_set の中で wav と JSON が置かれている場所。

    このリポの TTS はどの経路も <training_set>/data_stereo/<stem>.{wav,json} に
    書く（report_tts_smoke.py もそこを見る）。data_stereo そのものを渡された
    場合も通す。
    """
    nested = training_dir / "data_stereo"
    if nested.is_dir():
        return nested
    return training_dir


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
        # 頭の無音だけ落とす。尻は落とさない -- 無音判定はピークの 5% を
        # 下回ったところで切るので、「うん」の終わりの鼻音のように緩やかに
        # 減衰する音は最後が削れる。「うんが最後まで言っていない」はこれ。
        # 尻に残る無音は聞き手チャンネルの無音なので、残っても害が無い。
        start = int(round(stats["lead_silence_sec"] * sample_rate))
        trimmed = signal[start:] if start < signal.size else signal
        if trimmed.size == 0:
            continue
        clips.append(
            {
                "wav": entry["wav"],
                "text": entry["text"],
                "temperature": entry.get("temperature"),
                "signal": trimmed,
                "sample_rate": sample_rate,
                # 尺の突き合わせに使うのは実際に鳴っている長さ。尻の無音を
                # 含めると、短い音を長い区間に当ててしまう。
                "speech_sec": float(stats["speech_sec"]),
                "clip_sec": trimmed.size / sample_rate,
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


def backchannel_anchors(
    placements: Sequence[dict[str, Any]], max_chars: int, vocab: set[str] | None
) -> list[dict[str, Any]]:
    """各相槌が「どの user 発話の、どこで」打たれたかを取り出す。

    相手の発話の終わりからの差だけで測ると、かぶりを表せない。相槌は言い終わる
    のを待って打つものではなく、発話の途中に入る -- 実録音でも「うん」の 71% は
    相手の発話に重なっていて、終わりから測った差は p10 で -17 秒だった。つまり
    「終わりの何秒前」ではなく「発話のどのあたり」が本体。

    そこで 2 通りに分けて持つ:

      inside  発話の最中に入った。発話長に対する割合で持つ。既存側の発話が
              長くても短くても、同じ「あたり」に入る
      after   言い終わってから入った。終わりからの秒数で持つ

    anchor は user 発話の何番目か。既存側の同じ順番の発話に当てはめる。
    """
    rows = sorted(placements, key=lambda row: row["start_sec"])
    users = [row for row in rows if row["label"] != MAIN_LABEL]
    anchors: list[dict[str, Any]] = []
    for row in rows:
        if row["label"] != MAIN_LABEL or not is_backchannel(
            str(row["text"]), max_chars, vocab
        ):
            continue
        start = float(row["start_sec"])
        prior = [
            (index, user)
            for index, user in enumerate(users)
            if float(user["start_sec"]) <= start
        ]
        if not prior:
            # 相手が話し始める前。位置をそのまま持つしかない。
            anchors.append({"anchor": -1, "mode": "absolute", "value": round(start, 4)})
            continue
        index, anchor = prior[-1]
        anchor_start = float(anchor["start_sec"])
        anchor_end = float(anchor["end_sec"])
        span = anchor_end - anchor_start
        if start < anchor_end and span > 0:
            anchors.append(
                {
                    "anchor": index,
                    "mode": "inside",
                    "value": round((start - anchor_start) / span, 4),
                }
            )
        else:
            anchors.append(
                {"anchor": index, "mode": "after", "value": round(start - anchor_end, 4)}
            )
    return anchors


def retime_backchannels(
    rows: Sequence[Sequence[Any]],
    anchors: Sequence[dict[str, Any]],
    max_chars: int,
    vocab: set[str] | None,
    duration_sec: float,
) -> dict[int, float]:
    """既存の音声の相槌に KABURI の位置を当てはめ、{行番号: 新しい開始秒} を返す。

    相槌どうしの対応は出てくる順で、anchor の user 発話も出てくる順で取る。
    本数が合わないときは何も返さない -- 途中でずれた対応を黙って通すより、
    その対話を飛ばした方がよい。
    """
    indexed = sorted(range(len(rows)), key=lambda i: rows[i][1][0])
    users = [i for i in indexed if rows[i][2] != MAIN_LABEL]
    targets = [
        i
        for i in indexed
        if rows[i][2] == MAIN_LABEL and is_backchannel(str(rows[i][0]), max_chars, vocab)
    ]
    if len(targets) != len(anchors):
        return {}

    new_starts: dict[int, float] = {}
    for position, row_index in enumerate(targets):
        anchor = anchors[position]
        index = int(anchor.get("anchor", -1))
        if index < 0 or index >= len(users):
            moved = float(anchor.get("value", rows[row_index][1][0]))
        else:
            user_row = rows[users[index]]
            user_start = float(user_row[1][0])
            user_end = float(user_row[1][1])
            if anchor.get("mode") == "inside":
                span = max(0.0, user_end - user_start)
                moved = user_start + float(anchor["value"]) * span
            else:
                moved = user_end + float(anchor["value"])
        new_starts[row_index] = min(max(0.0, moved), max(0.0, duration_sec - 0.01))
    return new_starts


def load_placements(placement_dir: Path) -> dict[str, dict[str, Any]]:
    """配置 JSON を対話 ID で引けるようにする。

    ファイル名(stem)では引かない。既存のコーパスはシャードに割ってから
    合成しているので sample_00001 の番号は振り直されており、同じ名前が別の
    対話を指す。metadata.dialogue.id なら、どう割られていても同じものを指す。
    """
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(placement_dir.glob("*.placement.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = str(payload.get("id") or "")
        if key:
            index[key] = payload
    if not index:
        raise SystemExit(f"配置 JSON がありません（または id が無い）: {placement_dir}")
    return index


def pick_clip(
    clips: Sequence[dict[str, Any]],
    target_sec: float,
    top_k: int,
    rng: random.Random,
    want_text: str = "",
) -> tuple[dict[str, Any], bool]:
    """尺の近い順に top_k 本を出し、その中から引く。

    want_text があれば、まず同じ語のクリップだけに絞る。対話が「うん」と
    「そっか」を打ち分けているのに、どちらにも「うん」を差したら打ち分けが
    消える。同じ語が 1 本も無いときはバンク全体から引き、その旨を返す。
    """
    pool = clips
    fell_back = False
    if want_text:
        same = [clip for clip in clips if str(clip["text"]).strip() == want_text]
        if same:
            pool = same
        else:
            fell_back = True
    ordered = sorted(pool, key=lambda clip: abs(clip["speech_sec"] - target_sec))
    return rng.choice(ordered[: max(1, top_k)]), fell_back


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
        want = str(text).strip().strip("。、．，!?！？…・ 　") if args.match_text else ""
        clip, fell_back = pick_clip(clips, slot_sec, args.match_top_k, rng, want)
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
                "text_fallback": fell_back,
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
    parser.add_argument(
        "--bank-text",
        default="",
        help="バンクをこの 1 語に絞る。既定は絞らず、発話ごとに同じ語を引く",
    )
    parser.add_argument("--bank-temperature", type=float, default=None)
    parser.add_argument("--keep-bank-outliers", action="store_true")
    parser.add_argument("--aizuchi-max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--aizuchi-vocab-file", type=Path, default=None, help="1 行 1 語。文字数判定の代わり"
    )
    parser.add_argument("--match-top-k", type=int, default=5)
    parser.add_argument(
        "--no-match-text",
        dest="match_text",
        action="store_false",
        help="語を合わせず、尺だけでバンクから引く",
    )
    parser.add_argument("--gain", choices=("match", "none"), default="match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="対話数の上限(0 で全部)")
    args = parser.parse_args(argv)

    import torch
    import torchaudio

    source_dir = stereo_dir(args.training_dir)
    json_paths = sorted(source_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(
            f"対話 JSON がありません: {source_dir}"
            "（<training_set>/data_stereo/<stem>.json を探します）"
        )
    # 配置で絞り込むときは、--limit を入力ファイルの頭から数えてはいけない。
    # 既存コーパスの先頭 N 本が、配置を取った N 本とは限らない。
    limit = max(0, int(args.limit))
    if limit and args.placement_dir is None:
        json_paths = json_paths[:limit]

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
    placements = (
        load_placements(args.placement_dir) if args.placement_dir is not None else None
    )
    if placements is not None:
        print(f"[placement] {len(placements)} 対話ぶんの配置を読みました")
    rng = random.Random(args.seed)

    # 出力も同じ形にしておく。そうでないと merge_training_shards.py も
    # report_tts_smoke.py も読めない。
    out_stereo = args.out_dir / "data_stereo"
    out_stereo.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "bank_swaps.jsonl"
    report_path.unlink(missing_ok=True)

    total_swaps = 0
    retimed_total = 0
    processed = 0
    missing = 0
    replaced_texts: dict[str, int] = {}
    left_alone: dict[str, int] = {}
    text_fallbacks = 0
    missing_texts: dict[str, int] = {}
    anchor_modes: dict[str, int] = {}
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

        # 配置を先に引く。持っていない対話の wav を読む意味は無く、既存コーパスは
        # 数千本あるのに配置は数本ということがある。
        placement = None
        if placements is not None:
            dialogue_id = str(
                (payload.get("metadata", {}).get("dialogue", {}) or {}).get("id") or ""
            )
            placement = placements.get(dialogue_id)
            if placement is None:
                missing += 1
                continue

        wav, sample_rate = torchaudio.load(str(wav_path))
        stereo = wav.numpy().astype(np.float64)

        new_starts: dict[int, float] = {}
        if placement is not None:
            anchors = backchannel_anchors(
                placement.get("placements", []), args.aizuchi_max_chars, vocab
            )
            for anchor in anchors:
                anchor_modes[str(anchor.get("mode"))] = (
                    anchor_modes.get(str(anchor.get("mode")), 0) + 1
                )
            duration = stereo.shape[-1] / sample_rate
            new_starts = retime_backchannels(
                rows, anchors, args.aizuchi_max_chars, vocab, duration
            )
            if not new_starts and anchors:
                # 本数が合わない = 対応が取れない。黙ってずらすより飛ばす。
                print(
                    f"[skip] 相槌の本数が配置と合いません: {json_path.name} "
                    f"(KABURI {len(anchors)} 箇所)"
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
            "match": "text+duration" if args.match_text else "duration",
            "match_top_k": args.match_top_k,
            "gain": args.gain,
            "seed": args.seed,
            "n_swapped": len(swaps),
            "n_retimed": len(new_starts),
            "placement_dir": str(args.placement_dir) if args.placement_dir else None,
            "turns_text_rewritten": False,
        }

        out_wav = out_stereo / wav_path.name
        torchaudio.save(
            str(out_wav),
            torch.from_numpy(spliced).to(torch.float32),
            sample_rate,
            channels_first=True,
        )
        (out_stereo / json_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        with report_path.open("a", encoding="utf-8") as handle:
            for swap in swaps:
                handle.write(
                    json.dumps({"stem": json_path.stem, **swap}, ensure_ascii=False) + "\n"
                )
        processed += 1
        for swap in swaps:
            key = str(swap["original_text"]).strip()
            replaced_texts[key] = replaced_texts.get(key, 0) + 1
            if swap.get("text_fallback"):
                text_fallbacks += 1
                missing_texts[key] = missing_texts.get(key, 0) + 1
        # 置き換えなかった聞き手側の短い発話。語彙を広げるかどうかの判断材料に
        # なるので、黙って落とさず数えておく。
        for text, _span, label in rows:
            stripped = str(text).strip()
            if (
                label == MAIN_LABEL
                and len(stripped.strip("。、．，!?！？ 　")) <= SHORT_UTTERANCE_CHARS
                and not is_backchannel(stripped, args.aizuchi_max_chars, vocab)
            ):
                left_alone[stripped] = left_alone.get(stripped, 0) + 1
        total_swaps += len(swaps)
        errors.extend(swap["length_error_sec"] for swap in swaps)
        moves.extend(swap["moved_sec"] for swap in swaps if swap["moved_sec"])
        print(f"[ok] {json_path.stem}: {len(swaps)} 箇所を差し替え -> {out_wav.name}")
        if limit and processed >= limit:
            break

    print()
    print(f"対話 {processed} 本 / 差し替え {total_swaps} 箇所")
    if replaced_texts:
        print("差し替えた語:")
        for text, count in sorted(replaced_texts.items(), key=lambda kv: -kv[1]):
            print(f"  {text} x{count}")
    if left_alone:
        print(f"置き換えなかった短い発話（語彙外。{args.bank_text or 'バンク'} で")
        print("代用できるものがあれば --aizuchi-vocab-file に足す）:")
        for text, count in sorted(left_alone.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {text} x{count}")
    if missing:
        print(f"  配置が無くて飛ばしたもの: {missing} 本")
    if errors:
        absolute = [abs(value) for value in errors]
        print(
            f"尺のずれ: 中央値 {statistics.median(absolute):.3f}s / "
            f"最大 {max(absolute):.3f}s / 平均 {statistics.fmean(errors):+.3f}s"
        )
        over = sum(1 for value in errors if value > 0)
        print(f"  うち区間より長くなった: {over}/{len(errors)} (相手に食い込む方向)")
    if text_fallbacks:
        print(f"バンクに同じ語が無くて別の語を差したもの: {text_fallbacks} 箇所")
        for word, count in sorted(missing_texts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {word} x{count}")
    if anchor_modes:
        inside = anchor_modes.get("inside", 0)
        after = anchor_modes.get("after", 0)
        total = sum(anchor_modes.values())
        print(
            f"KABURI が置いた位置: 相手の発話中 {inside} / 言い終えた後 {after}"
            + (f" / 発話前 {anchor_modes['absolute']}" if anchor_modes.get("absolute") else "")
        )
        if total and inside == 0:
            print("  <- ひとつも重なっていない。KABURI 側がターン制で置いている。")
            print("     placement JSON を見て gap の分布を確かめること。")
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
