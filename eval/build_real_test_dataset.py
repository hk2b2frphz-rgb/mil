#!/usr/bin/env python3
"""
人手アノテーション由来のテストデータを、推論スクリプトが読む形へ変換する。

入力は scripts/build_test_wav_from_tsv.py が作った本番用テストデータ
(既定 data/test_data/real_dialogue)。出力は eval/run_full_duplex_bench.py が
そのまま読めるデータセット形状。

  <out-dir>/
    manifest.json                       samples 一覧
    real_response/<dialogue>__<idx>/
      input.wav                         User の発話(テスト入力)
      metadata.json                     評価点の情報と Staff の実応答(正解)

metadata.json の human_reference には、その User 発話の直後に**実際の相談員が
返した応答**が入る。model_id=gold の評価はこれを参照して走るので、人間の
応答速度・音声品質がそのケースの実質的な上限として同じ表に並ぶ。

入力に文脈は入れない(既定 --context-sec 0)。1ch 録音なので直前の文脈には
相談員の声が混ざっており、それを入れるとモデルへの入力に「その場面で相談員が
何を言ったか」が漏れる。応答速度・応答率・音声品質はいずれも当該 User 発話への
反応として定義できるので、文脈なしでも指標は成立する。

使い方:
    uv run python eval/build_real_test_dataset.py \\
        --test-data-dir data/test_data/real_dialogue \\
        --out-dir data/eval_sets/real_response
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_build_test_wav_module():
    """scripts/build_test_wav_from_tsv.py をファイルパスから読む。

    `import scripts.build_test_wav_from_tsv` は、sys.path の手前に別の
    `scripts` パッケージ(依存ライブラリや実行環境が持ち込むもの)があると
    そちらに解決されて ModuleNotFoundError になる。リポジトリ内の 1 ファイル
    を読みたいだけなので、パッケージ名の解決に頼らない。
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "build_test_wav_from_tsv.py"
    if not path.is_file():
        raise ModuleNotFoundError(f"見つかりません: {path}")
    spec = importlib.util.spec_from_file_location("_miltoka_build_test_wav", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_test_wav = _load_build_test_wav_module()
CLIP_MARGIN_SEC = _build_test_wav.CLIP_MARGIN_SEC
read_annotation = _build_test_wav.read_annotation

TASK = "real_response"
PROTOCOL_NAME = "Real dialogue single-turn response (annotation ground truth)"

# 相槌とみなす短い応答の語彙。judge 側で相槌かどうかを見分けられるようにする
# だけで、指標の分岐には使わない。
_AIZUCHI = ("はい", "ええ", "うん", "そう", "なるほど", "へえ", "ふん", "ああ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-data-dir", type=Path,
                        default=Path("data/test_data/real_dialogue"),
                        help="build_test_wav_from_tsv.py の出力ルート")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/eval_sets/real_response"),
                        help="データセットの出力先")
    parser.add_argument("--context-sec", type=float, default=0.0,
                        help="User 発話の手前に付ける文脈の秒数(既定 0)。"
                             "0 より大きくすると相談員の声が入力に混ざる")
    parser.add_argument("--gold-window-sec", type=float, default=30.0,
                        help="Staff の応答として拾う最大秒数(既定 30)")
    parser.add_argument("--gold-reply-gap-sec", type=float, default=3.0,
                        help="連続する Staff 発話を 1 つの応答としてまとめる間隔")
    parser.add_argument("--gold-next-user-closes", action="store_true",
                        help="次の User 発話で応答窓を閉じる(旧挙動)。既定では"
                             "閉じない。User の 1 ターンが複数行に割れていると"
                             "窓が潰れて無応答になるため")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_aizuchi(text: str) -> bool:
    stripped = text.strip().rstrip("。、.,!！?？")
    return bool(stripped) and stripped in _AIZUCHI


def load_clips(dialogue_dir: Path) -> list[dict[str, str]]:
    path = dialogue_dir / "clips.tsv"
    if not path.is_file():
        raise SystemExit(f"{path} がありません。先にテストデータを作ってください。")
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def gold_response(
    utterances: list, user_end: float, next_user_start: float | None,
    window_sec: float, target_speaker: str, reply_gap_sec: float = 3.0,
    next_user_closes: bool = False,
) -> list:
    """User 発話の直後に相談員が返した発話を拾う。

    起点は user_end。そこから window_sec 以内で**最初に始まる Staff 発話**を
    応答の頭とし、以降 reply_gap_sec 以内で続く Staff 発話を同じ応答として
    まとめる。複数発話にまたがることがあるのでリストで返す。

    次の User 発話では窓を閉じない。1ch のアノテーションでは User の 1 ターンが
    複数行に割れていることが多く、次の User 行が user_end の直後に来るだけで
    窓の幅がほぼ 0 になり、相談員が実際に応答していても「無応答」に落ちて
    いた。ユーザーが喋り続けたことは、相談員が応答しなかったことを意味しない。

    next_user_closes=True で以前の挙動(次の User 発話で窓を閉じる)に戻せる。
    """
    limit = user_end + window_sec
    if next_user_closes and next_user_start is not None:
        limit = min(limit, next_user_start)

    candidates = sorted(
        (u for u in utterances
         if u.speaker.lower() != target_speaker.lower()
         and u.end > user_end and u.start < limit),
        key=lambda u: u.start,
    )
    if not candidates:
        return []

    replies = [candidates[0]]
    for u in candidates[1:]:
        if u.start - replies[-1].end > reply_gap_sec:
            break
        replies.append(u)
    return replies


def build_dialogue(
    summary: dict, out_root: Path, args: argparse.Namespace
) -> list[dict]:
    name = summary["name"]
    dialogue_dir = Path(summary["out_dir"])
    wav_path = Path(summary["wav"])
    annotation = Path(summary["annotation"])
    target = summary["speaker"]

    utterances = read_annotation(annotation)
    user_starts = sorted(
        u.start for u in utterances if u.speaker.lower() == target.lower()
    )

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    samples: list[dict] = []
    for clip in load_clips(dialogue_dir):
        index = int(clip["index"])
        start = float(clip["start_sec"])
        end = float(clip["end_sec"])

        next_user = next((s for s in user_starts if s > start), None)
        replies = gold_response(
            utterances, end, next_user, args.gold_window_sec, target,
            reply_gap_sec=args.gold_reply_gap_sec,
            next_user_closes=args.gold_next_user_closes,
        )

        case_id = f"{name}__{index:04d}"
        case_dir = out_root / TASK / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # 入力: User の発話。--context-sec を付けた分だけ手前も含める。
        lo = max(0, int((start - args.context_sec - CLIP_MARGIN_SEC) * sr))
        hi = min(audio.size, int((end + CLIP_MARGIN_SEC) * sr))
        if hi <= lo:
            continue
        sf.write(case_dir / "input.wav", audio[lo:hi], sr)
        input_sec = (hi - lo) / sr

        latency = round(replies[0].start - end, 4) if replies else None
        reply_audio = None
        if replies:
            rlo = max(0, int((replies[0].start - CLIP_MARGIN_SEC) * sr))
            rhi = min(audio.size, int((replies[-1].end + CLIP_MARGIN_SEC) * sr))
            if rhi > rlo:
                reply_audio = {
                    "start_sec": round(rlo / sr, 4),
                    "end_sec": round(rhi / sr, 4),
                }

        metadata = {
            "case_id": case_id,
            "task": TASK,
            "language": "ja",
            "expected_behavior": (
                "ユーザーの発話が終わったら、間を置きすぎずに応答を返す。"
            ),
            "source": {
                "dialogue": name,
                "wav": str(wav_path),
                "annotation": str(annotation),
                "user_speaker": target,
                "user_index": index,
                "user_start_sec": start,
                "user_end_sec": end,
                "user_text": clip.get("text", ""),
                "user_overlap_sec": float(clip.get("overlap_sec", 0.0) or 0.0),
                "context_sec": args.context_sec,
                "input_duration_sec": round(input_sec, 4),
                # 評価点 = User の発話が終わった瞬間。応答速度はここが起点。
                "eval_start_sec": end,
                "eval_end_sec": round(end + args.gold_window_sec, 4),
            },
            "human_reference": {
                "responded": bool(replies),
                "response_latency_sec": latency,
                # 応答が始まる前に User が次の行を喋り出していたか。応答窓は
                # これで閉じないが、後から絞り込めるように残しておく。
                "user_continued_before_reply": bool(
                    replies and next_user is not None
                    and next_user < replies[0].start
                ),
                "next_user_start_sec": (
                    round(next_user, 4) if next_user is not None else None
                ),
                "audio": reply_audio,
                "segments": [
                    {
                        "start_sec": round(u.start, 4),
                        "end_sec": round(u.end, 4),
                        "text": u.text,
                        "is_aizuchi": is_aizuchi(u.text),
                    }
                    for u in replies
                ],
            },
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        samples.append({
            "task": TASK,
            "id": case_id,
            "path": f"{TASK}/{case_id}",
            "dialogue": name,
            "human_responded": bool(replies),
        })

    return samples


def main() -> int:
    args = parse_args()
    manifest_path = args.test_data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} がありません。\n"
            "先に scripts/build_test_wav_from_tsv.py を実行してください。"
        )
    source = json.loads(manifest_path.read_text(encoding="utf-8"))

    out_root = args.out_dir
    if out_root.exists() and any(out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"{out_root} は空ではありません。作り直すなら --overwrite を付けてください。"
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for summary in source["dialogues"]:
        built = build_dialogue(summary, out_root, args)
        print(f"[real-set] {summary['name']}: {len(built)} ケース")
        samples.extend(built)

    if not samples:
        raise SystemExit("ケースが 1 件も作れませんでした。")

    responded = sum(1 for s in samples if s["human_responded"])
    manifest = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "note": (
                "入力は実対話の User 発話 1 つ。正解は同じ場面で相談員が実際に"
                "返した応答。合成 Full-Duplex-Bench-JA とは別トラックであり、"
                "スコアを同じ表に並べてはいけない。"
            ),
        },
        "source_test_data": str(args.test_data_dir),
        "context_sec": args.context_sec,
        "gold_window_sec": args.gold_window_sec,
        "totals": {
            "cases": len(samples),
            "dialogues": len(source["dialogues"]),
            "human_responded": responded,
        },
        "samples": samples,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[real-set] 合計 {len(samples)} ケース -> {out_root}")
    print(f"[real-set] うち相談員が実際に応答したもの: {responded} 件")
    if responded < len(samples):
        print(
            f"[real-set] 注意: {len(samples) - responded} 件は相談員も応答して"
            "いません。gold の応答率はこの分だけ 100% を下回ります。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
