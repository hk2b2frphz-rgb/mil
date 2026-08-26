#!/usr/bin/env python3
"""クローン参照(refNN)の WAV パスと書き起こしを 1 箇所で解決する。

対話合成(generate_qwen3_tts_data.py)へクローン参照を渡すとき、参照 WAV と
その書き起こしを手で転記するのは事故のもと。日本語のテキストをシェル引数へ
貼り付けることになり、引用符・記号・改行で簡単に壊れる。参照の選別ロジックは
clone_voice_examples.py と共有し、ここでは「取り出すだけ」を担う。

解決元は 2 通り:

  --analysis-dir   analyze_real_dialogue.py の出力から select_references で
                   再選別する。ソートが安定なので、同じ timeline.jsonl なら
                   clone_voice_examples.py が選んだ refNN と必ず一致する。

  --clone-out-dir  clone_voice_examples.py の出力から refNN の manifest.json を
                   読む。参照 WAV は出力先に reference.wav としてコピー済みな
                   ので、元の analysis_dir が消えていてもこちらは再現できる。
                   「あの時の refNN」を厳密に使いたい場合はこちら。

使い方(1 フィールドずつシェル変数へ):

    WAV=$(uv run python scripts/resolve_clone_refs.py \
        --analysis-dir data/real_dialogue/<jobid>/<stem> --speaker B --field wav)
    TEXT=$(uv run python scripts/resolve_clone_refs.py \
        --analysis-dir data/real_dialogue/<jobid>/<stem> --speaker B --field text)

--field json(既定)なら全項目を JSON で出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.clone_voice_examples import (
        DEFAULT_MAX_REF_SEC,
        DEFAULT_MIN_REF_SEC,
        find_segment_wav,
        load_timeline,
        select_references,
    )
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from clone_voice_examples import (  # type: ignore[no-redef]
        DEFAULT_MAX_REF_SEC,
        DEFAULT_MIN_REF_SEC,
        find_segment_wav,
        load_timeline,
        select_references,
    )


def resolve_from_analysis_dir(
    analysis_dir: Path, speaker: str, rank: int, min_sec: float, max_sec: float
) -> dict[str, Any]:
    segs = load_timeline(analysis_dir)
    # rank 番目が要るので、そこまでの順位を選別させる。
    chosens = select_references(segs, speaker, min_sec, max_sec, rank + 1)
    if len(chosens) <= rank:
        raise SystemExit(
            f"話者 {speaker} の参照候補が {len(chosens)} 件しかありません "
            f"(ref{rank:02d} は取れません)。--min-ref-sec/--max-ref-sec を緩めるか、"
            "別の話者を指定してください。"
        )
    chosen = chosens[rank]
    wav = find_segment_wav(analysis_dir, speaker, chosen["_index"])
    if wav is None:
        raise SystemExit(
            f"区間 {chosen['_index']:04d} の WAV が {analysis_dir}/segments/{speaker}/ に"
            "見つかりません。analyze_real_dialogue.py の出力が欠けています。"
        )
    return {
        "source": "analysis-dir",
        "speaker": speaker,
        "rank": rank,
        "wav": str(wav),
        "text": str(chosen["text"]).strip(),
        "timeline_index": chosen["_index"],
        "duration_sec": round(float(chosen["end"]) - float(chosen["start"]), 2),
    }


def resolve_from_clone_out_dir(
    clone_out_dir: Path, rank: int, mode: str
) -> dict[str, Any]:
    tag = f"ref{rank:02d}_{mode}"
    subdir = clone_out_dir / tag
    manifest = subdir / "manifest.json"
    if not manifest.is_file():
        available = sorted(p.name for p in clone_out_dir.glob("ref*_*") if p.is_dir())
        raise SystemExit(
            f"{manifest} がありません。"
            + (f" このディレクトリにあるのは: {available}" if available else "")
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    reference = data.get("reference") or {}
    # 元パスは消えている可能性があるので、出力先へコピーされた実体を優先する。
    copied = subdir / "reference.wav"
    wav = copied if copied.is_file() else Path(str(reference.get("wav", "")))
    if not wav.is_file():
        raise SystemExit(f"参照 WAV が見つかりません: {wav}")
    return {
        "source": "clone-out-dir",
        "tag": tag,
        "rank": rank,
        "mode": mode,
        "wav": str(wav),
        "text": str(reference.get("text", "")).strip(),
        "timeline_index": reference.get("timeline_index"),
        "duration_sec": reference.get("duration_sec"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--analysis-dir", type=Path,
                     help="analyze_real_dialogue.py の出力 <wav_stem>/")
    src.add_argument("--clone-out-dir", type=Path,
                     help="clone_voice_examples.py の出力ディレクトリ")
    parser.add_argument("--speaker", default="A",
                        help="--analysis-dir 使用時の話者 A / B(既定 A)")
    parser.add_argument("--rank", type=int, default=0,
                        help="参照の順位。0 = ref00(既定)")
    parser.add_argument("--mode", default="in-context", choices=["in-context", "x-vector"],
                        help="--clone-out-dir 使用時の refNN_<mode> サブディレクトリ")
    parser.add_argument("--min-ref-sec", type=float, default=DEFAULT_MIN_REF_SEC)
    parser.add_argument("--max-ref-sec", type=float, default=DEFAULT_MAX_REF_SEC)
    parser.add_argument("--field", default="json",
                        choices=["json", "wav", "text", "timeline_index", "duration_sec"],
                        help="出力する項目。既定は全項目の JSON。")
    args = parser.parse_args()

    if args.rank < 0:
        parser.error("--rank must be >= 0")

    if args.analysis_dir is not None:
        info = resolve_from_analysis_dir(
            args.analysis_dir, args.speaker, args.rank,
            args.min_ref_sec, args.max_ref_sec,
        )
    else:
        info = resolve_from_clone_out_dir(args.clone_out_dir, args.rank, args.mode)

    if args.field == "json":
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    value = info.get(args.field)
    if value is None or (isinstance(value, str) and not value):
        raise SystemExit(f"{args.field} が空です: {json.dumps(info, ensure_ascii=False)}")
    print(value)


if __name__ == "__main__":
    main()
