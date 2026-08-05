#!/usr/bin/env python3
"""
Audacity のラベル書き出しを、アノテーション TSV へ変換する。

入力は Audacity の File > Export > Export Labels が吐く形式:

    開始秒<TAB>終了秒<TAB>話者: 書き起こし

    15.500000	21.200000	User: もしもし、あの、相談したいことがありまして
    21.000000	21.600000	Staff: はい

出力は scripts/build_test_wav_from_tsv.py が読む形式:

    Speaker	Transcription	Start	End
    User	もしもし、あの、相談したいことがありまして	00:00:15.500	00:00:21.200
    Staff	はい	00:00:21.000	00:00:21.600

話者はラベル文字列の**最初のコロンまで**を見る(全角コロンも可)。書き起こし側に
コロンが入っていても、最初の 1 つでしか切らないので影響しない。既定で認める
話者は Staff と User。それ以外が出てきたら、綴り間違いを黙って通さないために
行番号付きで止める(意図したものなら --speakers で足す)。

使い方(フォルダを渡すと中の .txt を全部処理する):
    uv run python scripts/labels_to_annotation_tsv.py /path/labels/

    # 音声の隣へ直接出す
    uv run python scripts/labels_to_annotation_tsv.py /path/labels/ --out-dir /path/dialogues/

出力は入力と同じ名前で拡張子を .tsv にしたもの。**音声と同じ名前で音声の隣に
置く**必要があるので、名前が違うならリネームすること。既存の .tsv は
--overwrite を付けない限り上書きしない。

GPU も外部ライブラリも要らない。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

try:
    from scripts.build_test_wav_from_tsv import (  # noqa: E402
        AnnotationError,
        parse_timestamp,
        write_tsv,
    )
except ImportError:  # scripts/ を直接 sys.path に持つ場合
    from build_test_wav_from_tsv import (  # type: ignore[no-redef]  # noqa: E402
        AnnotationError,
        parse_timestamp,
        write_tsv,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SPEAKERS = ("Staff", "User")
_COLONS = (":", "：")


def format_time(sec: float) -> str:
    """00:00:15.500 形式。build_test_wav_from_tsv がこの形式を読む。"""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def split_speaker(label: str) -> tuple[str, str]:
    """ラベル文字列を (話者, 書き起こし) に割る。最初のコロンでしか切らない。"""
    positions = [label.find(c) for c in _COLONS if label.find(c) >= 0]
    if not positions:
        return "", label.strip()
    at = min(positions)
    return label[:at].strip(), label[at + 1:].strip()


def read_labels(path: Path, speakers: tuple[str, ...]) -> list[tuple[float, float, str, str]]:
    """ラベルファイルを (start, end, speaker, text) の列にする。

    問題のある行は**1 つも書き出す前に**全部挙げてから止める。ここを黙って
    通すと、壊れた行だけが静かに落ちたテストデータができる。
    """
    canonical = {s.lower(): s for s in speakers}
    rows: list[tuple[float, float, str, str]] = []
    errors: list[str] = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        # スペクトル選択付きで書き出すと、ラベルごとに "\ 低域 高域" の行が
        # 挟まる。周波数の情報なのでここでは読み飛ばす。
        if raw.lstrip().startswith("\\"):
            continue

        fields = raw.split("\t")
        if len(fields) < 3:
            errors.append(f"{lineno} 行目: 列が {len(fields)} 個しかありません: {raw!r}")
            continue

        try:
            start = parse_timestamp(fields[0], lineno, "開始")
            end = parse_timestamp(fields[1], lineno, "終了")
        except AnnotationError as exc:
            errors.append(str(exc))
            continue
        if end <= start:
            errors.append(
                f"{lineno} 行目: 終了 <= 開始 です ({fields[0]} -> {fields[1]})。"
                "点ラベル(長さ 0)は区間にしてください。"
            )
            continue

        # ラベル文字列にタブは入らないが、貼り付け由来で混ざっていても拾う。
        speaker, text = split_speaker("\t".join(fields[2:]).replace("\t", " "))
        if not speaker:
            errors.append(f"{lineno} 行目: 話者がありません(「話者: 本文」の形): {raw!r}")
            continue
        if speaker.lower() not in canonical:
            errors.append(
                f"{lineno} 行目: 知らない話者 {speaker!r} です"
                f"(認めているのは {list(speakers)})。"
            )
            continue
        if not text:
            errors.append(f"{lineno} 行目: 書き起こしが空です: {raw!r}")
            continue

        rows.append((start, end, canonical[speaker.lower()], text))

    if errors:
        for message in errors:
            logger.error("%s", message)
        raise SystemExit(f"{path}: {len(errors)} 行に問題があります。直してから再実行してください。")
    if not rows:
        raise SystemExit(f"{path}: ラベルが 1 つもありません。")
    return rows


def convert(path: Path, out_dir: Path | None, speakers: tuple[str, ...], overwrite: bool) -> dict:
    rows = read_labels(path, speakers)
    # Audacity は開始時刻順に書き出すが、手で編集すると崩れることがある。
    # アノテーション TSV は時間順に並んでいる前提なので、ここで揃える。
    rows.sort(key=lambda r: (r[0], r[1]))

    destination = (out_dir or path.parent) / f"{path.stem}.tsv"
    if destination.exists() and not overwrite:
        raise SystemExit(
            f"{destination} が既にあります。上書きするなら --overwrite を付けてください。"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(
        destination,
        ["Speaker", "Transcription", "Start", "End"],
        [[speaker, text, format_time(start), format_time(end)]
         for start, end, speaker, text in rows],
    )

    per_speaker: dict[str, int] = {}
    for _, _, speaker, _ in rows:
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
    # 異話者どうしの重なりは 1ch 対話では正常。件数だけ見えるようにしておく。
    overlaps = sum(
        1 for i, a in enumerate(rows)
        for b in rows[i + 1:]
        if b[0] < a[1] and b[2] != a[2]
    )
    spoken = sum(end - start for start, end, _, _ in rows)
    logger.info(
        "%s -> %s (%d 行 / %s / 発話 %.1f 秒 / 異話者の重なり %d 組)",
        path.name, destination, len(rows),
        " ".join(f"{k}:{v}" for k, v in sorted(per_speaker.items())),
        spoken, overlaps,
    )
    return {
        "source": str(path),
        "tsv": str(destination),
        "rows": len(rows),
        "per_speaker": per_speaker,
        "spoken_sec": round(spoken, 1),
        "cross_speaker_overlaps": overlaps,
        "last_end_sec": round(rows[-1][1], 3),
    }


def collect(inputs: list[Path], pattern: str) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            matched = sorted(p for p in item.glob(pattern) if p.is_file())
            if not matched:
                raise SystemExit(f"{item}: {pattern} に合うファイルがありません。")
            found.extend(matched)
        elif item.is_file():
            found.append(item)
        else:
            raise SystemExit(f"見つかりません: {item}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="ラベルファイル、またはそれが入ったフォルダ")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="出力先(既定: 入力と同じ場所)。音声の隣に出すなら音声のフォルダを指定")
    parser.add_argument("--pattern", default="*.txt",
                        help="フォルダを渡したときに拾うパターン(既定: *.txt)")
    parser.add_argument("--speakers", default=",".join(DEFAULT_SPEAKERS),
                        help=f"認める話者をカンマ区切りで(既定: {','.join(DEFAULT_SPEAKERS)})")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存の .tsv を上書きする")
    args = parser.parse_args()

    speakers = tuple(s.strip() for s in args.speakers.split(",") if s.strip())
    if not speakers:
        parser.error("--speakers が空です。")

    paths = collect(args.inputs, args.pattern)
    results = [convert(path, args.out_dir, speakers, args.overwrite) for path in paths]

    print("\n=== ラベル -> アノテーション TSV ===")
    for result in results:
        print(
            f"  {Path(result['tsv']).name:<28} {result['rows']:>4} 行"
            f"  {result['spoken_sec']:>7.1f} 秒"
            f"  最終 {result['last_end_sec']:.1f}s"
            f"  重なり {result['cross_speaker_overlaps']}"
        )
    print(f"  合計 {sum(r['rows'] for r in results)} 行 / {len(results)} ファイル")
    print("\nTSV は**音声と同じ名前で音声の隣**に置いてください。そのあと:")
    print("  uv run python scripts/build_test_wav_from_tsv.py /path/dialogues/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
