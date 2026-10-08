#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""複数コーパスの merged マニフェストを比率指定で 1 本に混ぜる。

なぜ必要か:
    aizuchi-only コーパスは sanitize_aizuchi_only_turns が moshi 側の語彙外
    発話を機械的に落とすため、学習後のモデルは「はい。」「そうですか…。」
    しか発話経験がない。外部 LLM の回答をテキストストリームに注入して読ま
    せようとしても、対応する acoustic コードを持たず、実トークンを連続で
    出す挙動も学んでいないので破綻する。

    そこで multi-agent モード（systemAI の本応答を含む）で作ったコーパスを
    少量混ぜ、「相づちを打ち続ける」挙動と「本応答を喋る」挙動の両方を 1 つ
    のモデルに持たせる。混合比がそのまま実験変数になるので、比率を指定でき
    る形にしてある。

merge_training_shards.py との違い:
    あちらは 1 バッチ内の shard_* を全部連結する（比率の概念がない）。こちら
    は別バッチ同士を、ソースごとの採用件数を指定して混ぜる。音声はコピーせず
    絶対パスで参照するのは同じで、prepare_nu_fullft_dataset.py がそのまま
    読める形式（training_set/synthetic_moshi_train.jsonl）で書き出す。

使い方:
    uv run python scripts/mix_training_manifests.py \
        --source aizuchi=data/runs/aizuchi_normal_10000_v2/tts/merged \
        --source response=data/runs/response_2000_v1/tts/merged \
        --take   aizuchi=all \
        --take   response=2000 \
        --out-dir data/runs/mixed_normal_10000_v1/mixed

    --take は "all" / 整数（件数）/ 0<x<=1 の小数（そのソースに対する割合）。
    指定しなかったソースは all 扱い。

    学習側はできた out-dir をそのまま SRC_RUN_DIR に渡す。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


MANIFEST_NAME = "synthetic_moshi_train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix merged training manifests from multiple corpora."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "混合するコーパス。PATH は training_set/ を含む run ディレクトリ"
            "（TTS ジョブの <OUT_ROOT>/merged）。training_set 自体を渡しても可。"
            "複数回指定する。"
        ),
    )
    parser.add_argument(
        "--take",
        action="append",
        default=[],
        metavar="LABEL=SPEC",
        help="採用件数。all / 整数 / 0<x<=1 の割合。省略時は all。",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="混合後のシャッフルを行わず、ソース順に並べる（デバッグ用）。",
    )
    parser.add_argument(
        "--duration-warn-sec",
        type=float,
        default=170.0,
        help=(
            "この秒数を超えるサンプルの割合をサマリに出す。moshi-finetune は"
            "duration_sec を超える対話を等分割し、2 個目以降のチャンクが冒頭"
            "挨拶なしで始まるため、学習前に必ず確認する値。"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="wav / sidecar JSON が欠けた行があっても続行する。",
    )
    parser.add_argument(
        "--allow-duplicate-stems",
        action="store_true",
        help=(
            "ソース間で wav の stem が衝突しても続行する（先勝ちで 1 件のみ"
            "採用）。prepare_nu_fullft_dataset.py は stem で出力先を決めるので、"
            "既定では衝突を検出したら失敗する。"
        ),
    )
    return parser.parse_args()


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"ERROR: {flag} must be LABEL=VALUE: {item}")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"ERROR: empty label in {flag}: {item}")
        if label in out:
            raise SystemExit(f"ERROR: duplicate label in {flag}: {label}")
        out[label] = value.strip()
    return out


def resolve_training_set(path: Path) -> Path:
    """run ディレクトリと training_set ディレクトリのどちらを渡されても受ける。"""
    path = path.resolve()
    if (path / MANIFEST_NAME).is_file():
        return path
    candidate = path / "training_set"
    if (candidate / MANIFEST_NAME).is_file():
        return candidate
    raise SystemExit(
        f"ERROR: no {MANIFEST_NAME} under {path} (nor {path}/training_set)"
    )


def load_rows(
    training_set: Path, manifest_name: str, allow_missing: bool, label: str
) -> tuple[list[dict[str, Any]], int]:
    manifest = training_set / manifest_name
    rows: list[dict[str, Any]] = []
    dropped = 0
    for lineno, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        raw = str(row.get("path") or "")
        if not raw:
            dropped += 1
            continue
        wav = Path(raw)
        if not wav.is_absolute():
            wav = training_set / wav
        wav = wav.resolve()
        if not wav.exists():
            print(f"[mix] {label}: line {lineno}: wav not found: {wav}")
            dropped += 1
            continue
        if not wav.with_suffix(".json").exists():
            print(f"[mix] {label}: line {lineno}: sidecar JSON missing for {wav.name}")
            dropped += 1
            continue
        rows.append(
            {
                "path": str(wav),
                "duration": float(row.get("duration") or 0.0),
                "_label": label,
                "_stem": wav.stem,
            }
        )
    if dropped and not allow_missing:
        raise SystemExit(
            f"ERROR: {label}: {dropped} row(s) referenced a missing wav/sidecar. "
            f"Re-run the failed TTS shards, or pass --allow-missing."
        )
    return rows, dropped


def resolve_take(spec: str, available: int, label: str) -> int:
    spec = spec.strip()
    if spec in ("", "all"):
        return available
    try:
        value = float(spec)
    except ValueError:
        raise SystemExit(f"ERROR: --take {label}={spec} is not all/int/float")
    if value <= 0:
        raise SystemExit(f"ERROR: --take {label}={spec} must be positive")
    # 小数表記かつ 1 以下なら割合、それ以外は件数。
    if value <= 1.0 and "." in spec:
        return max(1, round(available * value))
    count = int(value)
    if count > available:
        raise SystemExit(
            f"ERROR: --take {label}={spec} exceeds available rows ({available}). "
            f"Generate more data, or lower the count."
        )
    return count


def duration_stats(values: list[float], warn_sec: float) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    over = sum(1 for v in ordered if v > warn_sec)
    return {
        "count": n,
        "hours": round(sum(ordered) / 3600.0, 3),
        "p50_sec": round(ordered[n // 2], 1),
        "p95_sec": round(ordered[min(n - 1, int(n * 0.95))], 1),
        "max_sec": round(ordered[-1], 1),
        "warn_sec": warn_sec,
        "over_warn_pct": round(100.0 * over / n, 2),
    }


def main() -> int:
    args = parse_args()
    sources = parse_pairs(args.source, "--source")
    takes = parse_pairs(args.take, "--take")
    if not sources:
        raise SystemExit("ERROR: at least one --source LABEL=PATH is required")
    unknown = set(takes) - set(sources)
    if unknown:
        raise SystemExit(f"ERROR: --take refers to unknown labels: {sorted(unknown)}")

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []
    dialogue_lines: list[str] = []

    for label, raw_path in sources.items():
        training_set = resolve_training_set(Path(raw_path))
        rows, dropped = load_rows(
            training_set, args.manifest_name, args.allow_missing, label
        )
        take = resolve_take(takes.get(label, "all"), len(rows), label)
        picked = rows if take >= len(rows) else rng.sample(rows, take)
        selected.extend(picked)

        dialogues = training_set / "dialogues.jsonl"
        if dialogues.is_file():
            dialogue_lines.extend(
                line
                for line in dialogues.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        stats = duration_stats([r["duration"] for r in picked], args.duration_warn_sec)
        per_source.append(
            {
                "label": label,
                "training_set": str(training_set),
                "available": len(rows),
                "dropped": dropped,
                "taken": len(picked),
                **stats,
            }
        )
        print(
            f"[mix] {label}: took {len(picked)}/{len(rows)} "
            f"({stats.get('hours', 0)} h, p95={stats.get('p95_sec')}s, "
            f"over{args.duration_warn_sec:.0f}s={stats.get('over_warn_pct')}%)"
        )

    # stem 衝突チェック。prepare_nu_fullft_dataset.py は wav の stem で出力先を
    # 決めるので、衝突したまま渡すと片方が静かに上書きされる。
    seen: dict[str, str] = {}
    collisions: list[str] = []
    deduped: list[dict[str, Any]] = []
    for row in selected:
        stem = row["_stem"]
        if stem in seen:
            collisions.append(f"{stem} ({seen[stem]} vs {row['_label']})")
            continue
        seen[stem] = row["_label"]
        deduped.append(row)
    if collisions:
        head = collisions[:10]
        if args.allow_duplicate_stems:
            print(f"[mix] WARNING: {len(collisions)} duplicate stem(s), kept first: {head}")
        else:
            print(f"[mix] ERROR: {len(collisions)} duplicate wav stem(s): {head}")
            print("[mix] Rename one corpus's wavs, or pass --allow-duplicate-stems.")
            return 1
    selected = deduped

    if not selected:
        print("[mix] ERROR: no samples selected")
        return 1

    if not args.no_shuffle:
        rng.shuffle(selected)

    out_training = (args.out_dir / "training_set").resolve()
    out_training.mkdir(parents=True, exist_ok=True)
    out_manifest = out_training / args.manifest_name
    with out_manifest.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(
                json.dumps(
                    {"path": row["path"], "duration": row["duration"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    if dialogue_lines:
        (out_training / "dialogues.jsonl").write_text(
            "\n".join(dialogue_lines) + "\n", encoding="utf-8"
        )

    mixed_stats = duration_stats([r["duration"] for r in selected], args.duration_warn_sec)
    summary = {
        "out_dir": str(args.out_dir.resolve()),
        "manifest": str(out_manifest),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "sources": per_source,
        "mixed": mixed_stats,
        "mix_ratio": {
            entry["label"]: round(entry["taken"] / len(selected), 4)
            for entry in per_source
        },
        "duplicate_stems": len(collisions),
    }
    (args.out_dir / "mix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[mix] wrote {len(selected)} samples ({mixed_stats['hours']} h) -> {out_manifest}")
    print(f"[mix] ratio: {summary['mix_ratio']}")
    if mixed_stats["over_warn_pct"] > 0:
        print(
            f"[mix] NOTE: {mixed_stats['over_warn_pct']}% of samples exceed "
            f"{args.duration_warn_sec:.0f}s and will be split into chunks by "
            f"moshi-finetune (chunks after the first start mid-conversation)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
