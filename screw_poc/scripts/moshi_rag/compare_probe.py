#!/usr/bin/env python3
"""3条件の推論結果を並べて、参照が発話を変えたかどうかを判定する。

run_inference は 1 WAV につき out_<variant>/<stem>.json を書く。中身は
inference_job.py の trace で、model_text（moshi が喋ったテキスト）、
reference_text（実際に注入された参照）、rag_trigger_step などが入っている。

判定:
    reads_reference … true と altered で moshi の発話が違う。参照を読んでいる。
    memorised       … 参照を変えても発話が同じ。重みの中の知識で答えている。
    no_answer       … どちらの条件でも値を言っていない。判定に使えない。

empty 条件は別枠で見る。参照が無いときに数値を口にしたら、それは作話。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

VARIANTS = ("true", "altered", "empty")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def spoken(trace: dict[str, Any] | None) -> str:
    if not trace:
        return ""
    text = trace.get("model_text") or []
    if isinstance(text, list):
        return "".join(str(part) for part in text).strip()
    return str(text).strip()


def numbers(text: str) -> list[str]:
    """規格名の数字まで拾うと差が埋もれるので、M8 のような表記は落とす。"""
    return NUMBER.findall(re.sub(r"\bM\s?\d+\b", " ", text, flags=re.IGNORECASE))


def main() -> None:
    args = parse_args()
    manifest_path = args.probe_dir / "probe_manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")

    ids: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in ids:
            ids.append(row["id"])

    results: list[dict[str, Any]] = []
    for probe_id in ids:
        traces = {
            variant: load_trace(args.probe_dir / f"out_{variant}" / f"{probe_id}.json")
            for variant in VARIANTS
        }
        said = {variant: spoken(trace) for variant, trace in traces.items()}
        true_numbers, altered_numbers = numbers(said["true"]), numbers(said["altered"])

        if not true_numbers and not altered_numbers:
            verdict = "no_answer"
        elif true_numbers != altered_numbers:
            verdict = "reads_reference"
        else:
            verdict = "memorised"

        results.append(
            {
                "id": probe_id,
                "verdict": verdict,
                "spoken": said,
                "numbers": {"true": true_numbers, "altered": altered_numbers,
                            "empty": numbers(said["empty"])},
                "injected": {
                    variant: (traces[variant] or {}).get("reference_text", "")
                    for variant in VARIANTS
                },
                "rag_trigger_step": {
                    variant: (traces[variant] or {}).get("rag_trigger_step", -1)
                    for variant in VARIANTS
                },
            }
        )

    print("=" * 78)
    print("MoshiRAG reference probe")
    print("=" * 78)
    for row in results:
        print(f"\n[{row['id']}]  verdict={row['verdict']}")
        for variant in VARIANTS:
            print(f"  {variant:<8} said: {row['spoken'][variant][:150] or '(nothing)'}")
        print(f"  numbers   true={row['numbers']['true']} "
              f"altered={row['numbers']['altered']} empty={row['numbers']['empty']}")
        if row["numbers"]["empty"]:
            print("  WARNING: spoke a number with no reference injected (possible confabulation)")

    counts = {verdict: sum(1 for row in results if row["verdict"] == verdict)
              for verdict in ("reads_reference", "memorised", "no_answer")}
    print("\n" + "-" * 78)
    print(json.dumps(counts, ensure_ascii=False))
    if counts["reads_reference"]:
        print("At least one probe followed the injected reference: knowledge can live "
              "outside the weights.")
    elif counts["memorised"]:
        print("No probe followed the injected reference: the model answered from memory.")

    (args.probe_dir / "probe_result.json").write_text(
        json.dumps({"counts": counts, "probes": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"written: {args.probe_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
