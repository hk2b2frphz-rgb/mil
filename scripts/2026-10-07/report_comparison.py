#!/usr/bin/env python3
"""Three-column results; reject differing trial sets or partial benchmarks."""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aizuchi_comparison import ARMS, is_backchannel, read_rows, write_json


def numeric_leaves(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(numeric_leaves(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    return out


def corpus_stats(rows):
    count = repeats = adjacent = 0
    for row in rows:
        last_word = None
        last_backchannel = False
        for turn in row["turns"]:
            current = is_backchannel(turn)
            if current:
                count += 1
                repeats += turn["text"] == last_word
                adjacent += last_backchannel
                last_word = turn["text"]
            last_backchannel = current
    return {"dialogues": len(rows), "backchannels": count,
            "backchannels_per_dialogue": count / len(rows),
            "same_phrase_as_previous_backchannel": repeats,
            "adjacent_backchannel_turns": adjacent}


def output_repeat_stats(cases, vocab):
    # Longest phrase first: a vocabulary entry such as "hai, hai" counts as
    # one phrase, rather than two separate instances of "hai".
    normalize = lambda text: re.sub(r"[^\w]", "", text)
    words = {normalize(word) for word in vocab if normalize(word)}
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w))))
    selected = [r for r in cases if r["task"] == "backchannel"]
    if not selected:
        return {}
    phrases = repeats = 0
    for row in selected:
        matches = pattern.findall(normalize(row.get("assistant_text", "")))
        phrases += len(matches)
        repeats += sum(a == b for a, b in zip(matches, matches[1:]))
    return {"trials": len(selected), "vocab_phrase_count": phrases,
            "vocab_phrases_per_trial": phrases / len(selected),
            "consecutive_same_vocab_phrase": repeats}


def build_report(root, ident, repo, allow_pending=False):
    sources = {a: repo / "eval_runs/full_duplex" / f"{ident}_{a}" / "benchmark_results" for a in ARMS}
    missing = [a for a, p in sources.items() if not (p / "summary.json").is_file()]
    if missing:
        if allow_pending:
            print(f"[comparison] pending: {', '.join(missing)}")
            return
        raise ValueError(f"Missing evaluation results: {missing}")
    summaries = {a: json.loads((p / "summary.json").read_text(encoding="utf-8")) for a, p in sources.items()}
    keys = []
    cases_by_arm = {}
    for arm, summary in summaries.items():
        if summary.get("evaluation", {}).get("status") != "complete":
            raise ValueError(f"{arm}: partial evaluation cannot be compared")
        cases = read_rows(sources[arm] / "per_case.jsonl")
        cases_by_arm[arm] = cases
        case_keys = [(r["task"], r["case_id"], r["seed"]) for r in cases]
        if len(set(case_keys)) != len(case_keys) or len(cases) != summary["n"]:
            raise ValueError(f"{arm}: duplicate/missing trial results")
        keys.append(set(case_keys))
    if not keys[0] or any(k != keys[0] for k in keys[1:]):
        raise ValueError("Benchmark trial IDs/seeds differ across models")
    source = read_rows(root / "shared/dialogues.jsonl")
    ai = read_rows(root / "ai_placement/dialogue/llm_dialogues/dialogues.jsonl")
    metrics = {}
    vocab = [line.split("\t")[0] for line in (repo / "scripts/2026-09-15/aizuchi_vocab_no_bare_un.tsv").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    for arm in ARMS:
        metrics[arm] = numeric_leaves(summaries[arm]["tasks"], "benchmark")
        metrics[arm].update(numeric_leaves(corpus_stats(ai if arm == "ai_placement" else source), "training_corpus"))
        metrics[arm].update(numeric_leaves(output_repeat_stats(cases_by_arm[arm], vocab), "output_text_backchannel"))
        adaptation = sources[arm] / "acoustic_adaptation.json"
        if adaptation.exists():
            metrics[arm].update(numeric_leaves(json.loads(adaptation.read_text(encoding="utf-8")), "acoustics"))
    names = sorted(set().union(*(m.keys() for m in metrics.values())))
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", {"id": ident, "trials_per_model": len(keys[0]), "metrics": metrics,
                                      "summaries": {a: str(p / "summary.json") for a, p in sources.items()}})
    with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", *ARMS])
        writer.writerows([name, *(metrics[a].get(name, "") for a in ARMS)] for name in names)
    lines = [f"# {ident}", "", f"同一の {len(keys[0])} 評価試行を3モデルで比較。", "",
             "training_corpus は教師データの統計、benchmark / acoustics はモデル出力の評価です。",
             "output_text_backchannel は内部書き起こしの語彙一致・同語反復です。独立ASRによる音声との整合性検査ではありません。",
             "欠けている指標は空欄です。自動評価には意味的な適切さのLLM判定は含みません。", "",
             "| 指標 | real-v2 | 従来の重畳 | AI配置 |", "|---|---:|---:|---:|"]
    for name in names:
        values = [format(metrics[a][name], ".5g") if name in metrics[a] else "" for a in ARMS]
        lines.append("| " + " | ".join([name, *values]) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[comparison] {out / 'results.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    build_report(args.root, args.id, Path(__file__).resolve().parents[2], args.allow_pending)
