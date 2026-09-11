#!/usr/bin/env python3
"""Print a diversity report for generated dialogue JSONL files."""

import argparse
import collections
import json
import os
import statistics
import sys


BACKCHANNELS = [
    "\u3046\u3093\u3046\u3093",
    "\u305d\u3046\u306a\u3093\u3067\u3059\u306d",
    "\u306a\u308b\u307b\u3069",
    "\u3078\u3048",
    "\u305d\u308c\u306f",
    "\u3046\u3093\u3001\u3046\u3093",
    "\u305d\u3063\u304b",
    "\u3048\u3048",
    "\u3075\u3080",
    "\u305d\u3046\u3067\u3059\u304b",
    "\u3046\u3093",
    "\u306f\u3044",
]


def normalize_text(text):
    return " ".join(str(text).lower().split())


def iter_dialogues(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: skipping malformed JSON on line {line_no}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(item, dict) or not isinstance(item.get("turns"), list):
                print(
                    f"WARNING: skipping malformed dialogue on line {line_no}: "
                    "expected object with turns list",
                    file=sys.stderr,
                )
                continue
            yield item


def non_silence_turns(dialogue):
    for turn in dialogue.get("turns", []):
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker")
        if speaker == "silence":
            continue
        yield speaker, str(turn.get("text", ""))


def normalized_transcript(dialogue):
    lines = [
        f"{speaker}: {normalize_text(text)}"
        for speaker, text in non_silence_turns(dialogue)
    ]
    return normalize_text("\n".join(lines))


def first_user_text(dialogue):
    for turn in dialogue.get("turns", []):
        if isinstance(turn, dict) and turn.get("speaker") == "user":
            return normalize_text(turn.get("text", ""))
    return ""


def char_ngrams(text, n):
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def distinct_score(texts, n):
    total = 0
    unique = set()
    for text in texts:
        grams = char_ngrams(text, n)
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return 0.0, 0, 0
    return len(unique) / total, len(unique), total


def jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def histogram(values):
    return collections.Counter(values)


def format_count_table(counter, empty_label="(empty)"):
    if not counter:
        return "  (none)"
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        label = str(key) if str(key) else empty_label
        rows.append(f"  {label}: {count}")
    return "\n".join(rows)


def sample_indices(total, samples):
    if total <= 0 or samples <= 0:
        return []
    count = min(total, samples)
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def build_report(dialogues, samples):
    lines = []
    turn_counts = [len(d.get("turns", [])) for d in dialogues]
    total_turns = sum(turn_counts)
    avg_turns = total_turns / len(dialogues)

    lines.append("Dialogue Diversity Report")
    lines.append("=========================")
    lines.append("")

    lines.append("1. Basic counts")
    lines.append(f"Dialogues: {len(dialogues)}")
    lines.append(f"Total turns: {total_turns}")
    lines.append(f"Avg turns per dialogue: {avg_turns:.2f}")
    lines.append(f"Min turns per dialogue: {min(turn_counts)}")
    lines.append(f"Max turns per dialogue: {max(turn_counts)}")
    lines.append("Turn-count histogram:")
    for turns, count in sorted(histogram(turn_counts).items()):
        lines.append(f"  {turns}: {count}")
    lines.append("")

    transcripts = [normalized_transcript(d) for d in dialogues]
    transcript_groups = collections.defaultdict(list)
    for idx, transcript in enumerate(transcripts):
        transcript_groups[transcript].append(idx)
    duplicate_dialogues = sum(len(items) for items in transcript_groups.values() if len(items) > 1)
    lines.append("2. Exact duplicates")
    lines.append(f"Dialogues in duplicate groups: {duplicate_dialogues}")
    lines.append(f"Unique transcripts: {len(transcript_groups)}")
    lines.append("")

    moshi_texts = []
    user_texts = []
    combined_texts = []
    for dialogue in dialogues:
        for speaker, text in non_silence_turns(dialogue):
            if speaker == "moshi":
                moshi_texts.append(text)
            elif speaker == "user":
                user_texts.append(text)
            combined_texts.append(text)

    lines.append("3. Lexical diversity")
    for label, texts in (
        ("moshi", moshi_texts),
        ("user", user_texts),
        ("combined", combined_texts),
    ):
        d1, u1, t1 = distinct_score(texts, 1)
        d2, u2, t2 = distinct_score(texts, 2)
        lines.append(
            f"{label}: distinct-1={d1:.4f} ({u1}/{t1}), "
            f"distinct-2={d2:.4f} ({u2}/{t2})"
        )
    lines.append("")

    openings = [first_user_text(d) for d in dialogues]
    opening_counts = collections.Counter(openings)
    opening_ratio = len(opening_counts) / len(openings) if openings else 0.0
    lines.append("4. Opening diversity")
    lines.append(f"Unique openings / total: {len(opening_counts)}/{len(openings)} ({opening_ratio:.4f})")
    lines.append("Top 5 repeated openings:")
    repeated = [(text, count) for text, count in opening_counts.most_common() if count > 1]
    if repeated:
        for text, count in repeated[:5]:
            label = text if text else "(empty)"
            lines.append(f"  {count}: {label}")
    else:
        lines.append("  (none repeated)")
    lines.append("")

    capped = dialogues[:300]
    capped_transcripts = transcripts[:300]
    gram_sets = [set(char_ngrams(text, 3)) for text in capped_transcripts]
    pair_scores = []
    high_pairs = []
    for i in range(len(gram_sets)):
        for j in range(i + 1, len(gram_sets)):
            score = jaccard(gram_sets[i], gram_sets[j])
            pair_scores.append(score)
            if score > 0.7:
                left_id = capped[i].get("id", i)
                right_id = capped[j].get("id", j)
                high_pairs.append((score, left_id, right_id))
    mean_similarity = statistics.fmean(pair_scores) if pair_scores else 0.0
    lines.append("5. Near-duplicate similarity")
    lines.append(f"Dialogues compared: {len(capped)}")
    lines.append(f"Mean pairwise char-3-gram Jaccard: {mean_similarity:.4f}")
    lines.append(f"Pairs with Jaccard > 0.7: {len(high_pairs)}")
    if high_pairs:
        lines.append("Up to 5 high-similarity pairs:")
        for score, left_id, right_id in sorted(high_pairs, reverse=True)[:5]:
            lines.append(f"  {left_id} <-> {right_id}: {score:.4f}")
    else:
        lines.append("Up to 5 high-similarity pairs: (none)")
    lines.append("")

    lines.append("6. Distribution")
    for field in ("category", "duplex_task", "risk_level"):
        lines.append(f"{field}:")
        lines.append(format_count_table(collections.Counter(d.get(field, "") for d in dialogues)))
    lines.append("")

    backchannel_counts = collections.Counter()
    moshi_turn_count = 0
    only_backchannel_count = 0
    backchannel_set = {normalize_text(token) for token in BACKCHANNELS}
    for dialogue in dialogues:
        for turn in dialogue.get("turns", []):
            if not isinstance(turn, dict) or turn.get("speaker") != "moshi":
                continue
            moshi_turn_count += 1
            text = str(turn.get("text", ""))
            normalized = normalize_text(text)
            if normalized in backchannel_set:
                only_backchannel_count += 1
            for token in BACKCHANNELS:
                backchannel_counts[token] += text.count(token)
    only_share = only_backchannel_count / moshi_turn_count if moshi_turn_count else 0.0
    lines.append("7. Backchannel variety")
    lines.append("Backchannel occurrences in moshi turns:")
    for token in BACKCHANNELS:
        lines.append(f"  {token}: {backchannel_counts[token]}")
    lines.append(
        f"Moshi turns only a backchannel: {only_backchannel_count}/{moshi_turn_count} "
        f"({only_share:.4f})"
    )
    lines.append("")

    all_text = "".join(combined_texts)
    lines.append("8. Vocabulary")
    lines.append(f"Unique-char count: {len(set(all_text))}")
    lines.append(f"Unique 2-char-token count: {len(set(char_ngrams(all_text, 2)))}")
    lines.append("")

    lines.append("9. Sample dialogues")
    for idx in sample_indices(len(dialogues), samples):
        dialogue = dialogues[idx]
        lines.append("")
        lines.append(
            f"Sample index {idx}: id={dialogue.get('id', '')} "
            f"category={dialogue.get('category', '')} title={dialogue.get('title', '')}"
        )
        for turn in dialogue.get("turns", []):
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "")
            text = str(turn.get("text", ""))
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Input dialogues.jsonl path")
    parser.add_argument("--out", dest="output_path", help="Optional output report path")
    parser.add_argument("--samples", type=int, default=5, help="Number of full sample dialogues")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_path):
        print(f"ERROR: input file does not exist: {args.input_path}", file=sys.stderr)
        return 1

    dialogues = list(iter_dialogues(args.input_path))
    if not dialogues:
        print(f"ERROR: input contains zero valid dialogues: {args.input_path}", file=sys.stderr)
        return 1

    report = build_report(dialogues, args.samples)
    print(report, end="")
    if args.output_path:
        output_dir = os.path.dirname(os.path.abspath(args.output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            handle.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
