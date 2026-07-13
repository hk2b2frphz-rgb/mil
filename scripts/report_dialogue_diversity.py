#!/usr/bin/env python3
"""Print a diversity report for generated dialogue JSONL files."""

import argparse
import collections
import heapq
import json
import os
import random
import sys

import numpy as np
from scipy import sparse


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


def build_ngram_matrix(transcripts, n=3):
    """Build a binary document-by-ngram matrix without retaining Python sets."""
    vocabulary = {}
    row_indices = []
    column_indices = []
    sizes = np.zeros(len(transcripts), dtype=np.int32)
    for row, text in enumerate(transcripts):
        grams = set(char_ngrams(text, n))
        sizes[row] = len(grams)
        for gram in grams:
            column = vocabulary.setdefault(gram, len(vocabulary))
            row_indices.append(row)
            column_indices.append(column)
    values = np.ones(len(row_indices), dtype=np.int32)
    matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(transcripts), len(vocabulary)),
        dtype=np.int32,
    )
    return matrix, sizes


def exact_nearest_jaccard(transcripts, block_size=128, thresholds=(0.5, 0.7, 0.8)):
    """Compute exact all-pairs Jaccard statistics with bounded working memory.

    Sparse matrix multiplication computes n-gram intersections in compiled code.
    Only ``block_size x number_of_dialogues`` scores are materialized at once, so
    a 10k-dialogue report does not retain the full 10k x 10k matrix.
    """
    total = len(transcripts)
    if total == 0:
        raise ValueError("at least one transcript is required")

    matrix, sizes = build_ngram_matrix(transcripts, n=3)
    nearest_scores = np.zeros(total, dtype=np.float64)
    nearest_indices = np.full(total, -1, dtype=np.int64)
    pair_sum = 0.0
    pair_count = total * (total - 1) // 2
    pair_threshold_counts = {threshold: 0 for threshold in thresholds}
    top_pairs = []

    for start in range(0, total, block_size):
        stop = min(total, start + block_size)
        intersections = (matrix[start:stop] @ matrix.T).toarray()
        unions = sizes[start:stop, None] + sizes[None, :] - intersections
        scores = np.zeros(intersections.shape, dtype=np.float64)
        np.divide(intersections, unions, out=scores, where=unions != 0)
        scores[unions == 0] = 1.0

        for local_row, global_row in enumerate(range(start, stop)):
            scores[local_row, global_row] = -1.0
            if total > 1:
                nearest_index = int(np.argmax(scores[local_row]))
                nearest_indices[global_row] = nearest_index
                nearest_scores[global_row] = scores[local_row, nearest_index]

            upper_scores = scores[local_row, global_row + 1 :]
            if upper_scores.size == 0:
                continue
            pair_sum += float(np.sum(upper_scores))
            for threshold in thresholds:
                pair_threshold_counts[threshold] += int(np.count_nonzero(upper_scores > threshold))
            candidate_count = min(5, upper_scores.size)
            candidate_offsets = np.argpartition(upper_scores, -candidate_count)[-candidate_count:]
            for offset in candidate_offsets:
                right = global_row + 1 + int(offset)
                item = (float(upper_scores[offset]), global_row, right)
                if len(top_pairs) < 5:
                    heapq.heappush(top_pairs, item)
                elif item > top_pairs[0]:
                    heapq.heapreplace(top_pairs, item)

    valid_nearest = nearest_scores if total > 1 else np.array([], dtype=np.float64)
    nearest_threshold_counts = {
        threshold: int(np.count_nonzero(valid_nearest > threshold))
        for threshold in thresholds
    }
    percentiles = {
        percentile: float(np.percentile(valid_nearest, percentile)) if valid_nearest.size else 0.0
        for percentile in (50, 90, 95, 99)
    }
    return {
        "dialogues": total,
        "pairs": pair_count,
        "mean_pairwise": pair_sum / pair_count if pair_count else 0.0,
        "pair_threshold_counts": pair_threshold_counts,
        "nearest_scores": nearest_scores,
        "nearest_indices": nearest_indices,
        "nearest_mean": float(np.mean(valid_nearest)) if valid_nearest.size else 0.0,
        "nearest_percentiles": percentiles,
        "nearest_max": float(np.max(valid_nearest)) if valid_nearest.size else 0.0,
        "nearest_threshold_counts": nearest_threshold_counts,
        "top_pairs": sorted(top_pairs, reverse=True),
    }


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


def build_report(
    dialogues,
    samples,
    similarity_block_size=128,
    scale_points=(),
    similarity_seed=0,
    similarity=None,
):
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

    if similarity is None:
        similarity = exact_nearest_jaccard(transcripts, block_size=similarity_block_size)
    lines.append("5. Near-duplicate similarity")
    lines.append(f"Dialogues compared: {similarity['dialogues']} (all dialogues)")
    lines.append(f"Pairs compared: {similarity['pairs']} (exact all-pairs)")
    lines.append(f"Mean pairwise char-3-gram Jaccard: {similarity['mean_pairwise']:.4f}")
    lines.append(f"Mean nearest-neighbor Jaccard: {similarity['nearest_mean']:.4f}")
    for percentile, value in similarity["nearest_percentiles"].items():
        lines.append(f"Nearest-neighbor Jaccard p{percentile}: {value:.4f}")
    lines.append(f"Maximum nearest-neighbor Jaccard: {similarity['nearest_max']:.4f}")
    for threshold, count in similarity["nearest_threshold_counts"].items():
        share = count / len(dialogues)
        pair_count = similarity["pair_threshold_counts"][threshold]
        lines.append(
            f"Dialogues with nearest Jaccard > {threshold:.1f}: "
            f"{count}/{len(dialogues)} ({share:.4f}); pairs above threshold: {pair_count}"
        )
    lines.append("Top 5 nearest pairs:")
    if similarity["top_pairs"]:
        for score, left, right in similarity["top_pairs"]:
            left_id = dialogues[left].get("id", left)
            right_id = dialogues[right].get("id", right)
            lines.append(f"  {left_id} <-> {right_id}: {score:.4f}")
    else:
        lines.append("  (none)")
    lines.append("")

    checkpoints = sorted({point for point in scale_points if 1 < point < len(dialogues)})
    checkpoints.append(len(dialogues))
    shuffled_indices = list(range(len(dialogues)))
    random.Random(similarity_seed).shuffle(shuffled_indices)
    lines.append("6. Diversity scaling (deterministic nested samples)")
    lines.append(f"Sample seed: {similarity_seed}")
    lines.append(
        "N | unique transcript ratio | unique opening ratio | "
        "mean nearest | p95 nearest | nearest > 0.7"
    )
    for point in checkpoints:
        if point == len(dialogues):
            selected_dialogues = dialogues
            selected_transcripts = transcripts
            point_similarity = similarity
        else:
            selected = shuffled_indices[:point]
            selected_dialogues = [dialogues[index] for index in selected]
            selected_transcripts = [transcripts[index] for index in selected]
            point_similarity = exact_nearest_jaccard(
                selected_transcripts,
                block_size=similarity_block_size,
            )
        unique_transcripts = len(set(selected_transcripts)) / point
        unique_openings = len({first_user_text(item) for item in selected_dialogues}) / point
        above = point_similarity["nearest_threshold_counts"][0.7] / point
        lines.append(
            f"{point} | {unique_transcripts:.4f} | {unique_openings:.4f} | "
            f"{point_similarity['nearest_mean']:.4f} | "
            f"{point_similarity['nearest_percentiles'][95]:.4f} | {above:.4f}"
        )
    lines.append("")

    lines.append("7. Distribution")
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
    lines.append("8. Backchannel variety")
    lines.append("Backchannel occurrences in moshi turns:")
    for token in BACKCHANNELS:
        lines.append(f"  {token}: {backchannel_counts[token]}")
    lines.append(
        f"Moshi turns only a backchannel: {only_backchannel_count}/{moshi_turn_count} "
        f"({only_share:.4f})"
    )
    lines.append("")

    all_text = "".join(combined_texts)
    lines.append("9. Vocabulary")
    lines.append(f"Unique-char count: {len(set(all_text))}")
    lines.append(f"Unique 2-char-token count: {len(set(char_ngrams(all_text, 2)))}")
    lines.append("")

    lines.append("10. Sample dialogues")
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


def write_nearest_jsonl(path, dialogues, similarity):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, dialogue in enumerate(dialogues):
            nearest_index = int(similarity["nearest_indices"][index])
            row = {
                "dialogue_index": index,
                "dialogue_id": dialogue.get("id", index),
                "nearest_dialogue_index": nearest_index if nearest_index >= 0 else None,
                "nearest_dialogue_id": (
                    dialogues[nearest_index].get("id", nearest_index)
                    if nearest_index >= 0
                    else None
                ),
                "char_3gram_jaccard": float(similarity["nearest_scores"][index]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Input dialogues.jsonl path")
    parser.add_argument("--out", dest="output_path", help="Optional output report path")
    parser.add_argument(
        "--nearest-jsonl",
        help="Optional JSONL output containing every dialogue's exact nearest neighbor",
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of full sample dialogues")
    parser.add_argument(
        "--similarity-block-size",
        type=int,
        default=128,
        help="Rows per exact Jaccard work block (controls temporary memory, not sampling)",
    )
    parser.add_argument(
        "--scale-points",
        default="100,300,1000,3000,10000",
        help="Comma-separated nested sample sizes for the diversity scaling table",
    )
    parser.add_argument(
        "--similarity-seed",
        type=int,
        default=0,
        help="Seed for deterministic nested scaling samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.similarity_block_size < 1:
        print("ERROR: --similarity-block-size must be positive", file=sys.stderr)
        return 2
    try:
        scale_points = tuple(int(value) for value in args.scale_points.split(",") if value.strip())
    except ValueError:
        print("ERROR: --scale-points must be comma-separated integers", file=sys.stderr)
        return 2
    if any(point < 2 for point in scale_points):
        print("ERROR: --scale-points values must be at least 2", file=sys.stderr)
        return 2
    if not os.path.exists(args.input_path):
        print(f"ERROR: input file does not exist: {args.input_path}", file=sys.stderr)
        return 1

    dialogues = list(iter_dialogues(args.input_path))
    if not dialogues:
        print(f"ERROR: input contains zero valid dialogues: {args.input_path}", file=sys.stderr)
        return 1

    transcripts = [normalized_transcript(dialogue) for dialogue in dialogues]
    similarity = exact_nearest_jaccard(
        transcripts,
        block_size=args.similarity_block_size,
    )
    report = build_report(
        dialogues,
        args.samples,
        similarity_block_size=args.similarity_block_size,
        scale_points=scale_points,
        similarity_seed=args.similarity_seed,
        similarity=similarity,
    )
    print(report, end="")
    if args.output_path:
        output_dir = os.path.dirname(os.path.abspath(args.output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            handle.write(report)
    if args.nearest_jsonl:
        write_nearest_jsonl(args.nearest_jsonl, dialogues, similarity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
