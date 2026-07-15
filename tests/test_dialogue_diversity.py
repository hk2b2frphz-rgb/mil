import itertools

import numpy as np

from scripts.report_dialogue_diversity import (
    char_ngrams,
    exact_nearest_jaccard,
    jaccard,
    write_nearest_jsonl,
)


def brute_force(transcripts):
    gram_sets = [set(char_ngrams(text, 3)) for text in transcripts]
    scores = np.zeros((len(transcripts), len(transcripts)), dtype=float)
    for left, right in itertools.combinations(range(len(transcripts)), 2):
        score = jaccard(gram_sets[left], gram_sets[right])
        scores[left, right] = score
        scores[right, left] = score
    nearest = []
    for index in range(len(transcripts)):
        candidates = np.delete(scores[index], index)
        nearest.append(float(np.max(candidates)) if candidates.size else 0.0)
    return scores, np.asarray(nearest)


def test_exact_nearest_jaccard_matches_brute_force():
    transcripts = [
        "user: 今日は晴れです moshi: そうですね",
        "user: 今日は晴れですね moshi: そうですね",
        "user: 明日は雨です moshi: 傘を持ちましょう",
        "",
    ]
    pair_scores, nearest = brute_force(transcripts)

    result = exact_nearest_jaccard(transcripts, block_size=2)

    upper = pair_scores[np.triu_indices(len(transcripts), 1)]
    assert result["pairs"] == 6
    assert result["mean_pairwise"] == np.mean(upper)
    np.testing.assert_allclose(result["nearest_scores"], nearest)
    assert result["nearest_max"] == np.max(nearest)
    for threshold in (0.5, 0.7, 0.8):
        assert result["pair_threshold_counts"][threshold] == int(np.count_nonzero(upper > threshold))
        assert result["nearest_threshold_counts"][threshold] == int(
            np.count_nonzero(nearest > threshold)
        )


def test_empty_transcripts_follow_existing_jaccard_semantics():
    result = exact_nearest_jaccard(["", ""], block_size=1)

    assert result["mean_pairwise"] == 1.0
    np.testing.assert_array_equal(result["nearest_scores"], [1.0, 1.0])
    assert result["nearest_threshold_counts"][0.8] == 2


def test_single_transcript_has_no_nearest_neighbor():
    result = exact_nearest_jaccard(["短い対話"], block_size=1)

    assert result["pairs"] == 0
    assert result["mean_pairwise"] == 0.0
    assert result["nearest_indices"].tolist() == [-1]
    assert result["nearest_scores"].tolist() == [0.0]


def test_write_nearest_jsonl_records_every_dialogue(tmp_path):
    import json

    dialogues = [{"id": "left"}, {"id": "right"}]
    similarity = exact_nearest_jaccard(["same dialogue", "same dialogue"], block_size=1)
    output = tmp_path / "neighbors.jsonl"

    write_nearest_jsonl(output, dialogues, similarity)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "dialogue_index": 0,
            "dialogue_id": "left",
            "nearest_dialogue_index": 1,
            "nearest_dialogue_id": "right",
            "char_3gram_jaccard": 1.0,
        },
        {
            "dialogue_index": 1,
            "dialogue_id": "right",
            "nearest_dialogue_index": 0,
            "nearest_dialogue_id": "left",
            "char_3gram_jaccard": 1.0,
        },
    ]
