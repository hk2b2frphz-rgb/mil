import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random  # noqa: E402

from clarify import slots  # noqa: E402
from clarify.metrics import (  # noqa: E402
    aggregate_by_condition, decision_quality, paired_bootstrap_delta,
    score_trial, wilson_interval,
)
from clarify.scenario import BaseItem, BenchmarkCase, case_seed  # noqa: E402
from clarify.train_data import generate_training_dialogues  # noqa: E402


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------

def test_parse_annot_utt():
    parsed = slots.parse_annot_utt("明日の[time : 朝7時]に[event_name : 会議]を入れて")
    assert parsed == [("time", "朝7時"), ("event_name", "会議")]


def test_split_around_slot():
    result = slots.split_around_slot("明日の[time : 朝7時]にアラームをかけて", "time")
    assert result is not None
    assert result.pre_text == "明日の"
    assert result.surface == "朝7時"
    assert result.post_text == "にアラームをかけて"


def test_split_rejects_duplicate_slot():
    annot = "[time : 3時]か[time : 5時]に起こして"
    assert slots.split_around_slot(annot, "time") is None


def test_normalize_time_variants():
    assert slots.normalize_slot_value("午後3時") == slots.normalize_slot_value("15時")
    assert slots.slot_value_in_text("15時", "はい、午後3時ですね。")
    assert not slots.slot_value_in_text("15時", "はい、5時ですね。")


def test_kanji_numbers():
    assert slots.slot_value_in_text("7時", "七時ですね")


def test_select_base_items_balances_and_validates():
    rows = [
        {"id": i, "intent": "alarm_set",
         "annot_utt": f"明日の[time : {h}時]にアラームをかけて",
         "utt": f"明日の{h}時にアラームをかけて"}
        for i, h in enumerate([6, 7, 8, 9])
    ] + [
        {"id": 100, "intent": "play_music",
         "annot_utt": "[artist_name : 米津玄師]の曲をかけて",
         "utt": "米津玄師の曲をかけて"},
        # broken: annot/utt mismatch -> must be skipped
        {"id": 101, "intent": "alarm_set",
         "annot_utt": "明日の[time : 7時]にアラーム",
         "utt": "違うテキスト"},
    ]
    items = slots.select_base_items(rows, max_items=10)
    ids = {i.base_id for i in items}
    assert "massive_101" not in ids
    assert any(i.slot_type == "artist_name" for i in items)
    for item in items:
        item.validate()


def test_underspecified_items_validate():
    items = slots.build_underspecified_items()
    assert len(items) >= 5
    for item in items:
        assert item.arm == "underspecified"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def make_case(condition="mask_silence", expected="ask") -> dict:
    item = BaseItem(
        base_id="massive_1", arm="acoustic", intent="alarm_set",
        slot_type="time", slot_value="15時", utterance_text="明日の15時にアラームをかけて",
        pre_text="明日の", slot_text="15時", post_text="にアラームをかけて",
        repair_text="15時です。",
    )
    case = BenchmarkCase(
        case_id=f"massive_1__{condition}", base=item, condition=condition,
        expected_behavior=expected, audio_path="a.wav",
        clean_audio_path="c.wav", repair_audio_path="r.wav",
        span_start_sec=0.5, span_end_sec=1.0, sample_rate=24000,
        seed_base=case_seed("massive_1", condition),
    )
    return case.to_json()


def _ev(t, piece):
    return {"step": int(t * 12.5), "time_sec": t, "piece": piece}


def test_score_trial_ask_then_recover():
    events = [_ev(3.5, "何時"), _ev(3.7, "ですか？"),
              _ev(9.0, "15時ですね。"), _ev(9.3, "設定しますね。")]
    policy = {"repair_injected": True, "repair_end_step": int(7.0 * 12.5),
              "frame_rate": 12.5, "turns": []}
    score = score_trial(make_case(), 0, events, policy, utterance_end_sec=3.0)
    assert score.asked and score.targeted
    assert score.slot_correct
    assert not score.hallucinated_confirmation


def test_score_trial_hallucinated_confirmation():
    events = [_ev(3.5, "5時ですね。"), _ev(3.8, "設定しますね。")]
    policy = {"repair_injected": False, "turns": []}
    score = score_trial(make_case(), 0, events, policy, utterance_end_sec=3.0)
    assert not score.asked
    assert score.hallucinated_confirmation
    assert not score.slot_correct


def test_score_trial_no_response():
    score = score_trial(make_case(), 0, [], {"repair_injected": False},
                        utterance_end_sec=3.0)
    assert score.no_response
    assert score.first_label == "no_response"


def test_aggregate_and_decision_quality():
    scores = []
    for cond, expected, ask in [("clean", "act", False),
                                ("clean", "act", False),
                                ("mask_silence", "ask", True),
                                ("mask_silence", "ask", False)]:
        events = ([_ev(3.5, "何時ですか？")] if ask
                  else [_ev(3.5, "15時ですね。設定しますね。")])
        policy = {"repair_injected": ask, "turns": []}
        s = score_trial(make_case(cond, expected), 0, events, policy, 3.0)
        scores.append(s)
    agg = aggregate_by_condition(scores)
    assert agg["mask_silence"]["CRR"]["rate"] == 0.5
    assert agg["clean"]["CRR"]["rate"] == 0.0
    dq = decision_quality(scores)
    assert dq["hit_rate"]["rate"] == 0.5
    assert dq["false_alarm_rate"]["rate"] == 0.0
    assert dq["balanced_accuracy"] == 0.75


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_paired_bootstrap_detects_difference():
    def mk(ask: bool, base_id: str):
        events = ([_ev(3.5, "何時ですか？")] if ask
                  else [_ev(3.5, "こんにちは。")])
        case = make_case()
        case["base"]["base_id"] = base_id
        return score_trial(case, 0, events,
                           {"repair_injected": False, "turns": []}, 3.0)

    a = [mk(False, f"b{i}") for i in range(20)]
    b = [mk(True, f"b{i}") for i in range(20)]
    delta = paired_bootstrap_delta(a, b, lambda s: s.asked, n_boot=500)
    assert delta["delta"] == 1.0
    assert delta["p_sign_flip"] < 0.05


# ---------------------------------------------------------------------------
# training data
# ---------------------------------------------------------------------------

def test_training_variants():
    rows = [
        {"id": i, "intent": "alarm_set",
         "annot_utt": f"明日の[time : {h}時]にアラームをかけて",
         "utt": f"明日の{h}時にアラームをかけて"}
        for i, h in enumerate([6, 7, 8, 9, 10, 11])
    ]
    items = slots.select_base_items(rows, max_items=6)
    under = slots.build_underspecified_items()
    rng = random.Random(0)

    task = generate_training_dialogues(items, under, "task_only", rng)
    assert all(d.behavior == "confirm" for d in task)
    assert all(d.corruption is None for d in task)

    rng = random.Random(0)
    lex = generate_training_dialogues(items, under, "clarify_lexical", rng)
    assert any(d.behavior == "ask_lexical" for d in lex)
    assert all(d.behavior != "ask_acoustic" for d in lex)
    assert all(d.corruption is None for d in lex)

    rng = random.Random(0)
    full = generate_training_dialogues(items, under, "clarify_full", rng,
                                       ask_ratio=0.9)
    asks = [d for d in full if d.behavior == "ask_acoustic"]
    assert asks, "clarify_full must contain acoustic ask dialogues"
    for d in asks:
        assert d.corruption is not None
        assert d.corruption["target_text"]
        assert d.pair_id is not None
        # minimal pair partner exists and is clean
        partners = [x for x in full
                    if x.pair_id == d.pair_id and x.behavior == "confirm"]
        assert len(partners) == 1
        assert partners[0].corruption is None
    # ask dialogues follow user -> moshi(ask) -> user(repair) -> moshi(confirm)
    d = asks[0]
    speakers = [t["speaker"] for t in d.turns]
    assert speakers[-1] == "moshi" and "user" in speakers[-3:-1]
