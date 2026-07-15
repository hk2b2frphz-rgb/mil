import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarify.detector import classify_turn, first_turn_after, segment_turns  # noqa: E402


def test_targeted_clarification_time():
    lab = classify_turn("すみません、何時でしたか？", "time", "15時")
    assert lab.label == "clarify_targeted"
    assert lab.is_clarification


def test_targeted_clarification_person():
    lab = classify_turn("すみません、お名前が聞き取れませんでした。どなたでしたか？",
                        "person", "田中さん")
    assert lab.label == "clarify_targeted"


def test_generic_clarification():
    lab = classify_turn("すみません、もう一度お願いします。", "time", "15時")
    assert lab.label == "clarify_generic"
    assert lab.is_clarification


def test_confirmation_with_value():
    lab = classify_turn("15時ですね。カレンダーに登録しておきますね。",
                        "time", "15時")
    assert lab.label == "confirm"
    assert not lab.is_clarification
    assert lab.slot_value_echoed


def test_confirmation_time_format_variants():
    lab = classify_turn("午後3時ですね。承知しました。", "time", "15時")
    assert lab.label == "confirm"
    assert lab.slot_value_echoed


def test_echo_question_is_clarification():
    lab = classify_turn("えっと、15時、ですか？", "time", "15時")
    assert lab.is_clarification
    assert lab.label == "clarify_targeted"


def test_chitchat_is_other():
    lab = classify_turn("今日はいい天気ですね。", "time", "15時")
    assert lab.label == "other"


def test_wrong_value_confirmation_not_echoed():
    lab = classify_turn("5時ですね。設定しますね。", "time", "15時")
    assert lab.label == "confirm"
    assert not lab.slot_value_echoed


def _ev(t, piece):
    return {"step": int(t * 12.5), "time_sec": t, "piece": piece}


def test_segment_turns_splits_on_gap():
    events = [_ev(1.0, "はい"), _ev(1.2, "。"),
              _ev(4.0, "何時"), _ev(4.2, "ですか"), _ev(4.4, "？")]
    turns = segment_turns(events, gap_sec=1.2)
    assert len(turns) == 2
    assert turns[0].text == "はい。"
    assert turns[1].text == "何時ですか？"


def test_first_turn_after_allows_overlap():
    events = [_ev(2.8, "何時"), _ev(3.0, "ですか"), _ev(3.2, "？")]
    turns = segment_turns(events)
    # Utterance ends at 3.0; model started at 2.8 (overlap) - still counts.
    turn = first_turn_after(turns, after_sec=3.0)
    assert turn is not None
    assert turn.text == "何時ですか？"


def test_first_turn_after_skips_early_backchannel():
    events = [_ev(0.5, "はい"), _ev(0.7, "。"),
              _ev(5.0, "何時"), _ev(5.2, "ですか？")]
    turns = segment_turns(events)
    turn = first_turn_after(turns, after_sec=4.0)
    assert turn is not None
    assert "何時" in turn.text
