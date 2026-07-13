import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarify import corpora, slots  # noqa: E402
from clarify.detector import classify_turn  # noqa: E402
from clarify.judge_pack import pack_trial  # noqa: E402
from clarify.lang import en_normalize, en_value_in_text, get_pack  # noqa: E402
from clarify.scenario import BaseItem, BenchmarkCase, case_seed  # noqa: E402
from clarify.train_data import generate_training_dialogues  # noqa: E402


# ---------------------------------------------------------------------------
# normalization / matching
# ---------------------------------------------------------------------------

def test_en_time_normalization_variants():
    assert en_normalize("3 pm") == en_normalize("3pm")
    assert en_normalize("three pm") == en_normalize("3 PM")
    assert "15:00" in en_normalize("3 p.m.")
    assert "07:00" in en_normalize("seven o'clock")
    assert "00:30" in en_normalize("12:30 am")


def test_en_value_in_text_matches_time_formats():
    assert en_value_in_text("5 pm", "Sure, 5 pm, setting the alarm.")
    assert en_value_in_text("5 pm", "Got it, five pm it is.")
    assert not en_value_in_text("5 pm", "Got it, five am it is.")


def test_en_value_word_boundaries():
    assert not en_value_in_text("ten", "there is tension in the room")
    assert en_value_in_text("ten", "I said ten, right?")


def test_en_value_multiword():
    assert en_value_in_text("taylor swift", "Playing Taylor Swift now.")
    assert not en_value_in_text("taylor swift", "Playing Taylor now.")


# ---------------------------------------------------------------------------
# detection (English)
# ---------------------------------------------------------------------------

def test_en_targeted_clarification():
    lab = classify_turn("Sorry, what time was it?", "time", "5 pm", lang="en")
    assert lab.label == "clarify_targeted"
    assert lab.is_clarification


def test_en_generic_clarification():
    lab = classify_turn("Sorry, could you say that again?", "time", "5 pm",
                        lang="en")
    assert lab.label == "clarify_generic"


def test_en_didnt_catch_is_clarification():
    lab = classify_turn("I didn't catch that.", "person", "Sarah", lang="en")
    assert lab.is_clarification


def test_en_confirmation_with_value():
    lab = classify_turn("5 pm, got it. I'll set the alarm.", "time", "5 pm",
                        lang="en")
    assert lab.label == "confirm"
    assert lab.slot_value_echoed
    assert not lab.is_clarification


def test_en_echo_question_is_targeted():
    lab = classify_turn("5 pm, was it?", "time", "5 pm", lang="en")
    assert lab.label == "clarify_targeted"


def test_en_chitchat_is_other():
    lab = classify_turn("Nice weather today.", "time", "5 pm", lang="en")
    assert lab.label == "other"


def test_en_wrong_value_confirm_not_echoed():
    lab = classify_turn("7 pm, got it, setting the alarm.", "time", "5 pm",
                        lang="en")
    assert lab.label == "confirm"
    assert not lab.slot_value_echoed


# ---------------------------------------------------------------------------
# selection (English word-length bounds)
# ---------------------------------------------------------------------------

EN_ROWS = [
    {"id": i, "intent": "alarm_set",
     "annot_utt": f"wake me up at [time : {h} am] tomorrow",
     "utt": f"wake me up at {h} am tomorrow"}
    for i, h in enumerate([6, 7, 8, 9])
] + [
    {"id": 100, "intent": "play_music",
     "annot_utt": "play songs by [artist_name : taylor swift] please",
     "utt": "play songs by taylor swift please"},
]


def test_en_select_base_items():
    items = slots.select_base_items(EN_ROWS, language="en", max_items=10)
    assert items
    for item in items:
        assert item.language == "en"
        assert item.audio_source == "tts"
        assert item.repair_text.startswith("It's ")
        item.validate()
    assert any(i.slot_type == "artist_name" for i in items)


def test_en_underspecified_items():
    items = slots.build_underspecified_items("en")
    assert len(items) >= 5
    assert all(i.language == "en" for i in items)


# ---------------------------------------------------------------------------
# SLURP loader
# ---------------------------------------------------------------------------

def test_load_slurp_rows(tmp_path):
    audio_dir = tmp_path / "slurp_real"
    audio_dir.mkdir()
    (audio_dir / "audio-001-headset.flac").write_bytes(b"\x00")
    (audio_dir / "audio-001.flac").write_bytes(b"\x00")
    meta = tmp_path / "test.jsonl"
    rows = [
        {"slurp_id": 1, "scenario": "alarm", "action": "set",
         "sentence": "wake me up at seven am",
         "sentence_annotation": "wake me up at [time : seven am]",
         "recordings": [{"file": "audio-001.flac"},
                        {"file": "audio-001-headset.flac"}]},
        {"slurp_id": 2, "scenario": "play", "action": "music",
         "sentence": "no audio here",
         "sentence_annotation": "no audio here",
         "recordings": [{"file": "missing.flac"}]},
    ]
    meta.write_text("\n".join(json.dumps(r) for r in rows),
                    encoding="utf-8")

    loaded = corpora.load_slurp(meta, audio_dir)
    assert len(loaded) == 1  # row without audio is dropped
    row = loaded[0]
    assert row["intent"] == "alarm_set"
    assert row["source"] == "slurp"
    assert "headset" in row["audio_file"]  # close-talk preferred

    items = slots.select_base_items(loaded, language="en", max_items=5)
    assert len(items) == 1
    assert items[0].audio_source == "real"
    assert items[0].meta["audio_file"] == row["audio_file"]


# ---------------------------------------------------------------------------
# training data (English + ablations)
# ---------------------------------------------------------------------------

def _en_items(n=6):
    rows = [
        {"id": i, "intent": "alarm_set",
         "annot_utt": f"wake me up at [time : {h} am] tomorrow",
         "utt": f"wake me up at {h} am tomorrow"}
        for i, h in enumerate(range(1, n + 1))
    ]
    return slots.select_base_items(rows, language="en", max_items=n)


def test_en_training_dialogues_use_en_templates():
    items = _en_items()
    under = slots.build_underspecified_items("en")
    rng = random.Random(0)
    dialogues = generate_training_dialogues(items, under, "clarify_full",
                                            rng, ask_ratio=0.9)
    asks = [d for d in dialogues if d.behavior == "ask_acoustic"]
    assert asks
    for d in asks:
        assert d.language == "en"
        moshi_turns = [t["text"] for t in d.turns if t["speaker"] == "moshi"]
        assert any("Sorry" in t for t in moshi_turns)  # EN ask template
        assert any("got it" in t.lower() or "alarm" in t.lower()
                   for t in moshi_turns)


def test_ablation_no_minimal_pairs():
    items = _en_items()
    rng = random.Random(0)
    dialogues = generate_training_dialogues(
        items, [], "clarify_full", rng, ask_ratio=0.9, minimal_pairs=False
    )
    asks = [d for d in dialogues if d.behavior == "ask_acoustic"]
    assert asks
    assert all(d.pair_id is None for d in dialogues)
    # No clean twin: confirm count is only the non-ask remainder.
    confirms = [d for d in dialogues if d.behavior == "confirm"]
    assert len(confirms) == len(items) - len(asks)


def test_ablation_no_mild_noise():
    items = _en_items(20)
    rng = random.Random(1)
    dialogues = generate_training_dialogues(
        items, [], "clarify_full", rng, ask_ratio=0.0,
        mild_noise_confirm_ratio=0.0,
    )
    assert all(d.corruption is None for d in dialogues)


# ---------------------------------------------------------------------------
# judge pack language selection
# ---------------------------------------------------------------------------

def _make_case(lang: str) -> dict:
    item = BaseItem(
        base_id="x1", arm="acoustic", intent="alarm_set", slot_type="time",
        slot_value="5 pm", utterance_text="wake me up at 5 pm tomorrow",
        pre_text="wake me up at ", slot_text="5 pm", post_text=" tomorrow",
        repair_text="It's 5 pm.", language=lang,
    )
    case = BenchmarkCase(
        case_id="x1__mask_silence", base=item, condition="mask_silence",
        expected_behavior="ask", audio_path="a.wav",
        clean_audio_path="c.wav", repair_audio_path="r.wav",
        span_start_sec=0.5, span_end_sec=1.0, sample_rate=24000,
        seed_base=case_seed("x1", "mask_silence"),
    )
    return case.to_json()


def test_judge_pack_selects_prompt_by_language():
    rec_en = pack_trial(_make_case("en"), 0, "What time?", "5 pm, got it.",
                        True)
    assert "evaluator of spoken dialogue" in rec_en["messages"][0]["content"]
    assert rec_en["meta"]["language"] == "en"
    rec_ja = pack_trial(_make_case("ja"), 0, "何時ですか？", "15時ですね。",
                        True)
    assert "音声対話システムの評価者" in rec_ja["messages"][0]["content"]
