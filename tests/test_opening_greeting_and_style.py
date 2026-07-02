from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_qwen3_tts_data import (
    AIZUCHI_OVERLAP_TEXTS,
    OPENING_GREETING_INSTRUCT,
    OPENING_GREETING_TEXT,
    resolve_auto_style_preset,
    synthesize_opening_greeting,
)
from scripts.generate_synthetic_moshi_training_data import (
    parse_aizuchi_insertions,
    split_text_into_clauses,
)


class FakeTTS:
    """Counts synthesize calls so cache reuse can be asserted."""

    def __init__(self) -> None:
        self.sample_rate = 0
        self.calls = 0

    def synthesize(self, text, speaker_role, instruct=None, speaker_override=None):
        self.calls += 1
        self.last_instruct = instruct
        self.last_role = speaker_role
        if self.sample_rate == 0:
            self.sample_rate = 24_000
        return np.linspace(0.0, 1.0, 2400, dtype=np.float32)


def test_opening_greeting_text_is_the_fixed_moshimoshi_line() -> None:
    assert OPENING_GREETING_TEXT == "もしもし、こちら孤独孤立相談窓口になります。"


def test_greeting_synthesized_once_and_reused_from_disk_cache(tmp_path) -> None:
    tts = FakeTTS()
    first = synthesize_opening_greeting(
        tts, OPENING_GREETING_TEXT, OPENING_GREETING_INSTRUCT, tmp_path,
        backend="qwen3", model_id="m", speaker="Serena",
    )
    assert tts.calls == 1
    assert tts.last_role == "moshi"
    assert tts.last_instruct == OPENING_GREETING_INSTRUCT

    # 2回目（別プロセス相当）はディスクキャッシュからビット同一で返る
    tts2 = FakeTTS()
    second = synthesize_opening_greeting(
        tts2, OPENING_GREETING_TEXT, OPENING_GREETING_INSTRUCT, tmp_path,
        backend="qwen3", model_id="m", speaker="Serena",
    )
    assert tts2.calls == 0
    assert tts2.sample_rate == 24_000  # キャッシュヒットでも sample_rate が確定する
    np.testing.assert_array_equal(first, second)


def test_greeting_cache_key_depends_on_voice_parameters(tmp_path) -> None:
    tts = FakeTTS()
    synthesize_opening_greeting(
        tts, OPENING_GREETING_TEXT, OPENING_GREETING_INSTRUCT, tmp_path,
        backend="qwen3", model_id="m", speaker="Serena",
    )
    synthesize_opening_greeting(
        tts, OPENING_GREETING_TEXT, OPENING_GREETING_INSTRUCT, tmp_path,
        backend="qwen3", model_id="m", speaker="Vivian",
    )
    assert tts.calls == 2  # 話者が違えば別キャッシュ


def test_auto_style_preset_prefers_explicit_emotional_state() -> None:
    assert (
        resolve_auto_style_preset("whatever", "不安が強い")
        == "counseling_anxious"
    )
    assert (
        resolve_auto_style_preset("whatever", "high_tension")
        == "counseling_high_tension"
    )


def test_auto_style_preset_reads_state_token_before_age_band() -> None:
    # build_use_cases.py の id 形式: {sit}_{conv}_{pers}_{state}_{age}_{gender}_{risk}_...
    assert (
        resolve_auto_style_preset(
            "night_alone_listening_quiet_tearful_30代_女性_medium_occasional_00042"
        )
        == "counseling_tearful"
    )
    assert (
        resolve_auto_style_preset(
            "job_loss_venting_worrier_high_tension_40代_男性_low_none_00001"
        )
        == "counseling_high_tension"
    )
    # state=low と risk=low を混同しない（年齢帯の直前だけを見る）
    assert (
        resolve_auto_style_preset(
            "holiday_smalltalk_cheerful_steady_20代_女性_low_none_00002"
        )
        == "counseling_neutral"
    )
    assert (
        resolve_auto_style_preset(
            "holiday_smalltalk_cheerful_low_20代_女性_high_none_00003"
        )
        == "counseling_sad"
    )


def test_auto_style_preset_falls_back_to_neutral() -> None:
    assert resolve_auto_style_preset("smalltalk_evening_001") == "counseling_neutral"


def test_generation_aizuchi_vocabulary_is_subset_of_tts_overlap_set() -> None:
    """生成側で許可する相づちは、TTS 側の重ね合わせ検出セットに必ず
    含まれていること（含まれないと overlap されず不自然な交互ターンになる）。"""
    clauses = split_text_into_clauses("昨日から眠れなくて、ずっと考えてしまって、つらいです。")
    for text in ["はい。", "ええ。", "えぇ。", "あぁ…。", "そうでしたか。"]:
        insertions = parse_aizuchi_insertions(
            f'{{"insertions":[{{"after_clause":1,"text":"{text}"}}]}}',
            clauses,
            max_insertions=1,
        )
        assert insertions, f"{text} が生成側で許可されていない"
        assert text in AIZUCHI_OVERLAP_TEXTS, f"{text} が TTS 側の overlap 検出に無い"


def test_naruhodo_is_no_longer_generated() -> None:
    clauses = split_text_into_clauses("昨日から眠れなくて、ずっと考えてしまって、つらいです。")
    insertions = parse_aizuchi_insertions(
        '{"insertions":[{"after_clause":1,"text":"なるほど。"}]}',
        clauses,
        max_insertions=1,
    )
    assert insertions == []
    # 旧データの再合成のため、TTS 側の検出セットには残す
    assert "なるほど。" in AIZUCHI_OVERLAP_TEXTS
