from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_qwen3_tts_data import (
    AIZUCHI_OVERLAP_TEXTS,
    OPENING_GREETING_INSTRUCT,
    OPENING_GREETING_TEXT,
    build_opening_greeting_cache_identity,
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


def test_greeting_cache_key_depends_on_canonical_runtime_identity(tmp_path) -> None:
    tts = FakeTTS()
    common = (
        OPENING_GREETING_TEXT,
        OPENING_GREETING_INSTRUCT,
        tmp_path,
    )
    synthesize_opening_greeting(
        tts,
        *common,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity={"clone_ref_sha256": "old", "qwen_engine": "transformers"},
    )
    synthesize_opening_greeting(
        tts,
        *common,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity={"qwen_engine": "transformers", "clone_ref_sha256": "old"},
    )
    assert tts.calls == 1  # dict順が違っても canonical JSON は同一

    synthesize_opening_greeting(
        tts,
        *common,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity={"clone_ref_sha256": "new", "qwen_engine": "transformers"},
    )
    assert tts.calls == 2


def test_greeting_cache_recovers_from_corrupt_npz_atomically(tmp_path) -> None:
    identity = {"clone_ref_sha256": "same"}
    first_tts = FakeTTS()
    synthesize_opening_greeting(
        first_tts,
        OPENING_GREETING_TEXT,
        OPENING_GREETING_INSTRUCT,
        tmp_path,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity=identity,
    )
    cache_path = next(tmp_path.glob("greeting_*.npz"))
    cache_path.write_bytes(b"not a zip archive")

    recovery_tts = FakeTTS()
    recovered = synthesize_opening_greeting(
        recovery_tts,
        OPENING_GREETING_TEXT,
        OPENING_GREETING_INSTRUCT,
        tmp_path,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity=identity,
    )
    assert recovery_tts.calls == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))

    cached_tts = FakeTTS()
    cached = synthesize_opening_greeting(
        cached_tts,
        OPENING_GREETING_TEXT,
        OPENING_GREETING_INSTRUCT,
        tmp_path,
        backend="qwen3",
        model_id="base",
        speaker="clone:moshi",
        cache_identity=identity,
    )
    assert cached_tts.calls == 0
    np.testing.assert_array_equal(recovered, cached)


def test_greeting_shared_cache_has_one_cross_thread_writer(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    calls_lock = threading.Lock()

    class BlockingTTS(FakeTTS):
        def synthesize(self, text, speaker_role, instruct=None, speaker_override=None):
            with calls_lock:
                calls.append(1)
            started.set()
            assert release.wait(timeout=5)
            return super().synthesize(
                text,
                speaker_role,
                instruct=instruct,
                speaker_override=speaker_override,
            )

    def render(tts):
        return synthesize_opening_greeting(
            tts,
            OPENING_GREETING_TEXT,
            OPENING_GREETING_INSTRUCT,
            tmp_path,
            backend="qwen3",
            model_id="base",
            speaker="clone:moshi",
            cache_identity={"clone_ref_sha256": "shared"},
        )

    first_tts = BlockingTTS()
    second_tts = BlockingTTS()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(render, first_tts)
        assert started.wait(timeout=5)
        second_future = executor.submit(render, second_tts)
        release.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert len(calls) == 1
    assert first_tts.calls + second_tts.calls == 1
    np.testing.assert_array_equal(first, second)


def test_opening_identity_fingerprints_clone_ref_and_stage_config(tmp_path) -> None:
    ref = tmp_path / "moshi.wav"
    stage = tmp_path / "stage.yaml"
    ref.write_bytes(b"voice-a")
    stage.write_text("temperature: 0.9\n", encoding="utf-8")
    args = SimpleNamespace(
        tts_backend="qwen3",
        dtype="float16",
        language="Japanese",
        qwen_engine="vllm-omni",
        qwen_voice_mode_resolved="mixed",
        attn_impl="default",
        vllm_max_new_tokens=2048,
        qwen_clone_max_new_tokens=4096,
        tts_batch_size=16,
        vllm_stage_config=stage,
        qwen_clone_enabled=True,
        qwen_clone_ref_audio_moshi=str(ref),
        qwen_clone_ref_text_moshi="参照文",
        qwen_clone_x_vector_only=False,
    )

    first = build_opening_greeting_cache_identity(
        args, model_id="base", speaker="clone:moshi"
    )
    ref.write_bytes(b"voice-b")
    second = build_opening_greeting_cache_identity(
        args, model_id="base", speaker="clone:moshi"
    )
    stage.write_text("temperature: 0.7\n", encoding="utf-8")
    third = build_opening_greeting_cache_identity(
        args, model_id="base", speaker="clone:moshi"
    )

    assert first["model_id"] == "base"
    assert first["qwen_engine"] == "vllm-omni"
    assert first["vllm_max_new_tokens"] == 2048
    assert first["clone_max_new_tokens"] == 4096
    assert first["clone_ref_audio_moshi_sha256"] != second[
        "clone_ref_audio_moshi_sha256"
    ]
    assert second["vllm_stage_config_sha256"] != third[
        "vllm_stage_config_sha256"
    ]


def test_opening_identity_fingerprints_effective_moss_moshi_reference(
    tmp_path,
) -> None:
    roles: dict[str, dict[str, str]] = {}
    for role in ("user", "moshi", "other", "background"):
        ref = tmp_path / f"{role}.wav"
        ref.write_bytes(f"{role}-voice-a".encode())
        roles[role] = {
            "path": ref.name,
            "transcript": f"{role} reference text",
        }
    refs_json = tmp_path / "refs.json"
    refs_json.write_text(
        json.dumps({"roles": roles}, ensure_ascii=False),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        tts_backend="moss-ttsd",
        dtype="float16",
        language="Japanese",
        moss_codec_model="moss-codec",
        moss_refs_json=refs_json,
    )

    first = build_opening_greeting_cache_identity(
        args, model_id="moss-model", speaker="moshi"
    )
    (tmp_path / "moshi.wav").write_bytes(b"moshi-voice-b")
    second = build_opening_greeting_cache_identity(
        args, model_id="moss-model", speaker="moshi"
    )

    assert first["moss_ref_audio_moshi"] == str(
        (tmp_path / "moshi.wav").resolve()
    )
    assert first["moss_ref_text_moshi"] == "moshi reference text"
    assert first["moss_ref_audio_moshi_sha256"] != second[
        "moss_ref_audio_moshi_sha256"
    ]


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
