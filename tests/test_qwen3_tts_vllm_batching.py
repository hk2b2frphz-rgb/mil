from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

import scripts.generate_qwen3_tts_data as generate_module
from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueTurn,
    MixedRoleAudioCache,
    ReplayTTS,
    build_segments_whole_utterance,
    cache_mixed_role_stage,
    iter_cached_mixed_role_render_jobs,
    iter_dialogue_render_jobs,
    parse_args,
    plan_whole_utterance_synthesis,
    prepare_pending_dialogue_render_jobs,
)
import pytest

from scripts.qwen3_tts_vllm_backend import (
    CloneReference,
    SynthesisRequest,
    VLLMQwen3TTS,
)


class RecordingTTS:
    def __init__(self) -> None:
        self.sample_rate = 24_000
        self.requests: list[SynthesisRequest] = []

    def synthesize(self, text, speaker_role, instruct=None, speaker_override=None):
        self.requests.append(
            SynthesisRequest(text, speaker_role, instruct, speaker_override)
        )
        return np.ones(2400, dtype=np.float32)


class ProportionalAligner:
    def align(self, audio, sample_rate, texts):
        duration = len(audio) / sample_rate
        step = duration / len(texts)
        spans = [(i * step, (i + 1) * step) for i in range(len(texts))]
        return spans, spans


def test_whole_utterance_plan_matches_render_call_order() -> None:
    dialogue = Dialogue(
        id="batch",
        category="test",
        risk_level="low",
        title="batch",
        turns=[
            DialogueTurn("user", "あ" * 8),
            DialogueTurn("moshi", "応答"),
            DialogueTurn("user", "別話者", voice_role="other"),
            DialogueTurn("user", "い" * 8),
        ],
    )
    kwargs = {
        "user_speaker_override": "Ono_Anna",
        "other_speaker_override": "Dylan",
        "background_speaker_override": "Eric",
        "instruct_user": "user-style",
        "instruct_moshi": "moshi-style",
        "max_chars_per_synthesis": 10,
    }
    planned = plan_whole_utterance_synthesis(dialogue, **kwargs)
    recorder = RecordingTTS()
    build_segments_whole_utterance(
        dialogue,
        recorder,
        ProportionalAligner(),
        0.3,
        0.2,
        **kwargs,
    )
    assert planned == recorder.requests
    assert len(planned) == 4


def test_replay_tts_preserves_planned_order() -> None:
    requests = [
        SynthesisRequest("one", "user", "calm", "Ono_Anna"),
        SynthesisRequest("two", "moshi", None, None),
    ]
    replay = ReplayTTS(
        requests,
        [np.ones(4, dtype=np.float32), np.ones(6, dtype=np.float32)],
        24_000,
    )
    assert replay.synthesize("one", "user", "calm", "Ono_Anna").size == 4
    assert replay.synthesize("two", "moshi").size == 6
    replay.assert_consumed()


def test_dialogue_iterator_batches_requests_across_dialogues() -> None:
    class FakeBatchTTS:
        sample_rate = 24_000

        def __init__(self) -> None:
            self.calls: list[list[SynthesisRequest]] = []

        def synthesize_many(self, requests):
            self.calls.append(list(requests))
            return [np.ones(2400, dtype=np.float32) for _ in requests]

    args = SimpleNamespace(
        no_emotion=True,
        no_opening_greeting=True,
        opening_greeting="",
        user_speaker_pool_list=["Ono_Anna"],
        tts_backend="qwen3",
        speaker_other="Dylan",
        speaker_background="Eric",
        log_every=20,
        whole_utterance=True,
        style_preset="none",
        instruct_user=None,
        instruct_moshi=None,
        qwen_engine="vllm-omni",
        dialogue_batch_size=2,
        tts_batch_size=16,
        whole_utterance_max_chars=150,
        resume=False,
    )
    templates = [
        {
            "id": f"d{i}",
            "category": "test",
            "risk_level": "low",
            "title": "test",
            "turns": [
                {"speaker": "user", "text": "相談です"},
                {"speaker": "moshi", "text": "伺います"},
            ],
        }
        for i in range(2)
    ]
    tts = FakeBatchTTS()
    jobs = list(iter_dialogue_render_jobs(templates, args, {}, tts, None, set()))

    assert len(jobs) == 2
    assert len(tts.calls) == 1
    assert len(tts.calls[0]) == 4
    assert all(isinstance(replay, ReplayTTS) for _, replay in jobs)


class FakeOmni:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, prompts, use_tqdm=False):
        self.batch_sizes.append(len(prompts))
        outputs = []
        # Deliberately reverse completion order; the backend must restore it.
        for i in reversed(range(len(prompts))):
            value = float(prompts[i]["marker"])
            mm = {
                "audio": [np.full(3, value, dtype=np.float32)],
                "sr": 24_000,
            }
            request_output = SimpleNamespace(
                request_id=f"{i}_fake",
                outputs=[SimpleNamespace(multimodal_output=mm)],
            )
            outputs.append(SimpleNamespace(request_output=request_output))
        return outputs


def _make_clone_backend(tmp_path, **overrides):
    config = tmp_path / "stage.yaml"
    config.write_text("stages: []\n", encoding="utf-8")
    kwargs = dict(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype_str="float16",
        speaker_user="Ono_Anna",
        speaker_moshi="Serena",
        language="Japanese",
        instruct_user=None,
        instruct_moshi="落ち着いて",
        batch_size=2,
        stage_configs_path=config,
        clone_refs={
            "user": CloneReference("refs/user.wav", "こんにちは"),
            "moshi": CloneReference("refs/moshi.wav", "もしもし"),
        },
    )
    kwargs.update(overrides)
    return VLLMQwen3TTS(**kwargs)


def test_clone_prompt_uses_base_task_and_role_reference(tmp_path) -> None:
    backend = _make_clone_backend(tmp_path)
    backend._tokenizer = object()  # _to_prompt only asserts presence
    prompt = backend._to_prompt(SynthesisRequest("お元気ですか", "moshi"))
    info = prompt["additional_information"]
    assert info["task_type"] == ["Base"]
    assert info["ref_audio"] == ["refs/moshi.wav"]
    assert info["ref_text"] == ["もしもし"]
    assert info["x_vector_only_mode"] == [False]
    assert info["instruct"] == ["落ち着いて"]
    assert "speaker" not in info
    assert len(prompt["prompt_token_ids"]) > 0


def test_clone_x_vector_only_allows_missing_ref_text(tmp_path) -> None:
    backend = _make_clone_backend(
        tmp_path,
        clone_refs={
            "user": CloneReference("refs/user.wav"),
            "moshi": CloneReference("refs/moshi.wav"),
        },
        clone_x_vector_only=True,
    )
    backend._tokenizer = object()
    info = backend._to_prompt(SynthesisRequest("テスト", "user"))["additional_information"]
    assert info["x_vector_only_mode"] == [True]
    assert "ref_text" not in info


def test_clone_mode_validation_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="Base model"):
        _make_clone_backend(
            tmp_path, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
    with pytest.raises(ValueError, match="ref_text"):
        _make_clone_backend(
            tmp_path,
            clone_refs={
                "user": CloneReference("refs/user.wav"),
                "moshi": CloneReference("refs/moshi.wav", "text"),
            },
        )
    # CustomVoice mode still rejects non-CustomVoice models.
    with pytest.raises(ValueError, match="CustomVoice"):
        _make_clone_backend(
            tmp_path, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", clone_refs=None
        )


def test_clone_backend_accepts_moshi_only_for_mixed_mode(tmp_path) -> None:
    backend = _make_clone_backend(
        tmp_path,
        clone_refs={"moshi": CloneReference("refs/moshi.wav", "text")},
    )
    backend._tokenizer = object()
    info = backend._to_prompt(
        SynthesisRequest("応答します", "moshi")
    )["additional_information"]
    assert info["ref_audio"] == ["refs/moshi.wav"]
    with pytest.raises(ValueError, match="No clone reference"):
        backend._to_prompt(SynthesisRequest("相談です", "user"))


def test_clone_unknown_speaker_override_raises(tmp_path) -> None:
    backend = _make_clone_backend(tmp_path)
    backend._tokenizer = object()
    with pytest.raises(ValueError, match="No clone reference"):
        backend._to_prompt(
            SynthesisRequest("テスト", "user", speaker_override="Dylan")
        )


def test_vllm_backend_batches_and_restores_original_order(tmp_path) -> None:
    config = tmp_path / "stage.yaml"
    config.write_text("stages: []\n", encoding="utf-8")
    backend = VLLMQwen3TTS(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype_str="float16",
        speaker_user="Ono_Anna",
        speaker_moshi="Serena",
        language="Japanese",
        instruct_user=None,
        instruct_moshi=None,
        batch_size=2,
        stage_configs_path=config,
    )
    fake_omni = FakeOmni()
    backend._omni = fake_omni
    backend._to_prompt = lambda request: {"marker": len(request.text)}  # type: ignore[method-assign]

    requests = [
        SynthesisRequest("aaaaa", "user"),
        SynthesisRequest("b", "user"),
        SynthesisRequest("ccc", "moshi"),
    ]
    audio = backend.synthesize_many(requests)

    assert fake_omni.batch_sizes == [2, 1]
    assert [float(item[0]) for item in audio] == [5.0, 1.0, 3.0]
    assert backend.sample_rate == 24_000
    stats = backend.performance_stats()
    assert stats["batches"] == 2
    assert stats["requests"] == 3
    assert stats["audio_sec"] == 9 / 24_000
    assert stats["inference_wall_sec"] >= 0


def test_mixed_role_two_pass_cache_restores_request_order(tmp_path) -> None:
    events: list[str] = []

    class FakeRoleTTS:
        def __init__(self, role: str, marker: float) -> None:
            self.role = role
            self.marker = marker
            self.sample_rate = 0
            self.calls: list[list[SynthesisRequest]] = []

        def synthesize_many(self, requests):
            if not self.calls:
                events.append(f"{self.role}:load")
            self.calls.append(list(requests))
            events.append(f"{self.role}:synthesize")
            self.sample_rate = 24_000
            return [
                np.full(8, self.marker + i, dtype=np.float32)
                for i, _ in enumerate(requests)
            ]

        def close(self):
            events.append(f"{self.role}:close")

    args = SimpleNamespace(
        no_emotion=True,
        no_opening_greeting=False,
        opening_greeting="固定の挨拶",
        user_speaker_pool_list=["Ono_Anna"],
        tts_backend="qwen3",
        speaker_other="Dylan",
        speaker_background="Eric",
        log_every=20,
        whole_utterance=True,
        style_preset="none",
        instruct_user=None,
        instruct_moshi=None,
        qwen_engine="vllm-omni",
        qwen_full_clone_enabled=False,
        dialogue_batch_size=1,
        tts_batch_size=16,
        whole_utterance_max_chars=150,
        resume=False,
    )
    templates = [
        {
            "id": f"mixed_{i}",
            "category": "test",
            "risk_level": "low",
            "title": "test",
            "turns": [
                {"speaker": "user", "text": f"相談{i}"},
                {"speaker": "moshi", "text": f"応答{i}"},
                {
                    "speaker": "user",
                    "text": f"第三者{i}",
                    "voice_role": "other",
                },
            ],
        }
        for i in range(2)
    ]
    jobs = prepare_pending_dialogue_render_jobs(templates, args, {}, set())
    cache = MixedRoleAudioCache(tmp_path / "cache", {"test": "mixed"})
    user_tts = FakeRoleTTS("user", 10.0)
    moshi_tts = FakeRoleTTS("moshi", 20.0)

    cache_mixed_role_stage(
        jobs,
        args,
        user_tts,
        cache,
        "user",
        greeting_pcm=np.zeros(0, dtype=np.float32),
    )
    user_tts.close()
    cache_mixed_role_stage(
        jobs,
        args,
        moshi_tts,
        cache,
        "moshi",
        greeting_pcm=np.ones(1, dtype=np.float32),
    )
    moshi_tts.close()

    assert events == [
        "user:load",
        "user:synthesize",
        "user:synthesize",
        "user:close",
        "moshi:load",
        "moshi:synthesize",
        "moshi:synthesize",
        "moshi:close",
    ]
    assert all(
        request.speaker_role == "user"
        for batch in user_tts.calls
        for request in batch
    )
    assert all(
        request.speaker_role == "moshi"
        for batch in moshi_tts.calls
        for request in batch
    )

    replay_jobs = list(
        iter_cached_mixed_role_render_jobs(
            jobs, args, cache, np.ones(1, dtype=np.float32)
        )
    )
    assert len(replay_jobs) == 2
    for job, replay in replay_jobs:
        planned = plan_whole_utterance_synthesis(
            job.dialogue,
            user_speaker_override=job.user_override,
            other_speaker_override=job.other_override,
            background_speaker_override=job.background_override,
            instruct_user=job.instruct_user,
            instruct_moshi=job.instruct_moshi,
            max_chars_per_synthesis=args.whole_utterance_max_chars,
            opening_greeting_pcm=np.ones(1, dtype=np.float32),
        )
        values = [
            float(
                replay.synthesize(
                    request.text,
                    request.speaker_role,
                    request.instruct,
                    request.speaker_override,
                )[0]
            )
            for request in planned
        ]
        assert values == [10.0, 20.0, 11.0]
        replay.assert_consumed()

    # A resumed pass reads every request from disk and does not load a model.
    resumed_user = FakeRoleTTS("resumed-user", 99.0)
    cache_mixed_role_stage(
        jobs,
        args,
        resumed_user,
        cache,
        "user",
        greeting_pcm=np.zeros(0, dtype=np.float32),
    )
    assert resumed_user.calls == []


def test_mixed_cli_requires_only_moshi_clone_reference(
    tmp_path, monkeypatch
) -> None:
    ref = tmp_path / "moshi.wav"
    ref.write_bytes(b"reference")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_qwen3_tts_data.py",
            "--out-dir",
            str(tmp_path / "out"),
            "--whole-utterance",
            "--qwen-voice-mode",
            "mixed",
            "--model",
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "--qwen-clone-model",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "--qwen-clone-ref-audio-moshi",
            str(ref),
            "--qwen-clone-ref-text-moshi",
            "参照テキスト",
            "--user-speaker-pool",
            "Ono_Anna,Sohee",
        ],
    )
    args = parse_args()
    assert args.qwen_voice_mode_resolved == "mixed"
    assert args.qwen_mixed_role_enabled
    assert not args.qwen_full_clone_enabled
    assert args.user_speaker_pool_list == ["Ono_Anna", "Sohee"]


def test_mixed_cli_rejects_user_clone_reference(tmp_path, monkeypatch) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"reference")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_qwen3_tts_data.py",
            "--out-dir",
            str(tmp_path / "out"),
            "--whole-utterance",
            "--qwen-voice-mode",
            "mixed",
            "--qwen-clone-ref-audio-moshi",
            str(ref),
            "--qwen-clone-ref-text-moshi",
            "moshi",
            "--qwen-clone-ref-audio-user",
            str(ref),
            "--qwen-clone-ref-text-user",
            "user",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_mixed_main_runs_models_sequentially_and_writes_role_metadata(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role
            self.sample_rate = 0
            self.closed = False

        def synthesize_many(self, requests):
            if self.sample_rate == 0:
                events.append(f"{self.role}:load")
                self.sample_rate = 24_000
            events.append(f"{self.role}:synthesize:{len(requests)}")
            value = 0.1 if self.role == "user" else 0.2
            return [
                np.full(2400, value, dtype=np.float32)
                for _ in requests
            ]

        def close(self):
            if not self.closed:
                events.append(f"{self.role}:close")
                self.closed = True

    class FakeAligner:
        def __init__(self, **_kwargs) -> None:
            pass

        def load(self) -> None:
            events.append("aligner:load")

        def align(self, audio, sample_rate, texts):
            duration = len(audio) / sample_rate
            step = duration / len(texts)
            spans = [(i * step, (i + 1) * step) for i in range(len(texts))]
            return spans, spans

    def fake_factory(_args, *, model_id, clone_refs):
        del model_id
        return FakeBackend("moshi" if clone_refs else "user")

    dialogue_path = tmp_path / "dialogues.jsonl"
    dialogue_path.write_text(
        json.dumps(
            {
                "id": "mixed-e2e",
                "category": "test",
                "risk_level": "low",
                "title": "mixed",
                "turns": [
                    {"speaker": "user", "text": "相談があります"},
                    {"speaker": "moshi", "text": "お話しください"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ref = tmp_path / "moshi.wav"
    ref.write_bytes(b"reference")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(generate_module, "create_qwen_backend", fake_factory)
    monkeypatch.setattr(generate_module, "ForcedAligner", FakeAligner)
    monkeypatch.setattr(
        generate_module,
        "write_wav",
        lambda path, _stereo, _sample_rate: path.write_bytes(b"wav"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_qwen3_tts_data.py",
            "--out-dir",
            str(out_dir),
            "--dialogues-jsonl",
            str(dialogue_path),
            "--allow-invalid-duplex",
            "--whole-utterance",
            "--no-opening-greeting",
            "--no-emotion",
            "--style-preset",
            "none",
            "--qwen-voice-mode",
            "mixed",
            "--qwen-clone-ref-audio-moshi",
            str(ref),
            "--qwen-clone-ref-text-moshi",
            "参照テキスト",
            "--user-speaker-pool",
            "Ono_Anna",
        ],
    )
    generate_module.main()

    assert events == [
        "user:load",
        "user:synthesize:1",
        "user:close",
        "moshi:load",
        "moshi:synthesize:1",
        "moshi:close",
        "aligner:load",
    ]
    metadata_paths = list((out_dir / "data_stereo").glob("*.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(
        metadata_paths[0].read_text(encoding="utf-8")
    )["metadata"]
    assert metadata["qwen_voice_mode"] == "mixed"
    assert metadata["tts_models_by_role"] == {
        "user": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "moshi": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }
    assert metadata["qwen_clone"]["roles"] == ["moshi"]
    assert not any((out_dir / ".qwen_mixed_cache").glob("*/config.json"))
