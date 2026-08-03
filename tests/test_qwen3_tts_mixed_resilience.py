from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import numpy as np

import scripts.generate_qwen3_tts_data as generate_module
from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueRenderJob,
    DialogueTurn,
    MixedRoleAudioCache,
    build_segments_whole_utterance,
    cache_mixed_role_stage,
    iter_cached_mixed_role_render_jobs,
    mixed_role_cache_config,
    plan_render_job_requests,
    prepare_dialogue_render_job,
)


def _job(index: int, name: str, turns: list[DialogueTurn]) -> DialogueRenderJob:
    dialogue = Dialogue(
        id=name,
        category="test",
        risk_level="low",
        title=name,
        turns=turns,
    )
    return DialogueRenderJob(
        index=index,
        template={
            "id": name,
            "category": "test",
            "risk_level": "low",
            "title": name,
            "turns": [],
        },
        dialogue=dialogue,
        stem=f"sample_{index:03d}_{name}",
        user_voice="Ono_Anna",
        user_override="Ono_Anna",
        other_override="Dylan",
        background_override="Ryan",
        instruct_user=None,
        instruct_moshi=None,
        resolved_preset_key=None,
        log_progress=False,
    )


class SelectiveFailureTTS:
    def __init__(self, role: str) -> None:
        self.role = role
        self.sample_rate = 0
        self.calls: list[list[str]] = []

    def synthesize_many(self, requests):
        texts = [request.text for request in requests]
        self.calls.append(texts)
        self.sample_rate = 24_000
        if any(text.startswith("explode-") for text in texts):
            raise RuntimeError(f"{self.role} synthetic failure")
        return [
            np.full(32, 0.1 if self.role == "user" else 0.2, dtype=np.float32)
            for _ in requests
        ]


def test_mixed_stage_quarantines_one_job_and_keeps_partial_cache(tmp_path) -> None:
    args = SimpleNamespace(dialogue_batch_size=2, whole_utterance_max_chars=150)
    bad = _job(
        1,
        "bad",
        [
            DialogueTurn("user", "keep-this"),
            DialogueTurn("moshi", "moshi-would-be-skipped"),
            DialogueTurn("user", "explode-user", voice_role="other"),
        ],
    )
    good = _job(
        2,
        "good",
        [
            DialogueTurn("user", "good-user"),
            DialogueTurn("moshi", "good-moshi"),
        ],
    )
    cache = MixedRoleAudioCache(tmp_path / "cache", {"case": "failure-map"})

    user_tts = SelectiveFailureTTS("user")
    user_failures = cache_mixed_role_stage(
        [bad, good],
        args,
        user_tts,
        cache,
        "user",
        greeting_pcm=None,
    )
    assert set(user_failures) == {bad.stem}

    bad_requests = plan_render_job_requests(bad, args, None)
    assert cache.load(bad, 0, bad_requests[0]) is not None
    assert cache.load(bad, 2, bad_requests[2]) is None

    moshi_tts = SelectiveFailureTTS("moshi")
    moshi_failures = cache_mixed_role_stage(
        [bad, good],
        args,
        moshi_tts,
        cache,
        "moshi",
        greeting_pcm=None,
        skip_stems=set(user_failures),
    )
    assert moshi_failures == {}
    assert all(
        "moshi-would-be-skipped" not in texts for texts in moshi_tts.calls
    )

    replayed = list(
        iter_cached_mixed_role_render_jobs([good], args, cache, None)
    )
    assert len(replayed) == 1
    assert replayed[0][0].stem == good.stem


def test_mixed_greeting_only_job_uses_greeting_sample_rate(tmp_path) -> None:
    args = SimpleNamespace(dialogue_batch_size=1, whole_utterance_max_chars=150)
    greeting = np.linspace(-0.2, 0.2, 240, dtype=np.float32)
    job = _job(
        1,
        "greeting-only",
        [
            DialogueTurn(
                "moshi",
                "固定挨拶",
                event="opening_greeting",
            )
        ],
    )
    cache = MixedRoleAudioCache(tmp_path / "cache", {"case": "greeting-only"})

    [(replay_job, replay)] = list(
        iter_cached_mixed_role_render_jobs(
            [job],
            args,
            cache,
            greeting,
            greeting_sample_rate=24_000,
        )
    )
    assert replay_job.stem == job.stem
    assert replay.sample_rate == 24_000
    replay.assert_consumed()

    class NeverAligner:
        def align(self, *_args, **_kwargs):
            raise AssertionError("greeting-only dialogue must not invoke alignment")

    segments, _ = build_segments_whole_utterance(
        job.dialogue,
        replay,
        NeverAligner(),
        lead_in_sec=0.3,
        gap_sec=0.2,
        opening_greeting_pcm=greeting,
    )
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0].pcm, greeting)


def test_mixed_cache_identity_includes_attention_and_batch_grouping(
    tmp_path,
) -> None:
    ref = tmp_path / "moshi.wav"
    ref.write_bytes(b"moshi-reference")
    args = SimpleNamespace(
        vllm_stage_config=tmp_path / "unused-stage.yaml",
        qwen_engine="transformers",
        model="customvoice-model",
        qwen_clone_model="clone-model",
        dtype="float16",
        language="Japanese",
        attn_impl="flash_attention_2",
        tts_batch_size=8,
        dialogue_batch_size=3,
        speaker_user="Ono_Anna",
        speaker_other="Dylan",
        speaker_background="Ryan",
        user_speaker_pool_list=["Ono_Anna"],
        instruct_user="user-style",
        instruct_moshi="moshi-style",
        qwen_clone_ref_audio_moshi=ref,
        qwen_clone_ref_text_moshi="reference text",
        qwen_clone_x_vector_only=False,
        whole_utterance_max_chars=150,
        vllm_max_new_tokens=2048,
        qwen_clone_max_new_tokens=4096,
    )

    identity = mixed_role_cache_config(args)

    assert identity["attn_impl"] == "flash_attention_2"
    assert identity["tts_batch_size"] == 8
    assert identity["dialogue_batch_size"] == 3


def test_full_clone_routes_special_user_channel_roles_to_user_reference() -> None:
    template = {
        "id": "clone-routing",
        "category": "test",
        "risk_level": "low",
        "title": "clone routing",
        "turns": [
            {"speaker": "user", "text": "normal"},
            {"speaker": "user", "text": "other", "voice_role": "other"},
            {
                "speaker": "user",
                "text": "background",
                "voice_role": "background",
            },
        ],
    }
    args = SimpleNamespace(
        no_emotion=True,
        no_opening_greeting=True,
        opening_greeting="",
        user_speaker_pool_list=["Ono_Anna"],
        tts_backend="qwen3",
        qwen_full_clone_enabled=True,
        speaker_other="Dylan",
        speaker_background="Ryan",
        log_every=1,
        instruct_user=None,
        instruct_moshi=None,
        whole_utterance=True,
        whole_utterance_max_chars=150,
        style_preset="none",
    )

    clone_job = prepare_dialogue_render_job(1, template, args, {}, 1)
    clone_requests = plan_render_job_requests(clone_job, args, None)

    assert clone_job.user_override == "user"
    assert clone_job.other_override == "user"
    assert clone_job.background_override == "user"
    assert {request.speaker_override for request in clone_requests} == {"user"}

    args.qwen_full_clone_enabled = False
    mixed_job = prepare_dialogue_render_job(1, template, args, {}, 1)
    mixed_requests = plan_render_job_requests(mixed_job, args, None)

    assert mixed_job.user_override == "Ono_Anna"
    assert mixed_job.other_override == "Dylan"
    assert mixed_job.background_override == "Ryan"
    assert {request.speaker_override for request in mixed_requests} == {
        "Ono_Anna",
        "Dylan",
        "Ryan",
    }


def test_mixed_main_excludes_both_stage_failures_and_renders_spare(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    backend_calls: dict[str, list[list[str]]] = {"user": [], "moshi": []}

    class FakeBackend(SelectiveFailureTTS):
        def __init__(self, role: str) -> None:
            super().__init__(role)

        def synthesize_many(self, requests):
            backend_calls[self.role].append(
                [request.text for request in requests]
            )
            return super().synthesize_many(requests)

        def close(self) -> None:
            pass

    class FakeAligner:
        def __init__(self, **_kwargs) -> None:
            pass

        def load(self) -> None:
            pass

        def align(self, audio, sample_rate, texts):
            duration = len(audio) / sample_rate
            step = duration / len(texts)
            spans = [(i * step, (i + 1) * step) for i in range(len(texts))]
            return spans, spans

    def fake_factory(_args, *, model_id, clone_refs):
        del model_id
        return FakeBackend("moshi" if clone_refs else "user")

    rows = [
        {
            "id": "bad-user",
            "category": "test",
            "risk_level": "low",
            "title": "bad user",
            "turns": [
                {"speaker": "user", "text": "keep-user-part"},
                {"speaker": "moshi", "text": "skip-this-moshi"},
                {
                    "speaker": "user",
                    "text": "explode-user",
                    "voice_role": "other",
                },
            ],
        },
        {
            "id": "bad-moshi",
            "category": "test",
            "risk_level": "low",
            "title": "bad moshi",
            "turns": [
                {"speaker": "user", "text": "cached-user-for-bad-moshi"},
                {"speaker": "moshi", "text": "explode-moshi"},
            ],
        },
        {
            "id": "good-spare",
            "category": "test",
            "risk_level": "low",
            "title": "good spare",
            "turns": [
                {"speaker": "user", "text": "good-user"},
                {"speaker": "moshi", "text": "good-moshi"},
            ],
        },
    ]
    dialogue_path = tmp_path / "dialogues.jsonl"
    dialogue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
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
            "--whole-utterance",
            "--no-opening-greeting",
            "--no-emotion",
            "--style-preset",
            "none",
            "--success-target",
            "1",
            "--dialogue-batch-size",
            "3",
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

    with caplog.at_level(logging.INFO):
        generate_module.main()

    manifest_rows = [
        json.loads(line)
        for line in (out_dir / "synthetic_moshi_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(manifest_rows) == 1
    assert "sample_003_good-spare.wav" in manifest_rows[0]["path"]
    assert all(
        "skip-this-moshi" not in texts for texts in backend_calls["moshi"]
    )
    assert "成功 1 件, 失敗 2 件" in caplog.text
