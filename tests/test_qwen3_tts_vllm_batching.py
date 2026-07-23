from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.generate_qwen3_tts_data import (
    Dialogue,
    DialogueTurn,
    ReplayTTS,
    build_segments_whole_utterance,
    iter_dialogue_render_jobs,
    plan_whole_utterance_synthesis,
)
from scripts.qwen3_tts_vllm_backend import SynthesisRequest, VLLMQwen3TTS


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
