from __future__ import annotations

import sys
import wave
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.qwen3_tts_vllm_backend import (
    CloneReference,
    SynthesisRequest,
    VLLMQwen3TTS,
)


def _make_backend(tmp_path, *, max_new_tokens: int = 321) -> VLLMQwen3TTS:
    config = tmp_path / "stage.yaml"
    config.write_text("stages: []\n", encoding="utf-8")
    return VLLMQwen3TTS(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype_str="float16",
        speaker_user="Ono_Anna",
        speaker_moshi="Serena",
        language="Japanese",
        instruct_user=None,
        instruct_moshi=None,
        batch_size=1,
        stage_configs_path=config,
        max_new_tokens=max_new_tokens,
        clone_refs={
            "moshi": CloneReference("ref.wav", "reference transcript"),
        },
    )


def test_ref_code_len_uses_nested_path_and_caches_soundfile_info(
    tmp_path, monkeypatch
) -> None:
    backend = _make_backend(tmp_path)
    backend._talker_config = SimpleNamespace(codec_frame_rate=12)
    calls: list[str] = []

    def fake_info(path: str) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(duration=1.01)

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(info=fake_info),
    )
    ref_path = tmp_path / "reference.wav"

    assert backend._estimate_ref_code_len([[str(ref_path)]]) == 12
    assert backend._estimate_ref_code_len(str(ref_path)) == 12
    assert calls == [str(ref_path)]

    # close() clears cached path metadata even when the engine was not loaded.
    backend.close()
    assert backend._ref_code_len_cache == {}


def test_ref_code_len_falls_back_to_wave_and_default_codec_rate(
    tmp_path, monkeypatch
) -> None:
    ref_path = tmp_path / "reference.wav"
    with wave.open(str(ref_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\0\0" * 12_000)

    def fail_info(_path: str) -> None:
        raise RuntimeError("soundfile unavailable")

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(info=fail_info),
    )
    backend = _make_backend(tmp_path)
    backend._talker_config = SimpleNamespace(codec_frame_rate=None)

    assert backend._estimate_ref_code_len([str(ref_path)]) == 6


def test_prompt_length_builder_receives_reference_length_estimator(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakePromptBuilder:
        @staticmethod
        def estimate_prompt_len_from_additional_information(
            *,
            additional_information,
            estimate_ref_code_len,
            **_kwargs,
        ):
            captured["callback"] = estimate_ref_code_len
            return estimate_ref_code_len(additional_information["ref_audio"])

    package_names = [
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.qwen3_tts",
    ]
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, package_name, package)
    module_name = (
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder"
    )
    builder_module = ModuleType(module_name)
    builder_module.Qwen3TTSPromptEmbedsBuilder = FakePromptBuilder
    monkeypatch.setitem(sys.modules, module_name, builder_module)

    backend = _make_backend(tmp_path)
    backend._tokenizer = lambda text, padding=False: {"input_ids": [text, padding]}
    backend._estimate_ref_code_len = lambda _audio: 37  # type: ignore[method-assign]
    info = {
        "task_type": ["Base"],
        "ref_audio": [["reference.wav"]],
    }

    assert backend._estimate_prompt_len(info) == 37
    assert captured["callback"] is backend._estimate_ref_code_len


def test_generate_passes_copied_stage_zero_token_limit(
    tmp_path,
) -> None:
    class FakeOmni:
        def __init__(self) -> None:
            self.default_sampling_params_list = [
                SimpleNamespace(max_tokens=4096, extra_args={"nested": [1]}),
                SimpleNamespace(max_tokens=65536),
            ]
            self.received_sampling_params = None

        def generate(
            self,
            prompts,
            sampling_params_list=None,
            *,
            use_tqdm=True,
        ):
            assert len(prompts) == 1
            assert use_tqdm is False
            self.received_sampling_params = sampling_params_list
            output = SimpleNamespace(
                request_id="0_fake",
                outputs=[
                    SimpleNamespace(
                        multimodal_output={
                            "audio": [np.ones(4, dtype=np.float32)],
                            "sr": 24_000,
                        }
                    )
                ],
            )
            return [SimpleNamespace(request_output=output)]

    backend = _make_backend(tmp_path, max_new_tokens=321)
    omni = FakeOmni()
    backend._omni = omni
    backend._to_prompt = lambda _request: {"prompt_token_ids": [0]}  # type: ignore[method-assign]

    audio = backend._generate_batch(
        [SynthesisRequest("hello", "moshi")]
    )

    received = omni.received_sampling_params
    assert received is not None
    assert received[0].max_tokens == 321
    assert received[1].max_tokens == 65536
    assert received[0] is not omni.default_sampling_params_list[0]
    assert received[1] is not omni.default_sampling_params_list[1]
    assert omni.default_sampling_params_list[0].max_tokens == 4096
    received[0].extra_args["nested"].append(2)
    assert omni.default_sampling_params_list[0].extra_args == {"nested": [1]}
    assert audio[0].shape == (4,)


def test_generate_failure_drops_engine_and_next_call_reloads(
    tmp_path,
) -> None:
    class FailingOmni:
        def __init__(self) -> None:
            self.default_sampling_params_list = [
                SimpleNamespace(max_tokens=4096),
                SimpleNamespace(max_tokens=65536),
            ]
            self.close_calls = 0

        def generate(self, _prompts, sampling_params_list=None, **_kwargs):
            assert sampling_params_list[0].max_tokens == 321
            raise ValueError("generation failed")

        def close(self) -> None:
            self.close_calls += 1
            # Simulate Omni.generate() having already closed the engine.
            raise RuntimeError("engine was already closed")

    class SuccessfulOmni:
        def __init__(self) -> None:
            self.default_sampling_params_list = [
                SimpleNamespace(max_tokens=4096),
                SimpleNamespace(max_tokens=65536),
            ]

        def generate(self, _prompts, sampling_params_list=None, **_kwargs):
            assert sampling_params_list[0].max_tokens == 321
            output = SimpleNamespace(
                request_id="0_reloaded",
                outputs=[
                    SimpleNamespace(
                        multimodal_output={
                            "audio": [np.ones(3, dtype=np.float32)],
                            "sr": 24_000,
                        }
                    )
                ],
            )
            return [SimpleNamespace(request_output=output)]

    backend = _make_backend(tmp_path)
    failing_omni = FailingOmni()
    backend._omni = failing_omni
    backend._to_prompt = lambda _request: {"prompt_token_ids": [0]}  # type: ignore[method-assign]
    request = SynthesisRequest("hello", "moshi")

    with pytest.raises(ValueError, match="generation failed"):
        backend._generate_batch([request])
    assert failing_omni.close_calls == 1
    assert backend._omni is None

    successful_omni = SuccessfulOmni()
    load_calls = 0

    def fake_load() -> None:
        nonlocal load_calls
        load_calls += 1
        assert backend._omni is None
        backend._omni = successful_omni

    backend.load = fake_load  # type: ignore[method-assign]
    audio = backend.synthesize_many([request])

    assert load_calls == 1
    assert backend._omni is successful_omni
    assert audio[0].shape == (3,)
