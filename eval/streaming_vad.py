"""Streaming utterance-end detection used by turn-based baseline evaluations.

This deliberately models the *online* decision: 20-ms capture frames are
observed in order, while Silero consumes its native 32-ms windows. An endpoint
is returned only once the configured silence has actually arrived. The returned
``detected_at_sec`` is therefore the time at which ASR/LLM/TTS may start,
rather than the earlier estimated end of speech.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from full_duplex_audio import resample_linear


@dataclass(frozen=True)
class VadEndpoint:
    detected_at_sec: float
    speech_end_sec: float | None
    audio_end_sample: int
    backend: str


def find_streaming_endpoint(
    pcm: np.ndarray,
    sample_rate: int,
    *,
    tail_sec: float = 8.0,
    silence_ms: int = 800,
    threshold: float = 0.5,
    input_frame_ms: int = 20,
) -> VadEndpoint | None:
    """Feed audio plus a silent tail to Silero's streaming ``VADIterator``.

    Silero accepts fixed 512-sample (32-ms) 16-kHz chunks.  A response is
    eligible only after the iterator emits ``end``; natural pauses before that
    point remain part of the actual stream, exactly as they would at runtime.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if tail_sec < 0:
        raise ValueError("tail_sec must be >= 0")
    if silence_ms <= 0:
        raise ValueError("silence_ms must be > 0")
    if input_frame_ms <= 0:
        raise ValueError("input_frame_ms must be > 0")
    try:
        import torch
        from silero_vad import VADIterator, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Streaming VAD requires silero-vad. Run `uv sync` in the main project environment."
        ) from exc

    received = np.concatenate([
        np.asarray(pcm, dtype=np.float32),
        np.zeros(int(round(tail_sec * sample_rate)), dtype=np.float32),
    ])
    stream_16k = resample_linear(received, sample_rate, 16000)
    # VADIterator requires an exact 512-sample window. Padding only affects
    # the final stream frame and is equivalent to receiving a little more zero
    # audio after the configured tail.
    chunk_size = 512
    if len(stream_16k) % chunk_size:
        stream_16k = np.pad(stream_16k, (0, chunk_size - len(stream_16k) % chunk_size))

    model = load_silero_vad()
    iterator = VADIterator(
        model,
        threshold=threshold,
        sampling_rate=16000,
        min_silence_duration_ms=silence_ms,
        speech_pad_ms=0,
    )
    capture_size = max(1, int(round(sample_rate * input_frame_ms / 1000)))
    vad_cursor_16k = 0
    for received_samples in range(capture_size, len(received) + capture_size, capture_size):
        received_samples = min(received_samples, len(received))
        available_16k = min(len(stream_16k), int(received_samples * 16000 / sample_rate))
        while vad_cursor_16k + chunk_size <= available_16k:
            end_sample_16k = vad_cursor_16k + chunk_size
            event = iterator(
                torch.from_numpy(stream_16k[vad_cursor_16k:end_sample_16k]),
                return_seconds=False,
            )
            vad_cursor_16k = end_sample_16k
            if not event or "end" not in event:
                continue
            speech_end_samples = int(round(float(event["end"]) * sample_rate / 16000))
            return VadEndpoint(
                # VAD's 32-ms window was only available after this 20-ms
                # capture callback completed; do not backdate latency.
                detected_at_sec=received_samples / sample_rate,
                speech_end_sec=max(0.0, speech_end_samples / sample_rate),
                audio_end_sample=received_samples,
                backend="silero_streaming_20ms_capture",
            )
        if received_samples == len(received):
            break
    return None
