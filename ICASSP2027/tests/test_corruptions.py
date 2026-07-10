import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarify import corruptions  # noqa: E402


SR = 24000


def make_speechlike(duration_sec: float = 2.0) -> np.ndarray:
    """Deterministic pseudo-speech: modulated tone, nonzero everywhere."""
    t = np.arange(int(duration_sec * SR)) / SR
    carrier = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (carrier * envelope).astype(np.float32)


def test_add_noise_span_is_local():
    pcm = make_speechlike()
    rng = np.random.default_rng(0)
    out = corruptions.add_noise_span(pcm, SR, 0.8, 1.2, snr_db=0.0, rng=rng)
    start, end = int(0.8 * SR), int(1.2 * SR)
    assert np.array_equal(out[: start], pcm[: start])
    assert np.array_equal(out[end:], pcm[end:])
    assert not np.array_equal(out[start:end], pcm[start:end])
    assert len(out) == len(pcm)


def test_add_noise_span_snr_scaling():
    pcm = make_speechlike()
    start, end = int(0.8 * SR), int(1.2 * SR)
    diffs = {}
    for snr in (10.0, -10.0):
        out = corruptions.add_noise_span(
            pcm, SR, 0.8, 1.2, snr_db=snr, rng=np.random.default_rng(1)
        )
        noise = out[start:end] - pcm[start:end]
        diffs[snr] = float(np.sqrt(np.mean(noise ** 2)))
    # -10 dB SNR noise must be ~10x the +10 dB noise RMS (clipping aside).
    assert diffs[-10.0] > diffs[10.0] * 5


def test_add_noise_deterministic_with_seed():
    pcm = make_speechlike()
    a = corruptions.add_noise_span(pcm, SR, 0.5, 1.0, 0.0,
                                   np.random.default_rng(42))
    b = corruptions.add_noise_span(pcm, SR, 0.5, 1.0, 0.0,
                                   np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_add_noise_rejects_silent_span():
    pcm = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="RMS"):
        corruptions.add_noise_span(pcm, SR, 0.2, 0.4, 0.0,
                                   np.random.default_rng(0))


def test_mask_silence_zeroes_span_interior():
    pcm = make_speechlike()
    out = corruptions.mask_span(pcm, SR, 0.8, 1.2, fill="silence")
    interior = out[int(0.85 * SR): int(1.15 * SR)]
    assert np.max(np.abs(interior)) < 1e-6
    assert np.array_equal(out[: int(0.8 * SR)], pcm[: int(0.8 * SR)])


def test_mask_noise_replaces_span():
    pcm = make_speechlike()
    out = corruptions.mask_span(pcm, SR, 0.8, 1.2, fill="noise",
                                rng=np.random.default_rng(0))
    interior_in = pcm[int(0.9 * SR): int(1.1 * SR)]
    interior_out = out[int(0.9 * SR): int(1.1 * SR)]
    corr = np.corrcoef(interior_in, interior_out)[0, 1]
    assert abs(corr) < 0.2  # original content gone


def test_lowpass_attenuates_high_band():
    pcm = make_speechlike()
    out = corruptions.lowpass_span(pcm, SR, 0.8, 1.2, cutoff_hz=400.0)
    seg_in = pcm[int(0.9 * SR): int(1.1 * SR)]
    seg_out = out[int(0.9 * SR): int(1.1 * SR)]
    fft_in = np.abs(np.fft.rfft(seg_in))
    fft_out = np.abs(np.fft.rfft(seg_out))
    freqs = np.fft.rfftfreq(len(seg_in), 1 / SR)
    high = freqs > 800
    assert fft_out[high].sum() < fft_in[high].sum() * 0.2


def test_span_outside_audio_raises():
    pcm = make_speechlike(1.0)
    with pytest.raises(ValueError):
        corruptions.add_noise_span(pcm, SR, 2.0, 3.0, 0.0,
                                   np.random.default_rng(0))


def test_default_conditions_registry():
    conds = corruptions.build_default_conditions()
    names = [c.name for c in conds]
    assert "clean" in names and "mask_silence" in names
    assert len(names) == len(set(names))
    clean = corruptions.get_condition("clean")
    assert clean.expected_behavior == "act"
    assert corruptions.get_condition("mask_noise").expected_behavior == "ask"
    pcm = make_speechlike()
    for cond in conds:
        out = cond.apply(pcm, SR, 0.8, 1.2, np.random.default_rng(7))
        assert out.shape == pcm.shape
        assert out.dtype == np.float32
