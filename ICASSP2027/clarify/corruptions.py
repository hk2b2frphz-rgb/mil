"""Span-localized acoustic corruption operators.

Every operator takes a mono float32 PCM array plus a (start_sec, end_sec)
span and returns a new array of identical length; samples outside the span
are bit-identical to the input. This locality is the experimental core: the
carrier utterance stays intelligible while the information-bearing span
(slot value) is degraded to a controlled degree.

Only numpy/scipy are used so the module runs on the CPU-only local machine
and inside any cluster environment.

Determinism: all stochastic operators take an explicit ``rng``
(numpy.random.Generator). Benchmark construction derives one rng per case
from a stable per-case seed, so a rebuilt dataset is bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _span_to_samples(
    pcm: np.ndarray, sample_rate: int, start_sec: float, end_sec: float
) -> tuple[int, int]:
    if end_sec <= start_sec:
        raise ValueError(f"empty span: [{start_sec}, {end_sec}]")
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(len(pcm), int(round(end_sec * sample_rate)))
    if end <= start:
        raise ValueError(
            f"span [{start_sec}, {end_sec}]s falls outside audio of "
            f"{len(pcm) / sample_rate:.2f}s"
        )
    return start, end


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _fade_window(n: int, ramp: int) -> np.ndarray:
    """Trapezoid window so span edits never click at the boundaries."""
    ramp = min(ramp, n // 2)
    win = np.ones(n, dtype=np.float32)
    if ramp > 0:
        edge = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        win[:ramp] = edge
        win[-ramp:] = edge[::-1]
    return win


def make_babble(
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    n_talkers: int = 8,
) -> np.ndarray:
    """Synthetic babble: sum of amplitude-modulated speech-shaped noise.

    Real multi-talker babble recordings are not redistributable with the
    dataset; a synthetic speech-shaped surrogate keeps the benchmark fully
    reconstructable from code + seeds. Speech-shaped = white noise through a
    2nd-order low-shelf approximating the long-term speech spectrum tilt
    (-6 dB/oct above ~500 Hz), each "talker" modulated at syllabic rate
    (2-6 Hz).
    """
    total = np.zeros(n_samples, dtype=np.float64)
    # -6 dB/oct tilt via one-pole lowpass at 500 Hz.
    b, a = sp_signal.butter(1, 500.0 / (sample_rate / 2.0), btype="low")
    t = np.arange(n_samples) / sample_rate
    for _ in range(n_talkers):
        white = rng.standard_normal(n_samples)
        shaped = sp_signal.lfilter(b, a, white)
        rate = rng.uniform(2.0, 6.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t + phase))
        total += shaped * envelope
    total /= max(_rms(total), 1e-9)
    return total.astype(np.float32)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

EDGE_RAMP_MS = 10.0


def add_noise_span(
    pcm: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    snr_db: float,
    rng: np.random.Generator,
    noise_kind: str = "babble",
) -> np.ndarray:
    """Additive noise on the span at the given local SNR.

    SNR is computed against the RMS of the span itself, so `snr_db=0` means
    the masker is as loud as the speech it covers regardless of the TTS
    output level.
    """
    start, end = _span_to_samples(pcm, sample_rate, start_sec, end_sec)
    n = end - start
    if noise_kind == "babble":
        noise = make_babble(n, sample_rate, rng)
    elif noise_kind == "white":
        noise = rng.standard_normal(n).astype(np.float32)
        noise /= max(_rms(noise), 1e-9)
    else:
        raise ValueError(f"unknown noise_kind: {noise_kind}")

    speech_rms = _rms(pcm[start:end])
    if speech_rms <= 1e-9:
        # Span is silence (alignment failure upstream); adding noise scaled
        # to zero would silently produce a clean file. Fail loudly instead.
        raise ValueError("span RMS is ~0; check slot-span alignment")
    noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    ramp = int(EDGE_RAMP_MS / 1000.0 * sample_rate)
    out = pcm.copy()
    out[start:end] = out[start:end] + noise * noise_rms * _fade_window(n, ramp)
    return np.clip(out, -1.0, 1.0)


def mask_span(
    pcm: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    fill: str = "silence",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace the span outright (packet-loss / dropout style).

    fill="silence" zeroes the span; fill="noise" replaces it with
    speech-shaped noise at the span's original level (energetic masking with
    no residual glimpses of the target).
    """
    start, end = _span_to_samples(pcm, sample_rate, start_sec, end_sec)
    n = end - start
    ramp = int(EDGE_RAMP_MS / 1000.0 * sample_rate)
    win = _fade_window(n, ramp)
    out = pcm.copy()
    if fill == "silence":
        out[start:end] = out[start:end] * (1.0 - win)
    elif fill == "noise":
        if rng is None:
            raise ValueError("fill='noise' requires rng")
        level = max(_rms(pcm[start:end]), 1e-4)
        noise = make_babble(n, sample_rate, rng) * level
        out[start:end] = out[start:end] * (1.0 - win) + noise * win
    else:
        raise ValueError(f"unknown fill: {fill}")
    return np.clip(out, -1.0, 1.0)


def lowpass_span(
    pcm: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    cutoff_hz: float = 800.0,
) -> np.ndarray:
    """Severe low-pass on the span (muffled / far-from-mic speech)."""
    start, end = _span_to_samples(pcm, sample_rate, start_sec, end_sec)
    n = end - start
    sos = sp_signal.butter(6, cutoff_hz / (sample_rate / 2.0), btype="low",
                           output="sos")
    filtered = sp_signal.sosfiltfilt(sos, pcm[start:end].astype(np.float64))
    ramp = int(EDGE_RAMP_MS / 1000.0 * sample_rate)
    win = _fade_window(n, ramp)
    out = pcm.copy()
    out[start:end] = (pcm[start:end] * (1.0 - win)
                      + filtered.astype(np.float32) * win)
    return np.clip(out, -1.0, 1.0)


def add_noise_full(
    pcm: np.ndarray,
    sample_rate: int,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Whole-utterance additive babble (non-localized control condition)."""
    return add_noise_span(
        pcm, sample_rate, 0.0, len(pcm) / sample_rate, snr_db, rng
    )


# ---------------------------------------------------------------------------
# condition registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """A named corruption condition applied during benchmark construction."""

    name: str
    # (pcm, sr, span_start, span_end, rng) -> pcm
    apply: Callable[..., np.ndarray] = field(repr=False)
    # Ground-truth expectation used by decision metrics:
    #   "act"     -> the information survived; the model should proceed.
    #   "ask"     -> the information is destroyed; the model should clarify.
    #   "graded"  -> intermediate; scored via the ASR recoverability oracle.
    expected_behavior: str = "graded"


def _cond_clean(pcm, sr, s, e, rng):  # noqa: ANN001 - registry signature
    return pcm.copy()


def build_default_conditions() -> list[Condition]:
    """The condition grid for the main experiment.

    clean                : control; asking here is a false alarm.
    babble_snr{+5,0,-5,-10}: graded local degradation (calibration axis).
    mask_silence         : slot span fully dropped; asking is mandatory.
    mask_noise           : slot span replaced by babble; asking is mandatory.
    lowpass800           : muffled span; graded.
    full_snr0            : same masker energy but spread over the whole
                           utterance; contrasts localized vs global
                           degradation at matched difficulty.
    """
    conds: list[Condition] = [Condition("clean", _cond_clean, "act")]
    for snr in (5, 0, -5, -10):
        conds.append(
            Condition(
                f"babble_snr{snr:+d}",
                lambda pcm, sr, s, e, rng, _snr=snr: add_noise_span(
                    pcm, sr, s, e, float(_snr), rng
                ),
                "graded",
            )
        )
    conds.append(
        Condition(
            "mask_silence",
            lambda pcm, sr, s, e, rng: mask_span(pcm, sr, s, e, "silence"),
            "ask",
        )
    )
    conds.append(
        Condition(
            "mask_noise",
            lambda pcm, sr, s, e, rng: mask_span(pcm, sr, s, e, "noise", rng),
            "ask",
        )
    )
    conds.append(
        Condition(
            "lowpass800",
            lambda pcm, sr, s, e, rng: lowpass_span(pcm, sr, s, e, 800.0),
            "graded",
        )
    )
    conds.append(
        Condition(
            "full_snr0",
            lambda pcm, sr, s, e, rng: add_noise_full(pcm, sr, 0.0, rng),
            "graded",
        )
    )
    return conds


def get_condition(name: str) -> Condition:
    for cond in build_default_conditions():
        if cond.name == name:
            return cond
    raise KeyError(f"unknown condition: {name}")
