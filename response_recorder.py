#!/usr/bin/env python3
from __future__ import annotations
"""
response_recorder.py
Batch experiment tool: feed fixed audio files to Moshi and record responses.

Each trial = one (input_file, seed) pair.
Outputs: response.wav, transcript.jsonl, transcript.txt, meta.json per trial.
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def _seed_all_fallback(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


try:
    from moshi.run_inference import seed_all  # type: ignore[import]
except ImportError:
    seed_all = _seed_all_fallback  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_wav_mono(path: str | Path, target_sr: int) -> np.ndarray:
    """Load WAV, convert to mono float32, and resample to target_sr."""
    import sphn  # type: ignore[import]
    import torch
    import torchaudio  # type: ignore[import]

    pcm, sr = sphn.read(str(path))  # (channels, samples) float32

    if pcm.ndim == 1:
        pcm = pcm[None, :]  # ensure (channels, samples)

    # Average channels -> mono (1, samples)
    pcm_mono = pcm.mean(axis=0, keepdims=True)

    # Resample if necessary
    if sr != target_sr:
        t = torch.from_numpy(pcm_mono)
        t = torchaudio.functional.resample(t, sr, target_sr)
        pcm_mono = t.numpy()

    return pcm_mono[0].astype(np.float32)  # (samples,)


def get_input_duration_sec(path: str | Path) -> float:
    """Return duration in seconds of a WAV file."""
    try:
        import sphn  # type: ignore[import]
        pcm, sr = sphn.read(str(path))
        return pcm.shape[-1] / sr
    except Exception:
        return 0.0


def collect_input_files(inputs: list[str]) -> list[Path]:
    """Expand file/directory arguments into a list of .wav paths."""
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")))
        elif p.is_file():
            files.append(p)
        else:
            logger.warning("Input path not found, skipping: %s", inp)
    return files


def collect_text_prompts(args: argparse.Namespace) -> list[str]:
    """Collect prompt text from --texts and --text-file."""
    prompts: list[str] = []
    if args.texts:
        prompts.extend(args.texts)
    if args.text_file:
        text_path = Path(args.text_file)
        with open(text_path, "r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    should_read_stdin = args.stdin or (
        not args.inputs and not args.texts and not args.text_file
        and not sys.stdin.isatty()
    )
    if should_read_stdin:
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            prompts.append(stdin_text)
    return prompts


def _safe_prompt_stem(text: str, index: int) -> str:
    stem = re.sub(r"\s+", "_", text.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)
    stem = stem[:40].strip("._")
    return f"text_{index:03d}_{stem}" if stem else f"text_{index:03d}"


def synthesize_text_inputs(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    """Synthesize text prompts to WAV files and return their paths."""
    prompts = collect_text_prompts(args)
    if not prompts:
        return []

    tts_dir = out_dir / "_tts_inputs"
    tts_dir.mkdir(parents=True, exist_ok=True)

    wav_paths: list[Path] = []
    for index, text in enumerate(prompts, start=1):
        wav_path = tts_dir / f"{_safe_prompt_stem(text, index)}.wav"
        logger.info("Synthesizing text prompt %d/%d: %s", index, len(prompts), text)
        synthesize_text_to_wav(
            text=text,
            wav_path=wav_path,
            voice=args.tts_voice,
            rate=args.tts_rate,
        )
        wav_paths.append(wav_path)
    return wav_paths


def synthesize_text_to_wav(
    text: str,
    wav_path: Path,
    voice: Optional[str],
    rate: int,
) -> None:
    """Create a WAV prompt using local TTS backends."""
    if synthesize_with_pyopenjtalk(text, wav_path):
        return

    try:
        import pyttsx3  # type: ignore[import]

        engine = pyttsx3.init()
        if voice:
            for candidate in engine.getProperty("voices"):
                if voice.lower() in candidate.name.lower():
                    engine.setProperty("voice", candidate.id)
                    break
            else:
                logger.warning("TTS voice not found in pyttsx3: %s", voice)
        engine.setProperty("rate", rate)
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()
        if wav_path.exists() and wav_path.stat().st_size > 0:
            return
    except Exception as exc:
        logger.info("pyttsx3 TTS unavailable, trying command-line TTS: %s", exc)

    if synthesize_with_cli_tts(text, wav_path, voice, rate):
        return

    if os.name != "nt":
        raise RuntimeError(
            "Text-to-speech requires a local backend: pyopenjtalk, pyttsx3, "
            "espeak-ng/espeak/pico2wave, or Windows System.Speech. "
            "Run `uv sync` to install pyopenjtalk, or provide WAV files with "
            "--inputs."
        )

    env = os.environ.copy()
    env["MOSHI_TTS_TEXT"] = text
    env["MOSHI_TTS_PATH"] = str(wav_path)
    env["MOSHI_TTS_RATE"] = str(max(-10, min(10, int((rate - 200) / 20))))
    env["MOSHI_TTS_VOLUME"] = "100"
    if voice:
        env["MOSHI_TTS_VOICE"] = voice

    script = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:MOSHI_TTS_VOICE) {
    $synth.SelectVoice($env:MOSHI_TTS_VOICE)
}
$synth.Rate = [int]$env:MOSHI_TTS_RATE
$synth.Volume = [int]$env:MOSHI_TTS_VOLUME
$synth.SetOutputToWaveFile($env:MOSHI_TTS_PATH)
$synth.Speak($env:MOSHI_TTS_TEXT)
$synth.Dispose()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        env=env,
    )


def synthesize_with_pyopenjtalk(text: str, wav_path: Path) -> bool:
    """Use pyopenjtalk for local Japanese TTS."""
    try:
        import pyopenjtalk  # type: ignore[import]
        import sphn  # type: ignore[import]

        pcm, sample_rate = pyopenjtalk.tts(text)
        pcm = np.asarray(pcm, dtype=np.float32)
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        if peak > 1.0:
            pcm = pcm / peak
        sphn.write_wav(str(wav_path), pcm, int(sample_rate))
        if wav_path.exists() and wav_path.stat().st_size > 0:
            logger.info("Synthesized prompt with pyopenjtalk")
            return True
    except Exception as exc:
        logger.warning("pyopenjtalk TTS failed: %s", exc)
    return False


def synthesize_with_cli_tts(
    text: str,
    wav_path: Path,
    voice: Optional[str],
    rate: int,
) -> bool:
    """Try common command-line TTS engines available on Linux servers."""
    engines = [
        shutil.which("espeak-ng"),
        shutil.which("espeak"),
        shutil.which("pico2wave"),
    ]
    for engine in [path for path in engines if path]:
        try:
            name = Path(engine).name.lower()
            if name in {"espeak-ng", "espeak"}:
                command = [engine, "-w", str(wav_path), "-s", str(rate)]
                if voice:
                    command.extend(["-v", voice])
                command.append(text)
            else:
                command = [engine, "-w", str(wav_path), text]
            subprocess.run(command, check=True)
            if wav_path.exists() and wav_path.stat().st_size > 0:
                logger.info("Synthesized prompt with %s", name)
                return True
        except Exception as exc:
            logger.warning("Command-line TTS failed (%s): %s", engine, exc)
    return False


# ---------------------------------------------------------------------------
# Model loading (called once)
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace):
    """
    Load and return: mimi, text_tokenizer, lm_gen, lm_gen_cfg, acoustic_delay.
    Models are loaded once and reused across all trials.
    """
    import torch
    from moshi.models import loaders  # type: ignore[import]

    dtype = torch.float16 if args.half else torch.bfloat16

    logger.info("Loading checkpoint from %s ...", args.hf_repo)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        tokenizer=args.tokenizer,
        config=args.config,
    )

    logger.info("Loading Mimi ...")
    mimi = checkpoint_info.get_mimi(device=args.device)
    mimi.eval()

    logger.info("Loading text tokenizer ...")
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    logger.info("Loading Moshi LM (dtype=%s, device=%s) ...", dtype, args.device)
    lm_gen = checkpoint_info.get_moshi(device=args.device, dtype=dtype)
    lm_gen.eval()

    # Get lm_gen config and apply CLI overrides
    lm_gen_cfg = checkpoint_info.lm_gen_config

    if args.temp is not None:
        _try_setattr(lm_gen_cfg, "temp", args.temp)
        _try_setattr(lm_gen, "temp", args.temp)
    if args.temp_text is not None:
        _try_setattr(lm_gen_cfg, "temp_text", args.temp_text)
        _try_setattr(lm_gen, "temp_text", args.temp_text)
    if args.cfg_coef is not None:
        _try_setattr(lm_gen_cfg, "cfg_coef", args.cfg_coef)
        _try_setattr(lm_gen, "cfg_coef", args.cfg_coef)

    # Determine acoustic delay from model config, falling back to 2
    acoustic_delay = _getattr_chain(lm_gen_cfg, "acoustic_delay",
                                    _getattr_chain(lm_gen, "acoustic_delay", 2))
    acoustic_delay = int(acoustic_delay)
    logger.info("Acoustic delay: %d frames", acoustic_delay)

    return mimi, text_tokenizer, lm_gen, lm_gen_cfg, acoustic_delay


def _try_setattr(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _getattr_chain(obj, name: str, default):
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

def run_trial(
    pcm: np.ndarray,
    seed: int,
    lm_gen,
    mimi,
    text_tokenizer,
    device: str,
    silence_sec: float,
    max_gen_sec: float,
    acoustic_delay: int,
) -> dict:
    """
    Run a single inference trial.

    Appends `silence_sec` of zeros after `pcm`, then feeds the concatenated
    signal frame-by-frame into mimi → lm_gen.  Stops at `max_gen_sec` or
    when all frames are consumed, whichever comes first.

    Returns a dict:
      audio_frames   : list of (num_codebooks, 1) cpu tensors
      text_events    : list of {"step", "time_sec", "piece"} dicts
      total_steps    : int
      first_response_step : int or None
    """
    import torch

    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(sample_rate / frame_rate)

    # ---- Build full PCM: input + silence --------------------------------
    silence_samples = int(silence_sec * sample_rate)
    full_pcm = np.concatenate(
        [pcm, np.zeros(silence_samples, dtype=np.float32)]
    )

    # ---- Determine number of steps to run --------------------------------
    max_steps = int(max_gen_sec * frame_rate)
    total_frames = (len(full_pcm) + frame_size - 1) // frame_size
    n_steps = min(total_frames, max_steps)

    # ---- Fix seed --------------------------------------------------------
    seed_all(seed)

    audio_frames: list[torch.Tensor] = []   # each: (num_codebooks, 1) cpu
    text_events: list[dict] = []
    first_response_step: Optional[int] = None

    # ---- Streaming inference loop ----------------------------------------
    with torch.no_grad():
        with lm_gen.streaming(1):
            with mimi.streaming(1):
                for step in range(n_steps):
                    start = step * frame_size
                    chunk_np = full_pcm[start: start + frame_size]

                    # Pad last chunk if shorter than frame_size
                    if len(chunk_np) < frame_size:
                        chunk_np = np.pad(
                            chunk_np, (0, frame_size - len(chunk_np))
                        )

                    # (1, 1, frame_size)
                    chunk = (
                        torch.from_numpy(chunk_np)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

                    # Mimi streaming encode -> (1, num_codebooks, 1) or None
                    codes = mimi.encode(chunk)
                    if codes is None:
                        continue

                    # LM step -> (1, 1+num_codebooks, 1) or None during warmup
                    out = lm_gen.step(codes)
                    if out is None:
                        continue

                    # Split text token and audio tokens
                    text_id = int(out[0, 0, 0].item())
                    audio_tok = out[0, 1:, :].cpu()  # (num_codebooks, 1)
                    audio_frames.append(audio_tok)

                    # Decode text token; skip padding ids 0 and 3
                    if text_id not in (0, 3):
                        piece = text_tokenizer.id_to_piece(text_id)
                        piece = piece.replace("\u2581", " ")  # ▁ -> space
                        time_sec = round(step / frame_rate, 4)
                        text_events.append(
                            {"step": step, "time_sec": time_sec, "piece": piece}
                        )
                        if first_response_step is None:
                            first_response_step = step

    return {
        "audio_frames": audio_frames,
        "text_events": text_events,
        "total_steps": n_steps,
        "first_response_step": first_response_step,
    }


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------

def decode_audio(
    audio_frames: list[torch.Tensor],
    acoustic_delay: int,
    mimi,
    device: str,
) -> Optional[np.ndarray]:
    """
    Stack audio token frames, apply acoustic delay, and decode to PCM.

    Returns (samples,) float32 numpy array, or None if there are not enough
    frames after applying the delay.
    """
    import torch

    if len(audio_frames) <= acoustic_delay:
        logger.warning(
            "Not enough audio frames (%d) after acoustic delay (%d); "
            "no audio will be saved.",
            len(audio_frames),
            acoustic_delay,
        )
        return None

    # Stack: (num_codebooks, total_frames)
    tokens = torch.cat(audio_frames, dim=1)

    # Apply acoustic delay by discarding the first `acoustic_delay` frames
    tokens = tokens[:, acoustic_delay:]

    # Decode: (1, num_codebooks, time) -> (1, 1, samples)
    tokens = tokens.unsqueeze(0).to(device)
    with torch.no_grad():
        pcm_out = mimi.decode(tokens)

    return pcm_out[0, 0].cpu().float().numpy()  # (samples,)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_trial_outputs(
    out_dir: Path,
    trial_result: dict,
    input_path: Path,
    seed: int,
    args: argparse.Namespace,
    mimi,
    acoustic_delay: int,
    wall_time: float,
) -> None:
    """Write response.wav, transcript.jsonl, transcript.txt, meta.json."""
    import sphn  # type: ignore[import]

    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)

    trial_dir = out_dir / input_path.stem / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    audio_frames = trial_result["audio_frames"]
    text_events = trial_result["text_events"]
    total_steps = trial_result["total_steps"]
    first_response_step = trial_result["first_response_step"]

    # ---- Audio -----------------------------------------------------------
    pcm_out = decode_audio(audio_frames, acoustic_delay, mimi, args.device)
    output_audio_sec = 0.0
    if pcm_out is not None and len(pcm_out) > 0:
        if args.response_sec and args.response_sec > 0:
            max_samples = int(args.response_sec * sample_rate)
            pcm_out = pcm_out[:max_samples]
        output_audio_sec = len(pcm_out) / sample_rate
        sphn.write_wav(str(trial_dir / "response.wav"), pcm_out, sample_rate)
    else:
        logger.warning("No audio output for input=%s seed=%d", input_path.stem, seed)

    # ---- transcript.jsonl ------------------------------------------------
    with open(trial_dir / "transcript.jsonl", "w", encoding="utf-8") as f:
        for event in text_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ---- transcript.txt --------------------------------------------------
    transcript = "".join(e["piece"] for e in text_events)
    with open(trial_dir / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    # ---- meta.json -------------------------------------------------------
    input_duration_sec = get_input_duration_sec(input_path)

    first_response_latency_sec = (
        first_response_step / frame_rate
        if first_response_step is not None
        else None
    )

    # Resolve effective temperature/cfg values from args or defaults
    temp = args.temp if args.temp is not None else float("nan")
    temp_text = args.temp_text if args.temp_text is not None else float("nan")
    cfg_coef = args.cfg_coef if args.cfg_coef is not None else float("nan")

    dtype_str = "float16" if args.half else "bfloat16"

    meta = {
        "input_path": str(input_path.resolve()),
        "input_duration_sec": round(input_duration_sec, 4),
        "silence_sec": args.silence_sec,
        "seed": seed,
        "model_repo": args.hf_repo,
        "dtype": dtype_str,
        "device": args.device,
        "temp": temp,
        "temp_text": temp_text,
        "cfg_coef": cfg_coef,
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "total_steps": total_steps,
        "first_response_step": first_response_step,
        "first_response_latency_sec": first_response_latency_sec,
        "wall_time_sec": round(wall_time, 3),
        "output_audio_sec": round(output_audio_sec, 4),
    }
    with open(trial_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch experiment tool: feed fixed audio inputs to Moshi "
            "and record responses (audio + text)."
        )
    )

    # Model selection
    parser.add_argument(
        "--hf-repo",
        default="llm-jp/llm-jp-moshi-v1",
        help="HuggingFace repo for the model checkpoint. "
             "Default: llm-jp/llm-jp-moshi-v1",
    )
    parser.add_argument(
        "--moshi-weight",
        default=None,
        metavar="PATH",
        help="Path to local Moshi model weights (overrides HF download).",
    )
    parser.add_argument(
        "--mimi-weight",
        default=None,
        metavar="PATH",
        help="Path to local Mimi codec weights (overrides HF download).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        metavar="PATH",
        help="Path to local SentencePiece tokenizer (overrides HF download).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to local model config file (overrides HF download).",
    )

    # Inputs
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Input WAV files and/or directories containing WAV files. "
             "Directories are expanded to all *.wav files inside.",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[],
        metavar="TEXT",
        help="Text prompts to synthesize into temporary WAV inputs.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        metavar="PATH",
        help="UTF-8 text file with one prompt per line. Each line is "
             "synthesized into a temporary WAV input.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one text prompt from standard input, synthesize it to WAV, "
             "and feed it to Moshi. If no --inputs/--texts/--text-file are "
             "provided and stdin is piped, this is enabled automatically.",
    )
    parser.add_argument(
        "--tts-voice",
        default=None,
        metavar="NAME",
        help="Optional TTS voice name. On Windows, use an installed "
             "System.Speech voice such as a Japanese voice if available.",
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        default=200,
        metavar="WPM",
        help="TTS speaking rate for pyttsx3. Default: 200. Windows SAPI "
             "maps this approximately to its -10..10 rate range.",
    )

    # Experiment parameters
    parser.add_argument(
        "--silence-sec",
        type=float,
        default=15.0,
        metavar="SEC",
        help="Seconds of zero-padded silence appended after each input audio "
             "to let Moshi generate a response. Default: 15.0",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        metavar="S1,S2,...",
        help='Comma-separated list of random seeds, e.g. "0,1,2". '
             "Takes precedence over --num-trials.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        metavar="N",
        help="Number of trials with seeds 0..N-1. Used only when --seeds is "
             "not specified.",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="Root output directory.",
    )

    # Device / dtype
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device. Default: cuda",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use float16. Without this flag, bfloat16 is used (recommended).",
    )

    # Sampling parameters
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature for audio tokens. "
             "Uses the model's default when not specified.",
    )
    parser.add_argument(
        "--temp-text",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature for text tokens. "
             "Uses the model's default when not specified.",
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=None,
        metavar="C",
        help="Classifier-free guidance coefficient. "
             "Uses the model's default when not specified.",
    )
    parser.add_argument(
        "--max-gen-sec",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Maximum generation time per trial in seconds (safety cap). "
             "Default: 60.0",
    )
    parser.add_argument(
        "--response-sec",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Maximum seconds of response.wav to save. Default: 10.0. "
             "Use 0 or a negative value to save the full decoded response.",
    )
    parser.add_argument(
        "--no-print-transcript",
        action="store_true",
        help="Do not print the generated transcript/ASR-style result.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect input files ------------------------------------------------
    input_files = collect_input_files(args.inputs)
    input_files.extend(synthesize_text_inputs(args, out_dir))
    if not input_files:
        logger.error("No input WAV files or text prompts found. Exiting.")
        return
    logger.info("Found %d input file(s).", len(input_files))

    # ---- Parse seeds --------------------------------------------------------
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.num_trials is not None:
        seeds = list(range(args.num_trials))
    else:
        seeds = [0]
    logger.info("Seeds: %s", seeds)

    # ---- Load models (once) -------------------------------------------------
    mimi, text_tokenizer, lm_gen, lm_gen_cfg, acoustic_delay = load_models(args)

    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)

    # ---- Save run-level metadata --------------------------------------------
    run_meta = {
        "model_repo": args.hf_repo,
        "datetime": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "acoustic_delay": acoustic_delay,
        "lm_gen_config": {
            "temp": _getattr_chain(lm_gen_cfg, "temp", None),
            "temp_text": _getattr_chain(lm_gen_cfg, "temp_text", None),
            "cfg_coef": _getattr_chain(lm_gen_cfg, "cfg_coef", None),
        },
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    logger.info("Wrote run_metadata.json")

    # ---- Experiment loop: input_file x seed ---------------------------------
    total_trials = len(input_files) * len(seeds)
    completed = 0
    skipped = 0

    for input_path in input_files:
        # Load and resample once per input file (not per seed)
        try:
            logger.info("Loading input: %s", input_path)
            pcm = load_wav_mono(str(input_path), sample_rate)
            input_duration_sec = len(pcm) / sample_rate
            logger.info(
                "  %.2f s / %d samples at %d Hz",
                input_duration_sec,
                len(pcm),
                sample_rate,
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s — skipping all seeds.", input_path, exc)
            skipped += len(seeds)
            continue

        for seed in seeds:
            trial_num = completed + skipped + 1
            logger.info(
                "[%d/%d] input=%s  seed=%d",
                trial_num,
                total_trials,
                input_path.stem,
                seed,
            )
            t0 = time.time()
            try:
                result = run_trial(
                    pcm=pcm,
                    seed=seed,
                    lm_gen=lm_gen,
                    mimi=mimi,
                    text_tokenizer=text_tokenizer,
                    device=args.device,
                    silence_sec=args.silence_sec,
                    max_gen_sec=args.max_gen_sec,
                    acoustic_delay=acoustic_delay,
                )
                wall_time = time.time() - t0

                save_trial_outputs(
                    out_dir=out_dir,
                    trial_result=result,
                    input_path=input_path,
                    seed=seed,
                    args=args,
                    mimi=mimi,
                    acoustic_delay=acoustic_delay,
                    wall_time=wall_time,
                )

                first_step = result["first_response_step"]
                latency_str = (
                    f"{first_step / frame_rate:.2f}s"
                    if first_step is not None
                    else "N/A"
                )
                transcript = "".join(
                    event["piece"] for event in result["text_events"]
                ).strip()
                if not args.no_print_transcript:
                    print(
                        f"ASR result [{input_path.stem} seed={seed}]: "
                        f"{transcript or '(empty)'}"
                    )
                logger.info(
                    "  done: %d steps | latency=%s | text_tokens=%d | wall=%.1fs",
                    result["total_steps"],
                    latency_str,
                    len(result["text_events"]),
                    wall_time,
                )
                completed += 1

            except Exception as exc:
                wall_time = time.time() - t0
                logger.error(
                    "  FAILED (input=%s, seed=%d, wall=%.1fs): %s",
                    input_path.stem,
                    seed,
                    wall_time,
                    exc,
                    exc_info=True,
                )
                skipped += 1

    logger.info(
        "All done. %d/%d trials completed, %d skipped.",
        completed,
        total_trials,
        skipped,
    )


if __name__ == "__main__":
    main()
