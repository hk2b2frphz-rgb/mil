#!/usr/bin/env python3
from __future__ import annotations
"""
Run a local cascade (ASR->LLM) or SpeechLLM (audio->LLM) baseline over an
already-built Full-Duplex-Bench-JA dataset (eval/build_full_duplex_ja_dataset.py),
writing run_full_duplex_bench.py-compatible trial directories so
evaluate_full_duplex_ja.py works unchanged:

    # Build the dataset once WITHOUT the Moshi-only opening greeting --
    # a cascade/SpeechLLM baseline was never trained to say it, and has no
    # streaming behavior to reserve lead-in time for anyway.
    uv run python eval/build_full_duplex_ja_dataset.py \
        --scenarios eval_sets/full_duplex_ja/scenarios_expanded.jsonl \
        --out-dir data/full_duplex_ja_v2_nogreeting --no-opening-greeting

    uv run python eval/run_local_baseline_full_duplex.py --system cascade \
        --dataset-dir data/full_duplex_ja_v2_nogreeting \
        --out-dir eval_runs/full_duplex/cascade_gemma2b/inference \
        --model-id cascade_gemma2b

    uv run python eval/evaluate_full_duplex_ja.py \
        --run-dir eval_runs/full_duplex/cascade_gemma2b/inference \
        --out-dir eval_runs/full_duplex/cascade_gemma2b/benchmark_results

**Read this before trusting the numbers.** Cascade/SpeechLLM baselines use a
streaming Silero VAD endpoint: 20-ms capture frames are observed in order and
ASR/LLM starts when the first end-of-utterance is detected. An 8-second silent
tail is fed after each recording so the endpoint is always observable. They
only ever produce ONE response and cannot react while generation is underway,
so:

- `pause_handling`/`smooth_turn_taking`/`user_interruption`: TOR and
  response-latency metrics will faithfully show these baselines never take
  over early and always respond late by roughly
  asr_wall + llm_wall + tts_wall seconds -- that IS the point of the
  comparison, not a bug.
- `backchannel`: cascade output is re-recognized with Japanese
  faster-whisper after TTS, so its chunks have audio-derived timestamps.
  `backchannel_frequency`/`jsd` still count *occurrences* of separate
  backchannel-like segments over time, and a turn-based system only ever
  produces one segment total (it cannot interject multiple times while the
  user keeps talking), so those metrics remain structurally not meaningful.
- `user_backchannel`/`talking_to_other`/`background_speech`: both noisy and
  clean inputs are processed with the same seed. The resulting paired metrics
  therefore measure how much the overlay changes the turn-based response.

The Azure/LLM-judge dimensions (contextual_relevance, safety_boundary,
etc.) via judge_full_duplex_azure.py / judge_llmjp_style.py stay fully
meaningful regardless -- they judge response content, not turn-taking.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from local_baseline_common import (  # noqa: E402
    COUNSELOR_SYSTEM_PROMPT,
    GemmaLLM,
    LocalASR,
    Qwen25Omni,
    SpeechLLM,
    init_baseline_tts,
    looks_truncated,
    validate_spoken_response,
)
from build_full_duplex_ja_dataset import synthesize  # noqa: E402
from full_duplex_audio import resample_linear, read_wav_mono, write_wav_mono, write_wav_stereo  # noqa: E402
from full_duplex_sample_selection import select_samples  # noqa: E402
from greeting_check import evaluate_opening_greeting  # noqa: E402
from streaming_vad import find_streaming_endpoint  # noqa: E402


FDB_V15_PROTOCOL_NAME = "Full-Duplex-Bench v1.5 static overlap protocol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local cascade or SpeechLLM baseline over a Full-Duplex-Bench-JA dataset."
    )
    parser.add_argument("--system", required=True, choices=["cascade", "speechllm", "qwen25_omni"])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or all.")
    parser.add_argument(
        "--cases-per-task",
        type=int,
        default=None,
        help="Deterministically evaluate only the first N cases of each selected task.",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-v15-profile",
        action="store_true",
        help=(
            "Reject a dataset that does not declare the repository's pinned "
            "Full-Duplex-Bench v1.5 static-overlap input profile. This checks "
            "the audio-input protocol only; this turn-based baseline cannot "
            "meet the full-duplex overlap-timing requirement."
        ),
    )

    parser.add_argument("--tts-backend",
                        choices=["qwen3", "kokoro", "pyopenjtalk", "auto"],
                        default="auto")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--tts-speaker", default="Serena")
    parser.add_argument("--tts-language", default="Japanese")
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")

    parser.add_argument(
        "--skip-output-alignment",
        action="store_true",
        help=(
            "Skip the second ASR pass over the synthesized output. The response "
            "text then comes from the LLM directly and output.json carries no "
            "timestamped chunks. Roughly a third of cascade wall time, at the "
            "cost of per-word timing (needed by metrics that read chunk times)."
        ),
    )
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument(
        "--vad-tail-sec", type=float, default=8.0,
        help="Zero-audio tail streamed after each recording to flush the VAD endpoint.",
    )
    parser.add_argument(
        "--vad-silence-ms", type=int, default=800,
        help="Observed silence required by streaming Silero VAD before inference starts.",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)

    # Keep this identical to the established dialogue-generation path.  This
    # public Gemma 4 checkpoint is not the gated Gemma 2/3 Hugging Face path.
    parser.add_argument("--llm-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--llm-python", default=None)
    parser.add_argument("--llm-device-map", default="auto")
    parser.add_argument("--llm-dtype", default="float16")
    parser.add_argument("--llm-task", default="any-to-any")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=200)
    parser.add_argument("--llm-timeout-sec", type=float, default=300.0)

    parser.add_argument("--speechllm-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--speechllm-python", default=None)
    parser.add_argument("--speechllm-device-map", default="auto")
    parser.add_argument("--speechllm-dtype", default="auto")
    parser.add_argument("--speechllm-max-new-tokens", type=int, default=200)
    parser.add_argument("--speechllm-timeout-sec", type=float, default=300.0)

    parser.add_argument("--qwen25-omni-model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--qwen25-omni-python", default=None)
    parser.add_argument("--qwen25-omni-device-map", default="auto")
    parser.add_argument("--qwen25-omni-dtype", choices=["auto", "float16", "float32"], default="float16")
    parser.add_argument("--qwen25-omni-speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--qwen25-omni-max-new-tokens", type=int, default=200)
    parser.add_argument("--qwen25-omni-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Do not run the unscored ASR/LLM/TTS warmup before evaluation.",
    )
    parser.set_defaults(warmup=True)

    return parser.parse_args()


def seed_local(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def warmup_baseline(
    args: argparse.Namespace,
    tts_backend: str,
    tts_engine: Any | None,
    asr: LocalASR | None,
    llm: GemmaLLM | None,
    speechllm: SpeechLLM | None,
    qwen25_omni: Qwen25Omni | None,
    pcm: np.ndarray,
    sample_rate: int,
) -> float:
    """Exercise first-request kernels without writing or scoring a trial."""
    print("[local-fdb] warming ASR/LLM/TTS; this request is excluded from metrics")
    started = time.perf_counter()
    warmup_text = "最近、少し疲れています。"
    tts_text = "お話しくださってありがとうございます。"
    find_streaming_endpoint(
        pcm, sample_rate, tail_sec=args.vad_tail_sec,
        silence_ms=args.vad_silence_ms, threshold=args.vad_threshold,
    )
    if args.system == "cascade":
        assert asr is not None and llm is not None
        asr.transcribe(pcm, sample_rate)
        llm.respond(warmup_text, COUNSELOR_SYSTEM_PROMPT, seed=0)
    elif args.system == "speechllm":
        assert speechllm is not None
        speechllm.respond(pcm, sample_rate, COUNSELOR_SYSTEM_PROMPT)
    else:
        assert qwen25_omni is not None
        qwen25_omni.respond(pcm, sample_rate, COUNSELOR_SYSTEM_PROMPT)

    if args.system != "qwen25_omni":
        synthesize(
            tts_text, sample_rate, args.tts_speed, tts_backend, tts_engine,
            args.tts_speaker, speaker_role="moshi",
        )
    elapsed = time.perf_counter() - started
    print(f"[local-fdb] warmup complete: {elapsed:.2f}s (unscored)")
    return elapsed


def process_variant(
    sample: dict[str, Any],
    seed: int,
    variant: str,
    dataset_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    tts_backend: str,
    tts_engine: Any | None,
    asr: LocalASR | None,
    llm: GemmaLLM | None,
    speechllm: SpeechLLM | None,
    qwen25_omni: Qwen25Omni | None,
    model_id: str,
) -> None:
    source_dir = dataset_dir / sample["path"]
    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "metadata.json", trial_dir / "metadata.json")

    input_wav = source_dir / f"{variant}.wav"
    shutil.copy2(input_wav, trial_dir / f"{variant}.wav")
    pcm, sample_rate = read_wav_mono(input_wav)
    input_duration_sec = len(pcm) / sample_rate
    output_stem = "clean_output" if variant == "clean_input" else "output"
    vad_started = time.perf_counter()
    endpoint = find_streaming_endpoint(
        pcm, sample_rate, tail_sec=args.vad_tail_sec,
        silence_ms=args.vad_silence_ms, threshold=args.vad_threshold,
    )
    vad_wall = time.perf_counter() - vad_started
    if endpoint is None:
        raise RuntimeError(
            "Streaming VAD did not observe an utterance end even after "
            f"{args.vad_tail_sec:g}s of silence. Increase --vad-tail-sec or inspect the input."
        )
    received_pcm = np.concatenate([
        pcm, np.zeros(int(round(args.vad_tail_sec * sample_rate)), dtype=np.float32),
    ])
    # This is all and only the audio available at the online endpoint. It
    # includes the silence that caused VAD to fire, never future fixture audio.
    inference_pcm = received_pcm[:endpoint.audio_end_sample]
    # VAD normally runs while the caller is still talking. Only work that
    # cannot keep up with the received stream delays the endpoint.
    vad_overrun = max(0.0, vad_wall - endpoint.detected_at_sec)
    # The end of actual speech, rather than the later moment at which enough
    # silence has arrived for VAD to confirm it.
    utterance_end_sec = (
        endpoint.speech_end_sec
        if endpoint.speech_end_sec is not None
        else endpoint.detected_at_sec
    )
    vad_complete_after_end = endpoint.detected_at_sec + vad_overrun - utterance_end_sec
    print(
        f"[local-fdb] VAD {sample['task']}/{sample['id']} {variant}: "
        f"endpoint={endpoint.detected_at_sec:.3f}s wall={vad_wall:.3f}s "
        f"overrun={vad_overrun:.3f}s complete_after_speech_end={vad_complete_after_end:.3f}s"
    )

    asr_text = None
    asr_wall = 0.0
    asr_complete_after_end = vad_complete_after_end
    if args.system == "cascade":
        assert asr is not None and llm is not None
        asr_text, asr_wall = asr.transcribe(inference_pcm, sample_rate)
        asr_complete_after_end = vad_complete_after_end + asr_wall
        print(
            f"[local-fdb] ASR complete {sample['task']}/{sample['id']} {variant}: "
            f"after_speech_end={asr_complete_after_end:.3f}s"
        )
        response_text, llm_wall = llm.respond(
            asr_text, COUNSELOR_SYSTEM_PROMPT, seed=seed
        )
    elif args.system == "speechllm":
        assert speechllm is not None
        response_text, llm_wall = speechllm.respond(inference_pcm, sample_rate, COUNSELOR_SYSTEM_PROMPT)
        response_pcm_native = None
        native_sample_rate = sample_rate
    else:
        assert qwen25_omni is not None
        response_text, response_pcm_native, native_sample_rate, llm_wall = qwen25_omni.respond(
            inference_pcm, sample_rate, COUNSELOR_SYSTEM_PROMPT
        )
    llm_complete_after_end = asr_complete_after_end + llm_wall
    print(
        f"[local-fdb] LLM complete {sample['task']}/{sample['id']} {variant}: "
        f"after_speech_end={llm_complete_after_end:.3f}s"
    )
    response_text = validate_spoken_response(response_text)

    tts_started = time.perf_counter()
    seed_local(seed)
    has_response = bool(response_text.strip())
    response_pcm = (
        resample_linear(response_pcm_native, native_sample_rate, sample_rate)
        if args.system == "qwen25_omni" and response_pcm_native is not None
        else
        synthesize(
            response_text,
            sample_rate,
            args.tts_speed,
            tts_backend,
            tts_engine,
            args.tts_speaker,
            speaker_role="moshi",
        )
        if has_response
        else np.zeros(1, dtype=np.float32)
    )
    tts_wall = 0.0 if args.system == "qwen25_omni" else time.perf_counter() - tts_started
    tts_complete_after_end = llm_complete_after_end + tts_wall
    print(
        f"[local-fdb] TTS complete {sample['task']}/{sample['id']} {variant}: "
        f"after_speech_end={tts_complete_after_end:.3f}s all_complete={tts_complete_after_end:.3f}s"
    )

    total_wall = asr_wall + llm_wall + tts_wall
    # Place the response at the real time it would actually start: after VAD
    # observes the endpoint AND all processing finishes. This makes
    # evaluate_full_duplex_ja.py's existing (unchanged) latency/TOR metrics,
    # which are derived from output.wav/output.json placement rather than
    # from output.meta.json, automatically reflect true cascade/SpeechLLM
    # latency instead of understating it.
    output_start_sec = endpoint.detected_at_sec + vad_overrun + total_wall
    print(
        f"[local-fdb] completion since speech end {sample['task']}/{sample['id']} {variant}: "
        f"VAD={vad_complete_after_end:.3f}s ASR={asr_complete_after_end:.3f}s "
        f"LLM={llm_complete_after_end:.3f}s TTS={tts_complete_after_end:.3f}s "
        f"all={tts_complete_after_end:.3f}s"
    )
    silence = np.zeros(int(round(output_start_sec * sample_rate)), dtype=np.float32)
    aligned = np.concatenate([silence, response_pcm])
    write_wav_mono(trial_dir / f"{output_stem}.wav", aligned, sample_rate)
    # Left = input speech, right = model output, for side-by-side listening
    # review; not used by any deterministic/LLM-judge metric.
    write_wav_stereo(
        trial_dir / f"{output_stem}_stereo.wav", pcm, aligned, sample_rate
    )

    alignment_text = response_text
    alignment_chunks: list[dict[str, Any]] = []
    output_asr_wall = 0.0
    if args.system == "cascade" and has_response and not args.skip_output_alignment:
        # Align the actual synthesized Japanese audio, not the LLM string.
        # The alignment input starts at zero, so shift every ASR time into the
        # full output.wav timeline after recognition.
        assert asr is not None
        alignment_text, alignment_chunks, output_asr_wall = asr.transcribe_aligned(
            response_pcm, sample_rate
        )
        for chunk in alignment_chunks:
            start, end = chunk["timestamp"]
            chunk["timestamp"] = [
                round(output_start_sec + float(start), 4),
                round(output_start_sec + float(end), 4),
            ]
    elif args.system == "cascade" and has_response:
        # --skip-output-alignment. 単語ごとの時刻は無いが、応答全体を覆う 1 つの
        # チャンクは置ける。相槌の種類判定は「その音声区間に重なるテキスト」を
        # 見るだけなので、これがあれば種類までは付く。時刻の精度が要る指標
        # (単語単位のアライメント)だけが失われる。
        alignment_chunks = [{
            "text": response_text,
            "timestamp": [
                round(output_start_sec, 4),
                round(output_start_sec + len(response_pcm) / sample_rate, 4),
            ],
        }]
    elif args.system in {"speechllm", "qwen25_omni"} and has_response:
        # Qwen2-Audio returns the generated text but not word alignment.  Keep
        # one response-wide timestamped chunk nevertheless: real-response
        # scoring and the LLM judge read output.json on the output.wav clock.
        # Leaving this empty made a perfectly audible SpeechLLM response look
        # textless to those downstream steps.
        alignment_chunks = [{
            "text": response_text,
            "timestamp": [
                round(output_start_sec, 4),
                round(output_start_sec + len(response_pcm) / sample_rate, 4),
            ],
        }]
    output_json = {
        "text": alignment_text,
        "chunks": alignment_chunks,
        "source": (
            f"{args.system}_asr_alignment"
            if args.system == "cascade" and not args.skip_output_alignment
            else f"{args.system}_generated_text"
        ),
        "generated_text": response_text,
        "language": "ja",
    }
    (trial_dir / f"{output_stem}.json").write_text(
        json.dumps(output_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    greeting = (
        evaluate_opening_greeting(metadata, output_json["chunks"])
        if variant == "input"
        else None
    )
    if greeting is not None:
        status = "OK" if greeting["matched"] else "MISMATCH"
        print(
            f"[cascade-eval] greeting {sample['task']}/{sample['id']} seed={seed}: "
            f"{status} similarity={greeting['similarity']}"
        )

    output_meta = {
        "model_id": model_id,
        "seed": seed,
        "task": sample["task"],
        "case_id": sample["id"],
        "input_duration_sec": round(input_duration_sec, 4),
        "vad_endpoint_detected_at_sec": round(endpoint.detected_at_sec, 4),
        "vad_speech_end_sec": round(endpoint.speech_end_sec, 4) if endpoint.speech_end_sec is not None else None,
        "vad_audio_sent_sec": round(len(inference_pcm) / sample_rate, 4),
        "vad_backend": endpoint.backend,
        "vad_wall_time_sec": round(vad_wall, 4),
        "vad_overrun_time_sec": round(vad_overrun, 4),
        "vad_silence_ms": args.vad_silence_ms,
        "vad_tail_sec": args.vad_tail_sec,
        "utterance_end_sec": round(utterance_end_sec, 4),
        "vad_complete_after_utterance_end_sec": round(vad_complete_after_end, 4),
        "asr_complete_after_utterance_end_sec": round(asr_complete_after_end, 4),
        "llm_complete_after_utterance_end_sec": round(llm_complete_after_end, 4),
        "tts_complete_after_utterance_end_sec": round(tts_complete_after_end, 4),
        "all_complete_after_utterance_end_sec": round(tts_complete_after_end, 4),
        "wall_time_sec": round(total_wall, 4),
        "language": "ja",
        "expected_behavior": metadata.get("expected_behavior"),
        "system": args.system,
        "variant": variant,
        # 言い切らずに終わっている = max_new_tokens で切られた疑い。断片が
        # そのまま音声になるので、件数を追えるようにしておく。
        "response_looks_truncated": looks_truncated(response_text),
        "response_chars": len(response_text),
        "asr_transcript": asr_text,
        "asr_wall_time_sec": round(asr_wall, 4),
        "output_asr_wall_time_sec": round(output_asr_wall, 4),
        "llm_wall_time_sec": round(llm_wall, 4),
        "tts_wall_time_sec": round(tts_wall, 4),
        "audible_response_start_sec": round(output_start_sec, 4),
        "tts_backend": tts_backend,
        "tts_model": args.tts_model,
        "tts_speaker": args.tts_speaker,
        "tts_dtype": args.dtype,
    }
    (trial_dir / f"{output_stem}.meta.json").write_text(
        json.dumps(output_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def process_sample(
    sample: dict[str, Any],
    seed: int,
    dataset_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    tts_backend: str,
    tts_engine: Any | None,
    asr: LocalASR | None,
    llm: GemmaLLM | None,
    speechllm: SpeechLLM | None,
    qwen25_omni: Qwen25Omni | None,
    model_id: str,
) -> None:
    source_dir = dataset_dir / sample["path"]
    variants = ["input"]
    if (source_dir / "clean_input.wav").exists():
        variants.append("clean_input")
    for variant in variants:
        process_variant(
            sample,
            seed,
            variant,
            dataset_dir,
            out_dir,
            args,
            tts_backend,
            tts_engine,
            asr,
            llm,
            speechllm,
            qwen25_omni,
            model_id,
        )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    selected_tasks = None if args.tasks == "all" else {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    protocol = manifest.get("protocol")
    if args.require_v15_profile:
        actual_protocol = protocol.get("name") if isinstance(protocol, dict) else None
        if actual_protocol != FDB_V15_PROTOCOL_NAME:
            raise SystemExit(
                "Dataset does not declare the required Full-Duplex-Bench v1.5 "
                f"input profile: expected {FDB_V15_PROTOCOL_NAME!r}, got "
                f"{actual_protocol!r}. Rebuild with "
                "eval/build_full_duplex_ja_dataset.py."
            )
    try:
        samples = select_samples(
            manifest["samples"], selected_tasks, args.cases_per_task
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not samples:
        raise SystemExit("No benchmark samples matched --tasks.")

    if args.system == "qwen25_omni":
        tts_backend, tts_engine = "qwen25_omni_native", None
    else:
        tts_backend, tts_engine = init_baseline_tts(
            args.tts_backend, args.tts_model, args.tts_speaker, args.tts_language,
            args.device, args.dtype,
        )

    asr = llm = speechllm = qwen25_omni = None
    if args.system == "cascade":
        asr = LocalASR(args.asr_model, args.asr_device, args.asr_compute_type)
        llm = GemmaLLM(
            args.llm_model, args.llm_python, args.llm_device_map, args.llm_dtype,
            args.llm_task,
            args.llm_temperature, args.llm_top_p, args.llm_max_new_tokens,
            args.llm_timeout_sec,
        )
        default_model_id = f"cascade_{args.asr_model}_{args.llm_model.split('/')[-1]}"
    elif args.system == "speechllm":
        speechllm = SpeechLLM(
            args.speechllm_model, args.speechllm_python, args.speechllm_device_map,
            args.speechllm_dtype, args.speechllm_max_new_tokens, args.speechllm_timeout_sec,
        )
        default_model_id = f"speechllm_{args.speechllm_model.split('/')[-1]}"
    else:
        qwen25_omni = Qwen25Omni(
            args.qwen25_omni_model, args.qwen25_omni_python,
            args.qwen25_omni_device_map, args.qwen25_omni_dtype,
            args.qwen25_omni_speaker, args.qwen25_omni_max_new_tokens,
            args.qwen25_omni_timeout_sec,
        )
        default_model_id = f"qwen25_omni_{args.qwen25_omni_model.split('/')[-1]}"
    model_id = args.model_id or default_model_id

    if asr is not None:
        asr.load()
    if llm is not None:
        llm.load()
    if speechllm is not None:
        speechllm.load()
    if qwen25_omni is not None:
        qwen25_omni.load()

    warmup_wall_time_sec = 0.0
    if args.warmup:
        warmup_pcm, warmup_sample_rate = read_wav_mono(
            dataset_dir / samples[0]["path"] / "input.wav"
        )
        warmup_wall_time_sec = warmup_baseline(
            args, tts_backend, tts_engine, asr, llm, speechllm, qwen25_omni,
            warmup_pcm, warmup_sample_rate,
        )

    run_config = {
        "model_id": model_id,
        "system": args.system,
        "git_commit": os.environ.get("FDB_GIT_COMMIT"),
        "seeds": seeds,
        "tasks": sorted(selected_tasks) if selected_tasks else "all",
        "cases_per_task": args.cases_per_task,
        "selected_case_count": len(samples),
        "upstream_full_duplex_bench_commit": manifest.get("upstream_commit"),
        "profile": manifest.get("profile"),
        "protocol": protocol,
        "protocol_conformance": {
            "input_profile": "verified" if args.require_v15_profile else "not_required",
            "overlap_timing": "streaming_vad_turn_based_baseline",
        },
        "asr_model": args.asr_model if args.system == "cascade" else None,
        "asr_compute_type": args.asr_compute_type if args.system == "cascade" else None,
        "vad_backend": "silero_streaming_20ms_capture",
        "vad_silence_ms": args.vad_silence_ms,
        "vad_tail_sec": args.vad_tail_sec,
        "vad_threshold": args.vad_threshold,
        "llm_model": args.llm_model if args.system == "cascade" else None,
        "llm_dtype": args.llm_dtype if args.system == "cascade" else None,
        "llm_temperature": args.llm_temperature if args.system == "cascade" else None,
        "llm_top_p": args.llm_top_p if args.system == "cascade" else None,
        "llm_max_new_tokens": args.llm_max_new_tokens if args.system == "cascade" else None,
        "warmup_enabled": args.warmup,
        "warmup_wall_time_sec": round(warmup_wall_time_sec, 4),
        "llm_startup_wall_time_sec": (
            round(llm.startup_wall_time_sec, 4) if llm is not None else None
        ),
        "speechllm_model": args.speechllm_model if speechllm is not None else None,
        "speechllm_startup_wall_time_sec": (
            round(speechllm.startup_wall_time_sec, 4) if speechllm is not None else None
        ),
        "qwen25_omni_model": args.qwen25_omni_model if qwen25_omni is not None else None,
        "qwen25_omni_startup_wall_time_sec": (
            round(qwen25_omni.startup_wall_time_sec, 4) if qwen25_omni is not None else None
        ),
        "tts_backend_requested": args.tts_backend,
        "tts_backend_effective": tts_backend,
        "tts_model": args.tts_model,
        "tts_speaker": args.tts_speaker,
        "tts_dtype": args.dtype,
        "note": (
            "Streaming-VAD turn-based local baseline, not full-duplex. See this "
            "script's module docstring for which metrics are/aren't meaningful."
        ),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total = len(samples) * len(seeds)
    completed = 0
    try:
        for sample in samples:
            for seed in seeds:
                trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
                expected_meta = [trial_dir / "output.meta.json"]
                if (dataset_dir / sample["path"] / "clean_input.wav").exists():
                    expected_meta.append(trial_dir / "clean_output.meta.json")
                if all(path.exists() for path in expected_meta) and not args.overwrite:
                    print(f"[local-fdb] skip existing {trial_dir}")
                    completed += 1
                    continue
                process_sample(
                    sample, seed, dataset_dir, out_dir, args, tts_backend, tts_engine,
                    asr, llm, speechllm, qwen25_omni, model_id,
                )
                completed += 1
                print(f"[local-fdb] {completed}/{total} {sample['task']}/{sample['id']} seed={seed}")
    finally:
        if llm is not None:
            llm.close()
        if speechllm is not None:
            speechllm.close()
        if qwen25_omni is not None:
            qwen25_omni.close()

    print(f"[local-fdb] inference complete -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
