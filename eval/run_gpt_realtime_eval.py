#!/usr/bin/env python3
"""Evaluate GPT Realtime against the real-dialogue audio test set.

This program is deliberately local-only.  It opens a WebSocket to the Azure
OpenAI (or OpenAI) Realtime API, so it refuses PBS/Slurm/LSF environments rather
than silently failing after a costly scheduler allocation.  Each test case
receives a fresh Realtime session; therefore no conversation context can leak
between cases.

Credentials come from eval/openai_judge_common.py, the same place the Azure
judge reads them, so one AZURE_OPENAI_* environment configures both halves of
the evaluation.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from full_duplex_audio import read_wav_mono, resample_linear, write_wav_mono, write_wav_stereo
from full_duplex_sample_selection import select_samples
from local_baseline_common import COUNSELOR_SYSTEM_PROMPT
from openai_judge_common import load_realtime_connection, realtime_proxy_options, resolve_provider

SCHEDULER_VARIABLES = ("PBS_JOBID", "SLURM_JOB_ID", "LSB_JOBID")
PCM_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="gpt-realtime")
    parser.add_argument("--provider", choices=["auto", "azure", "openai"], default="auto",
                        help="auto picks openai when OPENAI_BASE_URL is set, azure when AZURE_OPENAI_ENDPOINT is.")
    # Unset by default: the deployment/model name comes from the environment,
    # exactly as it does for the judge, and a hard-coded "gpt-realtime" here
    # would silently override a correctly configured one.
    parser.add_argument("--model", help="Azure realtime deployment name, or realtime model id on an OpenAI-compatible endpoint.")
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--instructions", default=COUNSELOR_SYSTEM_PROMPT)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--cases-per-task", type=int)
    parser.add_argument("--seeds", default="0", help="Retained for result-layout compatibility.")
    parser.add_argument("--max-output-tokens", type=int, default=200)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--response-timeout-sec", type=float, default=90.0)
    parser.add_argument("--input-mode", choices=["realtime", "fast"], default="realtime")
    parser.add_argument("--turn-detection", choices=["server_vad", "manual"], default="server_vad")
    parser.add_argument(
        "--realtime-schema",
        choices=["ga", "legacy"],
        default=os.environ.get("OPENAI_REALTIME_SCHEMA", "ga"),
        help="session.update dialect. 'legacy' is the beta schema an in-house gateway may still speak.",
    )
    # A refused or dropped WebSocket is normal against a corporate gateway, and
    # losing a whole 49-case run to one of them is not worth the API spend.
    parser.add_argument("--connect-retries", type=int,
                        default=int(os.environ.get("GPT_REALTIME_CONNECT_RETRIES", "4")))
    parser.add_argument("--case-retries", type=int,
                        default=int(os.environ.get("GPT_REALTIME_CASE_RETRIES", "3")))
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_server() -> None:
    seen = [key for key in SCHEDULER_VARIABLES if os.environ.get(key)]
    if seen:
        raise SystemExit(
            "GPT Realtime evaluation is local-only and must not run through PBS/Slurm/LSF "
            f"(detected {', '.join(seen)}). Run scripts/run_real_gpt_realtime_eval.sh on the local PC."
        )


def _websocket():
    try:
        import websocket
    except ImportError as exc:
        raise SystemExit(
            "websocket-client is required. Install it in the local Python environment: "
            "python -m pip install websocket-client"
        ) from exc
    return websocket


def _send(ws: Any, event: dict[str, Any]) -> None:
    ws.send(json.dumps(event, ensure_ascii=False))


def _receive_until_configured(ws: Any, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        ws.settimeout(max(0.1, deadline - time.monotonic()))
        event = json.loads(ws.recv())
        if event.get("type") == "session.updated":
            return
        if event.get("type") == "error":
            raise RuntimeError(f"Realtime session.update failed: {event.get('error')}")
    raise RuntimeError("Timed out waiting for Realtime session.updated")


def build_turn_detection(mode: str) -> dict[str, Any] | None:
    """Server VAD settings, or None for manual turn taking.

    None is serialized as an explicit "turn_detection": null rather than being
    omitted: the input here is a pre-recorded WAV streamed at wall-clock speed,
    and a server VAD left at its default fires on pauses inside it, cutting the
    turn short before the user utterance has finished.
    """
    if mode == "server_vad":
        return {
            "type": "server_vad", "create_response": True,
            "interrupt_response": True, "silence_duration_ms": 500,
        }
    return None


def build_session_update(args: argparse.Namespace) -> dict[str, Any]:
    """session.update in the dialect the endpoint speaks.

    The GA and beta schemas are not compatible -- field names, nesting and the
    modality list all differ -- so an endpoint that only implements the older
    one rejects the GA payload outright instead of degrading.
    """
    turn_detection = build_turn_detection(args.turn_detection)
    if args.realtime_schema == "legacy":
        return {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": args.instructions,
                "voice": args.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": turn_detection,
                "max_response_output_tokens": args.max_output_tokens,
            },
        }
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": args.instructions,
            "output_modalities": ["audio"],
            "max_output_tokens": args.max_output_tokens,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": PCM_RATE},
                    "turn_detection": turn_detection,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": PCM_RATE},
                    "voice": args.voice,
                },
            },
        },
    }


def open_realtime_socket(args: argparse.Namespace) -> Any:
    """Connect, retrying: a corporate gateway refuses or resets often enough
    that one failure should not end a run that is already paying per case."""
    websocket = _websocket()
    url, headers, _ = load_realtime_connection(args.provider, args.model)
    if args.realtime_schema == "legacy":
        # This header pins the beta schema, which is exactly what the legacy
        # payload needs; sending it with the GA payload is what breaks GA.
        headers = headers + ["OpenAI-Beta: realtime=v1"]
    proxy = realtime_proxy_options(url)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, args.connect_retries) + 1):
        try:
            return websocket.create_connection(
                url, header=headers, timeout=args.response_timeout_sec, **proxy
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max(1, args.connect_retries):
                break
            wait = args.retry_sleep_sec * attempt
            print(
                f"[gpt-realtime] connect retry {attempt}/{args.connect_retries} "
                f"in {wait:.1f}s after {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise RuntimeError(f"Could not open the Realtime WebSocket at {url}: {last_exc}") from last_exc


def run_realtime_turn(args: argparse.Namespace, pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, str, float | None, float]:
    ws = open_realtime_socket(args)
    try:
        _send(ws, build_session_update(args))
        _receive_until_configured(ws, args.response_timeout_sec)

        input_pcm = resample_linear(pcm, sample_rate, PCM_RATE)
        raw = np.clip(input_pcm, -1.0, 1.0)
        raw = (raw * 32767.0).astype("<i2").tobytes()
        chunk_bytes = max(2, int(PCM_RATE * args.chunk_ms / 1000) * 2)
        started = time.perf_counter()
        audio_parts: list[bytes] = []
        transcript: list[str] = []
        first_audio_sec: list[float | None] = [None]
        response_error: list[str | None] = [None]
        response_done = threading.Event()

        def receive_response() -> None:
            """Receive while audio is still being played to preserve early
            interjection timing. websocket-client permits one sending and one
            receiving thread; only this thread calls recv after setup."""
            try:
                while not response_done.is_set():
                    ws.settimeout(args.response_timeout_sec)
                    event = json.loads(ws.recv())
                    kind = str(event.get("type", ""))
                    if kind == "error":
                        response_error[0] = f"Realtime API error: {event.get('error')}"
                        response_done.set()
                        return
                    if kind in {"response.output_audio.delta", "response.audio.delta"}:
                        if first_audio_sec[0] is None:
                            first_audio_sec[0] = time.perf_counter() - started
                        audio_parts.append(base64.b64decode(event.get("delta", "")))
                    elif kind in {
                        "response.output_audio_transcript.delta",
                        "response.audio_transcript.delta",
                    }:
                        transcript.append(str(event.get("delta", "")))
                    elif kind == "response.done":
                        # Audio responses include a final transcript in the
                        # completed item.  Delta events are normally emitted,
                        # but this fallback keeps judge text intact if a
                        # gateway coalesces or omits transcript deltas.
                        if not transcript:
                            for item in (event.get("response") or {}).get("output") or []:
                                for content in item.get("content") or []:
                                    value = content.get("transcript") or content.get("text")
                                    if value:
                                        transcript.append(str(value))
                        response_done.set()
                        return
            except Exception as exc:  # surfaced on the main evaluation thread
                response_error[0] = f"Realtime receive failed: {type(exc).__name__}: {exc}"
                response_done.set()

        receiver = threading.Thread(target=receive_response, daemon=True)
        receiver.start()
        for offset in range(0, len(raw), chunk_bytes):
            chunk = raw[offset : offset + chunk_bytes]
            _send(ws, {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})
            if args.input_mode == "realtime":
                expected = started + min(len(raw), offset + len(chunk)) / (PCM_RATE * 2)
                delay = expected - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        if args.turn_detection == "manual":
            _send(ws, {"type": "input_audio_buffer.commit"})
            _send(ws, {"type": "response.create"})
        if not response_done.wait(args.response_timeout_sec):
            raise RuntimeError(f"Timed out after {args.response_timeout_sec}s waiting for Realtime response.done")
        receiver.join(timeout=1)
        if response_error[0]:
            raise RuntimeError(response_error[0])
        output = np.frombuffer(b"".join(audio_parts), dtype="<i2").astype(np.float32) / 32768.0
        return output, "".join(transcript).strip(), first_audio_sec[0], time.perf_counter() - started
    finally:
        ws.close()


def process_variant(args: argparse.Namespace, sample: dict[str, Any], seed: int, variant: str) -> None:
    source_dir = args.dataset_dir / sample["path"]
    trial_dir = args.out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "metadata.json", trial_dir / "metadata.json")
    input_wav = source_dir / f"{variant}.wav"
    shutil.copy2(input_wav, trial_dir / f"{variant}.wav")
    pcm, sample_rate = read_wav_mono(input_wav)
    # Retried at case level, not inside the turn: a socket that dropped mid-turn
    # has already lost part of the response, so the case is redone from a fresh
    # session rather than stitched back together.
    attempts = max(1, args.case_retries)
    for attempt in range(1, attempts + 1):
        try:
            response_pcm, transcript, first_audio, wall = run_realtime_turn(args, pcm, sample_rate)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts:
                raise
            wait = args.retry_sleep_sec * attempt
            print(
                f"[gpt-realtime] case retry {attempt}/{attempts} for "
                f"{sample['task']}/{sample['id']} ({variant}) in {wait:.1f}s "
                f"after {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
    output_stem = "clean_output" if variant == "clean_input" else "output"
    # first_audio is on the same clock as local playback of input.wav. This
    # preserves an early Realtime interjection instead of pretending it began
    # only after the user WAV finished.
    start_sec = first_audio if first_audio is not None else len(pcm) / sample_rate + wall
    silence = np.zeros(int(round(start_sec * sample_rate)), dtype=np.float32)
    response_at_input_rate = resample_linear(response_pcm, PCM_RATE, sample_rate)
    aligned = np.concatenate([silence, response_at_input_rate])
    write_wav_mono(trial_dir / f"{output_stem}.wav", aligned, sample_rate)
    write_wav_stereo(trial_dir / f"{output_stem}_stereo.wav", pcm, aligned, sample_rate)
    chunks = []
    if transcript and response_pcm.size:
        chunks.append({"text": transcript, "timestamp": [round(start_sec, 4), round(start_sec + len(response_at_input_rate) / sample_rate, 4)]})
    (trial_dir / f"{output_stem}.json").write_text(json.dumps({"text": transcript, "chunks": chunks, "source": "gpt_realtime_output_audio_transcript", "language": "ja"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (trial_dir / f"{output_stem}.meta.json").write_text(json.dumps({"model_id": args.model_id, "system": "gpt_realtime", "seed": seed, "task": sample["task"], "case_id": sample["id"], "variant": variant, "input_duration_sec": round(len(pcm) / sample_rate, 4), "wall_time_sec": round(wall, 4), "audible_response_start_sec": round(start_sec, 4), "response_chars": len(transcript), "realtime_provider": args.provider, "realtime_model": args.model, "realtime_schema": args.realtime_schema, "realtime_voice": args.voice, "input_mode": args.input_mode, "turn_detection": args.turn_detection}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    reject_server()
    # Resolve the endpoint, credentials and the deployment/model name once,
    # before any audio is read, so a misconfigured environment fails immediately
    # instead of after the first case has been prepared.
    args.provider = resolve_provider(args.provider)
    url, _, args.model = load_realtime_connection(args.provider, args.model)
    print(f"[gpt-realtime] provider={args.provider} schema={args.realtime_schema} endpoint={url}", file=sys.stderr)
    if args.chunk_ms <= 0:
        raise SystemExit("--chunk-ms must be positive")
    args.dataset_dir = args.dataset_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    tasks = None if args.tasks == "all" else {item.strip() for item in args.tasks.split(",") if item.strip()}
    samples = select_samples(manifest["samples"], tasks, args.cases_per_task)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    (args.out_dir / "run_config.json").write_text(json.dumps({"model_id": args.model_id, "system": "gpt_realtime", "provider": args.provider, "model": args.model, "realtime_schema": args.realtime_schema, "voice": args.voice, "seeds": seeds, "tasks": sorted(tasks) if tasks else "all", "cases_per_task": args.cases_per_task, "input_mode": args.input_mode, "turn_detection": args.turn_detection, "local_only": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = len(samples) * len(seeds)
    completed = 0
    for sample in samples:
        variants = ["input"] + (["clean_input"] if (args.dataset_dir / sample["path"] / "clean_input.wav").is_file() else [])
        for seed in seeds:
            trial_dir = args.out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
            expected = [trial_dir / ("clean_output.meta.json" if variant == "clean_input" else "output.meta.json") for variant in variants]
            if all(path.exists() for path in expected) and not args.overwrite:
                print(f"[gpt-realtime] skip existing {trial_dir}")
            else:
                for variant in variants:
                    process_variant(args, sample, seed, variant)
            completed += 1
            print(f"[gpt-realtime] {completed}/{total} {sample['task']}/{sample['id']} seed={seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
