"""Closed-loop Moshi driver: inject a scripted repair turn when the model
asks for clarification.

The protocol per trial:

  1. Play the (possibly corrupted) user request audio into the model.
  2. Wait for the model's response turn (its own text-token stream, no
     output ASR needed).
  3. If the turn is classified online as a clarification request, inject
     the pre-synthesized repair utterance (the cooperative user restating
     the slot value), then wait for the model's final turn.
  4. Stop after the final turn (or on timeout).

The decision logic lives in `ClosedLoopPolicy`, a pure-python frame-clocked
state machine, so it is unit-testable without torch. `run_closed_loop_trial`
wires the policy into the same mimi->lm_gen streaming loop that
`response_recorder.run_trial` uses (one 80 ms frame per step).

Design note: the online classifier only gates repair injection. Metrics are
recomputed offline from the full event log, and the online/offline label
agreement is itself reported (`policy_events` records the online label).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .detector import classify_turn

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# policy state machine (pure python)
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    slot_type: str
    slot_value: str | None
    utterance_end_sec: float        # when the base user audio ends
    language: str = "ja"            # detector language pack
    turn_gap_sec: float = 1.2       # silence in text stream that ends a turn
    overlap_slack_sec: float = 0.5  # text this early before utt end counts
    no_response_timeout_sec: float = 8.0
    post_turn_timeout_sec: float = 10.0  # max wait for the final turn
    repair_lead_silence_sec: float = 0.4
    tail_sec: float = 2.0           # silence appended after the final turn
    min_turn_chars: int = 2


# Actions the driver executes.
CONTINUE = "continue"
INJECT_REPAIR = "inject_repair"
STOP = "stop"


@dataclass
class PolicyEvent:
    time_sec: float
    kind: str                      # turn_end | inject | stop | timeout
    detail: dict[str, Any] = field(default_factory=dict)


class ClosedLoopPolicy:
    """Frame-clocked FSM. Call `on_step(time_sec, text_piece)` once per
    80 ms frame; it returns one of CONTINUE / INJECT_REPAIR / STOP."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.state = "WAIT_TURN"       # WAIT_TURN -> IN_TURN -> (repair)
        self.turn_index = 0            # 0 = first response, 1 = post-repair
        self.buffer: list[str] = []
        self.last_text_sec: float | None = None
        self.wait_started_sec = cfg.utterance_end_sec
        self.repair_injected = False
        self.events: list[PolicyEvent] = []
        self.turns: list[dict[str, Any]] = []
        self._listen_from = cfg.utterance_end_sec - cfg.overlap_slack_sec
        self._tail_until = float("inf")

    # -- helpers ----------------------------------------------------------

    def _turn_text(self) -> str:
        return "".join(self.buffer).strip()

    def _classify(self) -> Any:
        return classify_turn(
            self._turn_text(), self.cfg.slot_type, self.cfg.slot_value,
            lang=self.cfg.language,
        )

    def _end_turn(self, time_sec: float) -> str:
        label = self._classify()
        self.turns.append({
            "index": self.turn_index,
            "text": self._turn_text(),
            "label": label.label,
            "end_sec": time_sec,
        })
        self.events.append(PolicyEvent(time_sec, "turn_end", {
            "index": self.turn_index,
            "label": label.label,
            "text": self._turn_text(),
        }))
        self.buffer = []
        self.last_text_sec = None
        if self.turn_index == 0 and label.is_clarification \
                and not self.repair_injected:
            self.repair_injected = True
            self.turn_index = 1
            # Ignore text (e.g. backchannels) while the repair plays; the
            # driver's notify_repair_end() re-arms listening.
            self.state = "REPAIR_PLAYING"
            self.events.append(PolicyEvent(time_sec, "inject", {}))
            self.wait_started_sec = time_sec  # reset by driver after repair
            return INJECT_REPAIR
        self.state = "DONE_TAIL"
        self._tail_until = time_sec + self.cfg.tail_sec
        return CONTINUE

    def notify_repair_end(self, time_sec: float) -> None:
        """Driver calls this when the injected repair audio finished."""
        self.wait_started_sec = time_sec
        self._listen_from = time_sec - self.cfg.overlap_slack_sec
        self.state = "WAIT_TURN"

    # -- main entry ---------------------------------------------------------

    def on_step(self, time_sec: float, text_piece: Optional[str]) -> str:
        if self.state == "DONE_TAIL":
            if time_sec >= self._tail_until:
                self.events.append(PolicyEvent(time_sec, "stop", {}))
                return STOP
            return CONTINUE

        if self.state == "REPAIR_PLAYING":
            # Swallow tokens emitted during repair playback (backchannels);
            # a safety timeout guards against a driver that never calls
            # notify_repair_end.
            if time_sec - self.wait_started_sec >= 30.0:
                self.events.append(PolicyEvent(time_sec, "timeout",
                                               {"phase": "repair_playing"}))
                self.notify_repair_end(time_sec)
            return CONTINUE

        if text_piece and text_piece.strip() and time_sec >= self._listen_from:
            self.buffer.append(text_piece)
            self.last_text_sec = time_sec
            if self.state == "WAIT_TURN":
                self.state = "IN_TURN"
            return CONTINUE

        if self.state == "IN_TURN":
            assert self.last_text_sec is not None
            if (time_sec - self.last_text_sec >= self.cfg.turn_gap_sec
                    and len(self._turn_text()) >= self.cfg.min_turn_chars):
                return self._end_turn(time_sec)
            timeout = (self.cfg.post_turn_timeout_sec
                       if self.turn_index else self.cfg.no_response_timeout_sec)
            if time_sec - self.wait_started_sec >= timeout + 20.0:
                # Runaway turn that never pauses; force-close it.
                self.events.append(PolicyEvent(time_sec, "timeout",
                                               {"phase": "in_turn"}))
                return self._end_turn(time_sec)
            return CONTINUE

        # WAIT_TURN: no text yet for this turn.
        timeout = (self.cfg.post_turn_timeout_sec
                   if self.turn_index else self.cfg.no_response_timeout_sec)
        if time_sec - self.wait_started_sec >= timeout:
            self.events.append(PolicyEvent(time_sec, "timeout",
                                           {"phase": "wait_turn",
                                            "turn_index": self.turn_index}))
            self.turns.append({
                "index": self.turn_index, "text": "", "label": "no_response",
                "end_sec": time_sec,
            })
            self.events.append(PolicyEvent(time_sec, "stop", {}))
            return STOP
        return CONTINUE

    def summary(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "repair_injected": self.repair_injected,
            "events": [dataclasses.asdict(e) for e in self.events],
        }


# ---------------------------------------------------------------------------
# streaming trial runner (torch / moshi imported lazily)
# ---------------------------------------------------------------------------

def run_closed_loop_trial(
    base_pcm: np.ndarray,
    repair_pcm: np.ndarray,
    cfg: PolicyConfig,
    seed: int,
    lm_gen: Any,
    mimi: Any,
    text_tokenizer: Any,
    device: str,
    max_total_sec: float = 90.0,
) -> dict[str, Any]:
    """Run one closed-loop trial. Returns the same shape as
    `response_recorder.run_trial` plus `policy` (turns/events/injection)."""
    import sys
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import response_recorder as recorder

    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(sample_rate / frame_rate)

    recorder.seed_all(seed)

    policy = ClosedLoopPolicy(cfg)

    # Dynamic input buffer: starts with the user request; silence is
    # appended on demand; the repair turn is appended when the policy says
    # so. Everything is float32 mono at mimi's sample rate.
    buffer = np.asarray(base_pcm, dtype=np.float32).copy()
    lead = np.zeros(int(cfg.repair_lead_silence_sec * sample_rate),
                    dtype=np.float32)
    repair_with_lead = np.concatenate(
        [lead, np.asarray(repair_pcm, dtype=np.float32)]
    )

    max_steps = int(max_total_sec * frame_rate)
    audio_frames: list[Any] = []
    text_events: list[dict[str, Any]] = []
    first_audio_step: int | None = None
    first_response_step: int | None = None
    repair_start_step: int | None = None
    repair_end_step: int | None = None
    stop_step: int | None = None

    with torch.no_grad():
        with lm_gen.streaming(1), mimi.streaming(1):
            step = 0
            while step < max_steps:
                start = step * frame_size
                end = start + frame_size
                if end > len(buffer):
                    buffer = np.concatenate([
                        buffer,
                        np.zeros(end - len(buffer), dtype=np.float32),
                    ])
                chunk = (
                    torch.from_numpy(np.ascontiguousarray(buffer[start:end]))
                    .to(device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                codes = mimi.encode(chunk)
                piece: str | None = None
                if codes is not None:
                    out = lm_gen.step(codes)
                    if out is not None:
                        out_cpu = out[0].cpu()
                        text_id = int(out_cpu[0, 0].item())
                        audio_tok = out_cpu[1:, :]
                        if recorder._audio_tokens_are_decodable(audio_tok):
                            if first_audio_step is None:
                                first_audio_step = step
                            audio_frames.append(audio_tok)
                        piece = recorder._decode_text_piece(
                            text_tokenizer, text_id
                        )
                        if piece is not None:
                            time_sec = round(step / frame_rate, 4)
                            text_events.append({
                                "step": step, "time_sec": time_sec,
                                "piece": piece,
                            })
                            if first_response_step is None:
                                first_response_step = step

                time_sec = step / frame_rate
                if repair_end_step is not None and step == repair_end_step:
                    policy.notify_repair_end(time_sec)
                action = policy.on_step(time_sec, piece)
                if action == INJECT_REPAIR:
                    # Append the repair turn right after the current frame.
                    insert_at = (step + 1) * frame_size
                    if insert_at > len(buffer):
                        buffer = np.concatenate([
                            buffer,
                            np.zeros(insert_at - len(buffer), dtype=np.float32),
                        ])
                    buffer = np.concatenate(
                        [buffer[:insert_at], repair_with_lead]
                    )
                    repair_start_step = step + 1
                    repair_end_step = step + 1 + int(
                        np.ceil(len(repair_with_lead) / frame_size)
                    )
                elif action == STOP:
                    stop_step = step
                    break
                step += 1

    total_steps = (stop_step + 1) if stop_step is not None else max_steps
    return {
        "audio_frames": audio_frames,
        "text_events": text_events,
        "total_steps": total_steps,
        "input_steps": int(np.ceil(len(base_pcm) / frame_size)),
        "first_audio_step": first_audio_step,
        "first_response_step": first_response_step,
        "policy": policy.summary() | {
            "repair_start_step": repair_start_step,
            "repair_end_step": repair_end_step,
            "frame_rate": frame_rate,
        },
        "input_pcm_final": buffer[: total_steps * frame_size],
    }
