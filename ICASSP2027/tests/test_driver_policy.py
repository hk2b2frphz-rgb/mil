import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarify.driver import (  # noqa: E402
    CONTINUE, INJECT_REPAIR, STOP, ClosedLoopPolicy, PolicyConfig,
)

FRAME = 0.08  # 12.5 Hz frames


def make_cfg(**kw):
    defaults = dict(
        slot_type="time", slot_value="15時", utterance_end_sec=3.0,
        turn_gap_sec=1.2, no_response_timeout_sec=8.0,
        post_turn_timeout_sec=10.0, tail_sec=1.0,
    )
    defaults.update(kw)
    return PolicyConfig(**defaults)


def drive(policy, script):
    """script: list of (time_sec, piece or None). Returns list of actions."""
    actions = []
    for t, piece in script:
        actions.append(policy.on_step(t, piece))
    return actions


def steps(start, end, piece_at=None):
    """Frame times from start to end; piece_at maps time->text."""
    t = start
    out = []
    while t < end:
        piece = piece_at.get(round(t, 2)) if piece_at else None
        out.append((round(t, 2), piece))
        t += FRAME
    return out


def test_clarification_triggers_repair_injection():
    policy = ClosedLoopPolicy(make_cfg())
    # Model says a clarification 3.5-4.3s, then silence.
    pieces = {3.52: "すみません、", 3.92: "何時", 4.16: "ですか？"}
    script = steps(0.0, 6.0, pieces)
    actions = drive(policy, script)
    assert INJECT_REPAIR in actions
    inject_time = script[actions.index(INJECT_REPAIR)][0]
    assert 5.3 <= inject_time <= 5.7  # ~1.2s after last text
    assert policy.turns[0]["label"] == "clarify_targeted"


def test_confirmation_leads_to_stop_without_repair():
    policy = ClosedLoopPolicy(make_cfg())
    pieces = {3.52: "15時ですね。", 3.92: "登録しますね。"}
    script = steps(0.0, 8.0, pieces)
    actions = drive(policy, script)
    assert INJECT_REPAIR not in actions
    assert STOP in actions
    assert policy.turns[0]["label"] == "confirm"
    assert not policy.repair_injected


def test_no_response_times_out():
    policy = ClosedLoopPolicy(make_cfg(no_response_timeout_sec=4.0))
    script = steps(0.0, 9.0)
    actions = drive(policy, script)
    assert STOP in actions
    stop_time = script[actions.index(STOP)][0]
    assert 6.9 <= stop_time <= 7.3  # utterance_end(3.0) + 4.0 timeout
    assert policy.turns[0]["label"] == "no_response"


def test_full_closed_loop_with_repair_and_final_confirm():
    policy = ClosedLoopPolicy(make_cfg())
    actions = []
    time = 0.0
    injected_at = None
    repair_end = None
    while time < 20.0:
        piece = None
        if 3.5 <= time <= 4.2 and injected_at is None:
            piece = {False: "何時ですか？"}.get(bool(policy.buffer)) \
                if abs(time - 3.52) < 0.04 else None
        # After repair ends, model confirms.
        if repair_end and repair_end + 0.5 <= time <= repair_end + 0.6:
            piece = "15時ですね。設定しますね。"
        action = policy.on_step(round(time, 2), piece)
        actions.append(action)
        if action == INJECT_REPAIR:
            injected_at = time
            repair_end = time + 2.0  # driver plays 2s of repair audio
        if repair_end and abs(time - repair_end) < FRAME / 2:
            policy.notify_repair_end(time)
        if action == STOP:
            break
        time += FRAME
    assert injected_at is not None
    assert actions[-1] == STOP
    assert policy.turns[-1]["label"] == "confirm"
    assert policy.repair_injected


def test_text_during_repair_playback_is_ignored():
    policy = ClosedLoopPolicy(make_cfg())
    # Clarification turn.
    actions = drive(policy, steps(0.0, 5.6, {3.52: "もう一度お願いします。"}))
    assert INJECT_REPAIR in actions
    # Backchannel while repair audio plays must not open a turn.
    policy.on_step(5.7, "はい")
    assert policy.buffer == []
    policy.notify_repair_end(7.0)
    policy.on_step(7.5, "15時ですね。")
    assert policy.buffer == ["15時ですね。"]


def test_only_one_repair_injection():
    policy = ClosedLoopPolicy(make_cfg())
    actions = drive(policy, steps(0.0, 5.6, {3.52: "何時ですか？"}))
    assert INJECT_REPAIR in actions
    policy.notify_repair_end(7.0)
    # Model asks AGAIN after repair; no second injection, trial just ends.
    actions = drive(policy, steps(7.0, 15.0, {7.48: "何時ですか？"}))
    assert INJECT_REPAIR not in actions
    assert STOP in actions
