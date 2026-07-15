import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corrupt_training_audio import (  # noqa: E402
    locate_target, read_stereo_wav, write_stereo_wav,
)

SR = 24000
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "corrupt_training_audio.py"


def make_training_sample(tmp_path: Path, dialogue_id: str):
    stereo = tmp_path / "training_set" / "data_stereo"
    stereo.mkdir(parents=True)
    t = np.arange(int(4.0 * SR)) / SR
    left = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    wav = stereo / f"sample_001_{dialogue_id}.wav"
    write_stereo_wav(wav, left, right, SR)
    sidecar = {
        "segments": [
            {"speaker": "user", "text": "明日の", "start_sec": 0.2,
             "end_sec": 1.0},
            {"speaker": "user", "text": "15時", "start_sec": 1.0,
             "end_sec": 1.8},
            {"speaker": "user", "text": "にアラームをかけて",
             "start_sec": 1.8, "end_sec": 3.0},
            {"speaker": "moshi", "text": "15時ですね。", "start_sec": 3.2,
             "end_sec": 4.0},
        ]
    }
    wav.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )
    return wav


def test_locate_target_finds_user_slot_segment():
    sidecar = {"segments": [
        {"speaker": "user", "text": "15時", "start_sec": 1.0, "end_sec": 1.8},
        {"speaker": "moshi", "text": "15時ですね。", "start_sec": 3.0,
         "end_sec": 4.0},
    ]}
    span = locate_target(sidecar, "15時")
    assert span == (1.0, 1.8)  # moshi channel must not match


def test_locate_target_missing_returns_none():
    assert locate_target({"segments": []}, "15時") is None


def test_end_to_end_corruption(tmp_path):
    dialogue_id = "clarify_full_ask_00001"
    wav = make_training_sample(tmp_path, dialogue_id)
    plan_path = tmp_path / "corruption_plan.jsonl"
    plan_path.write_text(json.dumps({
        "id": dialogue_id,
        "corruption": {"condition": "mask_silence", "target_text": "15時",
                       "channel": "user"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--training-set-dir", str(tmp_path / "training_set"),
         "--corruption-plan", str(plan_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    left, right, sr = read_stereo_wav(wav)
    # Slot span on the user (right) channel is silenced...
    interior = right[int(1.1 * SR): int(1.7 * SR)]
    assert np.max(np.abs(interior)) < 1e-3
    # ...while audio outside the span and the moshi channel survive.
    assert np.max(np.abs(right[: int(0.9 * SR)])) > 0.1
    assert np.max(np.abs(left)) > 0.1
    # Backup of the clean version exists.
    assert wav.with_suffix(".clean.wav").exists()
    report = (tmp_path / "training_set" / "corruption_report.jsonl") \
        .read_text(encoding="utf-8")
    assert '"status": "ok"' in report


def test_failure_rate_gate(tmp_path):
    dialogue_id = "clarify_full_ask_00002"
    make_training_sample(tmp_path, dialogue_id)
    plan_path = tmp_path / "corruption_plan.jsonl"
    plan_path.write_text(json.dumps({
        "id": dialogue_id,
        "corruption": {"condition": "mask_silence",
                       "target_text": "存在しないテキスト",
                       "channel": "user"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--training-set-dir", str(tmp_path / "training_set"),
         "--corruption-plan", str(plan_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1  # span_not_found > 5% -> job fails
