import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_experiment_dag import build_context, load_yaml, stage_env

VLLM_COMMON = Path("scripts/run_qwen_tts_vllm_common.sh")
# Every scale runs the same body; only the corpus default differs.
VLLM_WRAPPERS = {
    1000: Path("scripts/run_qwen_tts_vllm_1000_4gpu.pbs"),
    3000: Path("scripts/run_qwen_tts_vllm_3000_4gpu.pbs"),
    10000: Path("scripts/run_qwen_tts_vllm_10000_4gpu.pbs"),
}


def test_example_dag_uses_v100_vllm_environment_and_real_stage_paths() -> None:
    config_path = Path("configs/experiment.example.yaml")
    config = load_yaml(config_path)
    context = build_context(config)
    stages = {stage["name"]: stage for stage in config["stages"]}

    dialogue_env = stage_env(config, stages["dialogue"], context)
    assert dialogue_env["OUT_ROOT"] == "data/runs/exp01/dialogue"

    tts = stages["tts"]
    tts_env = stage_env(config, tts, context)
    assert tts["pbs"] == "scripts/run_qwen_tts_vllm_10000_4gpu.pbs"
    assert (
        tts_env["DIALOGUES_JSONL"]
        == "data/runs/exp01/dialogue/llm_dialogues/dialogues.jsonl"
    )
    assert tts_env["OUT_ROOT"] == "data/runs/exp01/tts"

    train_env = stage_env(config, stages["train"], context)
    assert train_env["SRC_RUN_DIR"] == "data/runs/exp01/tts/merged"
    assert (
        train_env["MERGED_OUT"]
        == "data/runs/exp01/checkpoints/consolidated.safetensors"
    )

    eval_env = stage_env(config, stages["eval"], context)
    assert eval_env["MODEL_WEIGHT"] == train_env["MERGED_OUT"]
    assert eval_env["FDB_OUT_DIR"] == "data/runs/exp01/eval"

    wrapper = Path(tts["pbs"]).read_text(encoding="utf-8")
    assert "run_qwen_tts_vllm_common.sh" in wrapper

    shared = VLLM_COMMON.read_text(encoding="utf-8")
    assert "cuda12.6_cudnn9.7.1_nccl2.24.3" in shared
    assert ".venv-vllm-omni/bin/python" in shared
    assert "QWEN_VOICE_MODE" in shared
    assert "tts_vllm_mixed_4gpu" in shared
    assert "tts_vllm_clone_4gpu" in shared
    assert 'default_run_id "$default_batch_id"' in shared
    assert 'export QWEN_VOICE_MODE="$(resolve_qwen_voice_mode)"' in shared
    assert "a user clone source without a moshi clone source is unsupported" in shared

    production = Path(
        "scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs"
    ).read_text(encoding="utf-8")
    assert "--qwen-voice-mode" in production
    assert "--qwen-clone-model" in production
    assert "CLONE_OUT_DIR_MOSHI" in production
    assert ".qwen_mixed_cache" in production
    assert "generation_config.json" in production
    assert "dialogues_sha256" in production
    assert "moshi_ref_wav_sha256" in production
    assert "vllm_stage_config_sha256" in production
    assert "generator_sha256" in production
    assert "qwen3_tts_vllm_backend_sha256" in production
    assert "generation settings differ from the existing run" in production
    assert "existing legacy output has no generation config lock" in production
    assert "Qwen clone/mixed voices require TTS_BACKEND=qwen3" in production
    assert 'if ! [[ "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then' in production
    assert production.index('export TTS_BACKEND="${TTS_BACKEND:-kokoro}"') < (
        production.index('export QWEN_VOICE_MODE="$(resolve_qwen_voice_mode)"')
    )
    assert (
        '"$VLLM_PYTHON" scripts/patch_vllm_omni_qwen3_tts_v100.py --check'
        in production
    )

    pilot = Path("scripts/run_clone_dialogue_pilot.pbs").read_text(
        encoding="utf-8"
    )
    assert "QWEN_VOICE_MODE=mixed" in pilot
    assert "--qwen-voice-mode" in pilot
    assert "USER_SPEAKER_POOL" in pilot
    assert 'QWEN_VOICE_MODE="${QWEN_VOICE_MODE:-auto}"' in pilot
    assert "MOSHI_REF_TEXT is required with an explicit MOSHI_REF_WAV" in pilot
    assert 'if [[ -z "${ANALYSIS_DIR:-}" ]]' in pilot


def test_v100_profile_caps_code2wav_without_reducing_outer_batch() -> None:
    config = load_yaml(Path("configs/qwen3_tts_v100_batch16.yaml"))
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]

    assert extra["decode_cudagraph_capture_sizes"] == [25, 73, 97, 169, 325]
    assert extra["decode_cudagraph_batch_sizes"] == [1]
    # Capture-time cap only: throttling the live decoder aborted the CUDA
    # context with a device-side assert.
    assert "decode_batch_max_size" not in extra
    assert {stage["stage_id"]: stage["max_num_seqs"] for stage in config["stages"]} == {
        0: 16,
        1: 16,
    }

    shared = VLLM_COMMON.read_text(encoding="utf-8")
    production = Path(
        "scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs"
    ).read_text(encoding="utf-8")
    assert 'TTS_BATCH_SIZE="${TTS_BATCH_SIZE:-16}"' in shared
    assert 'DIALOGUE_BATCH_SIZE="${DIALOGUE_BATCH_SIZE:-16}"' in shared
    assert 'TTS_BATCH_SIZE="${TTS_BATCH_SIZE:-16}"' in production
    assert 'DIALOGUE_BATCH_SIZE="${DIALOGUE_BATCH_SIZE:-16}"' in production


def test_every_scale_has_a_vllm_wrapper_over_the_shared_body() -> None:
    for scale, path in VLLM_WRAPPERS.items():
        wrapper = path.read_text(encoding="utf-8")
        assert (
            f'SOURCE_BATCH_ID:-qwen_dialogues_{scale}_${{DIALOGUES_VERSION:-v1}}'
            in wrapper
        ), path
        assert "exec bash scripts/run_qwen_tts_vllm_common.sh" in wrapper, path
        # The body must not be copied into the wrappers, or a fix to the voice
        # mode rules would have to be made once per scale.
        assert "resolve_qwen_voice_mode() {" not in wrapper, path

    shared = VLLM_COMMON.read_text(encoding="utf-8")
    assert "SOURCE_BATCH_ID must be set" in shared


def _bash_executable() -> str | None:
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "bin"
            / "bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "usr"
            / "bin"
            / "bash.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash")


def test_changed_pbs_scripts_pass_bash_syntax_check() -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is unavailable")
    scripts = [
        *(str(path) for path in VLLM_WRAPPERS.values()),
        str(VLLM_COMMON),
        "scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs",
        "scripts/run_clone_dialogue_pilot.pbs",
    ]
    result = subprocess.run(
        [bash, "-n", *scripts],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _generation_config_lock_source() -> str:
    production = Path(
        "scripts/run_qwen_tts_whole_utterance_10000_4gpu.pbs"
    ).read_text(encoding="utf-8")
    anchor = 'uv run python - "$GENERATION_CONFIG_LOCK" <<\'PY\'\n'
    assert anchor in production
    return production.split(anchor, 1)[1].split("\nPY\n", 1)[0]


def test_generation_config_lock_embedded_python_has_valid_syntax() -> None:
    embedded = _generation_config_lock_source()
    ast.parse(embedded)


def _generation_config_env(
    tmp_path: Path,
    *,
    tts_backend: str = "qwen3",
    voice_mode: str = "mixed",
) -> tuple[dict[str, str], Path]:
    dialogues = tmp_path / "dialogues.jsonl"
    stage_config = tmp_path / "stage.yaml"
    moshi_ref = tmp_path / "moshi.wav"
    dialogues.write_text("{}\n", encoding="utf-8")
    stage_config.write_text("stages: []\n", encoding="utf-8")
    moshi_ref.write_bytes(b"reference-audio-v1")

    env = dict(os.environ)
    env.update(
        {
            "SOURCE_BATCH_ID": "test_dialogues",
            "DIALOGUES_JSONL": str(dialogues),
            "NUM_DIALOGUES": "1",
            "NUM_SHARDS": "1",
            "SPARE_RATIO": "0.15",
            "TTS_BACKEND": tts_backend,
            "QWEN_ENGINE": "vllm-omni",
            "QWEN_VOICE_MODE": voice_mode,
            "QWEN_TTS_MODEL": "test-custom-model",
            "QWEN_CLONE_MODEL": "test-clone-model",
            "TTS_DTYPE": "float16",
            "TTS_DEVICE": "cuda",
            "TTS_LANGUAGE": "Japanese",
            "SPEAKER_USER": "Ono_Anna",
            "SPEAKER_MOSHI": "Vivian",
            "SPEAKER_OTHER": "Ono_Anna",
            "SPEAKER_BACKGROUND": "Ono_Anna",
            "USER_SPEAKER_POOL": "Ono_Anna",
            "INSTRUCT_USER": "",
            "INSTRUCT_MOSHI": "",
            "STYLE_PRESET": "none",
            "LEAD_IN_SEC": "0.3",
            "GAP_SEC": "0.2",
            "WHOLE_UTTERANCE_MAX_CHARS": "150",
            "TTS_BATCH_SIZE": "16",
            "DIALOGUE_BATCH_SIZE": "16",
            "VLLM_MAX_NEW_TOKENS": "2048",
            "QWEN_CLONE_MAX_NEW_TOKENS": "4096",
            "VLLM_STAGE_CONFIG": str(stage_config),
            "KOKORO_VOICE_MOSHI": "jf_alpha",
            "KOKORO_USER_VOICES": "jf_alpha",
            "CLONE_MODE": "in-context",
            "REF_RANK": "0",
            "MOSHI_REF_WAV": str(moshi_ref),
            "MOSHI_REF_TEXT": "参照テキスト",
            "USER_REF_WAV": "",
            "USER_REF_TEXT": "",
        }
    )
    return env, moshi_ref


def _run_generation_config_lock(
    embedded: str,
    lock_path: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", str(lock_path)],
        cwd=Path(__file__).resolve().parents[1],
        input=embedded,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_generation_config_lock_create_match_and_reject_changed_reference(
    tmp_path: Path,
) -> None:
    embedded = _generation_config_lock_source()
    env, moshi_ref = _generation_config_env(tmp_path)
    lock_path = tmp_path / "run" / "generation_config.json"

    created = _run_generation_config_lock(embedded, lock_path, env)
    assert created.returncode == 0, created.stderr
    assert "generation config lock created" in created.stdout
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert "qwen3_tts_vllm_backend_sha256" in payload["config"]

    matched = _run_generation_config_lock(embedded, lock_path, env)
    assert matched.returncode == 0, matched.stderr
    assert "generation config lock matched" in matched.stdout

    moshi_ref.write_bytes(b"reference-audio-v2")
    changed = _run_generation_config_lock(embedded, lock_path, env)
    assert changed.returncode != 0
    assert "generation settings differ from the existing run" in changed.stderr
    assert "moshi_ref_wav_sha256" in changed.stderr


def test_generation_config_lock_does_not_hash_clone_refs_for_kokoro(
    tmp_path: Path,
) -> None:
    embedded = _generation_config_lock_source()
    env, _ = _generation_config_env(
        tmp_path,
        tts_backend="kokoro",
        voice_mode="mixed",
    )
    env["MOSHI_REF_WAV"] = ""
    lock_path = tmp_path / "kokoro-run" / "generation_config.json"

    created = _run_generation_config_lock(embedded, lock_path, env)
    assert created.returncode == 0, created.stderr
    config = json.loads(lock_path.read_text(encoding="utf-8"))["config"]
    assert "moshi_ref_wav_sha256" not in config
    assert "qwen3_tts_vllm_backend_sha256" not in config


@pytest.mark.parametrize(
    ("extra_env", "expected", "should_succeed"),
    [
        ({}, "customvoice", True),
        ({"CLONE_OUT_DIR_MOSHI": "moshi"}, "mixed", True),
        (
            {
                "CLONE_OUT_DIR_MOSHI": "moshi",
                "CLONE_OUT_DIR_USER": "user",
            },
            "clone",
            True,
        ),
        ({"CLONE_OUT_DIR_USER": "user"}, "", False),
        (
            {
                "QWEN_VOICE_MODE": "customvoice",
                "CLONE_OUT_DIR_MOSHI": "moshi",
            },
            "",
            False,
        ),
        (
            {
                "QWEN_VOICE_MODE": "mixed",
                "CLONE_OUT_DIR_MOSHI": "moshi",
                "CLONE_OUT_DIR_USER": "user",
            },
            "",
            False,
        ),
    ],
)
def test_vllm_wrapper_voice_mode_resolution_harness(
    extra_env: dict[str, str], expected: str, should_succeed: bool
) -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is unavailable")
    shared = VLLM_COMMON.read_text(encoding="utf-8")
    start = shared.index("resolve_qwen_voice_mode() {")
    end = shared.index("\n}\nexport QWEN_VOICE_MODE=", start) + 3
    function_source = shared[start:end]

    env = dict(os.environ)
    for name in (
        "QWEN_VOICE_MODE",
        "CLONE_OUT_DIR_MOSHI",
        "CLONE_OUT_DIR_USER",
        "MOSHI_REF_WAV",
        "MOSHI_REF_TEXT",
        "USER_REF_WAV",
        "USER_REF_TEXT",
    ):
        env.pop(name, None)
    env.update(extra_env)
    result = subprocess.run(
        [bash, "-c", f"set -u\n{function_source}\nresolve_qwen_voice_mode"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert (result.returncode == 0) is should_succeed, result.stderr
    if should_succeed:
        assert result.stdout == expected
