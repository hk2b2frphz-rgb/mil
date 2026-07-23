from pathlib import Path

from scripts.run_experiment_dag import build_context, load_yaml, stage_env


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
    assert "cuda12.6_cudnn9.7.1_nccl2.24.3" in wrapper
    assert ".venv-vllm-omni/bin/python" in wrapper
