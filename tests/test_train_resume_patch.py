"""チェーン学習のための resume パッチのテスト。

moshi-finetune 本体はこのリポジトリに無いので、パッチが差し込み先として
期待している最小限のソース断片をフィクスチャとして作り、そこへ当てて検証する。
アンカーが上流の更新で消えた場合は patch 関数が例外を投げるので、この
テストが落ちることで気づける。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.patch_kyutai_moshi_finetune import (  # noqa: E402
    patch_args_resume,
    patch_train_resume,
    patch_wrapped_model_resume,
)

ARGS_FIXTURE = '''from dataclasses import dataclass


@dataclass
class TrainArgs:
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False
    keep_best_only: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
'''

WRAPPED_MODEL_FIXTURE = '''import logging

import safetensors.torch

logger = logging.getLogger(__name__)


def get_fsdp_model(args, checkpointer_info):
    model = object()
    if get_rank() == 0:
        model.load_state_dict(model_state_dict, strict=False, assign=True)

        if args.lora.enable and not args.full_finetuning:
            logger.info("Initializing lora layers ...")
            # initialize LoRA layers
            initialize_lora_parameters(model, param_dtype)

    return model
'''

TRAIN_FIXTURE = '''def train(config: str):
    scheduler = build_scheduler()

    state = TrainState(args.max_steps)

    while state.step < args.max_steps:
        state.start_step()
'''


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_args_resume_adds_fields_and_is_idempotent(tmp_path: Path) -> None:
    path = _write(tmp_path, "args.py", ARGS_FIXTURE)

    assert patch_args_resume(path) is True
    src = path.read_text(encoding="utf-8")
    assert "resume_from: str" in src
    assert "resume_step: int = 0" in src
    assert "stop_at_step: int = 0" in src
    ast.parse(src)

    # 2 回目は何もしないこと。実験起動のたびに走るので冪等性は必須。
    assert patch_args_resume(path) is False
    assert path.read_text(encoding="utf-8") == src


def test_wrapped_model_resume_loads_before_sharding(tmp_path: Path) -> None:
    path = _write(tmp_path, "wrapped_model.py", WRAPPED_MODEL_FIXTURE)

    assert patch_wrapped_model_resume(path) is True
    src = path.read_text(encoding="utf-8")
    ast.parse(src)

    # 差し込み位置が initialize_lora_parameters の後であること。FSDP でラップ
    # された後だと state_dict のキーにシャードの接頭辞が付き、strict=False の
    # 下で何にもマッチしないまま素通りしてしまう。
    init_pos = src.index("initialize_lora_parameters(model, param_dtype)")
    load_pos = src.index("AUTO_PATCH_RESUME_FROM_LORA")
    assert init_pos < load_pos

    # 取り違えが静かに通らないこと。
    assert "shares no parameter names" in src
    assert "lora.rank" in src  # 形状不一致メッセージ
    assert "contains no LoRA tensors" in src

    assert patch_wrapped_model_resume(path) is False


def test_train_resume_sets_step_and_stop(tmp_path: Path) -> None:
    path = _write(tmp_path, "train.py", TRAIN_FIXTURE)

    assert patch_train_resume(path) is True
    src = path.read_text(encoding="utf-8")
    ast.parse(src)

    # scheduler 構築より後で max_steps を狭めること。先に狭めると warmup 長が
    # ジョブごとに変わり、チェーン全体で 1 本の schedule にならない。
    scheduler_pos = src.index("scheduler = build_scheduler()")
    narrow_pos = src.index("args.max_steps = min(args.max_steps, stop_at_step)")
    assert scheduler_pos < narrow_pos

    assert "state.step = resume_step" in src
    assert "scheduler.step()" in src

    assert patch_train_resume(path) is False


def test_train_resume_rejects_inconsistent_settings(tmp_path: Path) -> None:
    path = _write(tmp_path, "train.py", TRAIN_FIXTURE)
    patch_train_resume(path)
    src = path.read_text(encoding="utf-8")

    # resume_step だけ指定して resume_from を忘れると、warmup を飛ばした状態で
    # 初期化直後のアダプタを学習してしまう。黙って通してはいけない。
    assert "resume_from is empty" in src
    # 走る余地が無い指定も弾くこと。
    assert "must be below max_steps" in src
    assert "must be above resume_step" in src


def test_patch_raises_when_anchor_missing(tmp_path: Path) -> None:
    # 上流が該当ブロックを変えたらパッチは黙って諦めず落ちること。
    path = _write(tmp_path, "args.py", "class TrainArgs:\n    pass\n")
    with pytest.raises(RuntimeError):
        patch_args_resume(path)

    path = _write(tmp_path, "train.py", "def train():\n    pass\n")
    with pytest.raises(RuntimeError):
        patch_train_resume(path)

    path = _write(tmp_path, "wrapped_model.py", "def get_fsdp_model():\n    pass\n")
    with pytest.raises(RuntimeError):
        patch_wrapped_model_resume(path)
