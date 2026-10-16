#!/usr/bin/env python3
"""human_reference.responded=false のケースの周辺を並べる。

「相談員は応答していない」という判定が本当かを人が目で確かめるための道具。
判定そのもの(eval/build_real_test_dataset.py の gold_response)は触らない。

各ケースについて出すもの:
  - User 発話の終了時刻と本文
  - 応答窓を閉じた原因(次の User 発話 / 30 秒の上限)
  - 窓の幅
  - user_end の後ろ 30 秒に実在する Staff 発話(窓の外でも全部出す)

窓の幅がほぼ 0 で、その直後に Staff 発話が並んでいるなら、原因は
「次の User 発話で窓が潰れた」であって相談員の無応答ではない。

使い方:
    uv run python eval/diagnose_gold_no_response.py
    uv run python eval/diagnose_gold_no_response.py --dataset-dir data/eval_sets/real_response
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_test_wav_module():
    path = REPO_ROOT / "scripts" / "build_test_wav_from_tsv.py"
    if not path.is_file():
        raise SystemExit(f"見つかりません: {path}")
    spec = importlib.util.spec_from_file_location("_miltoka_build_test_wav", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


read_annotation = _load_build_test_wav_module().read_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("data/eval_sets/real_response"))
    parser.add_argument("--window-sec", type=float, default=30.0,
                        help="周辺として表示する Staff 発話の範囲(既定 30)")
    parser.add_argument("--all", action="store_true",
                        help="responded=true のケースも出す")
    return parser.parse_args()


def clip(text: str, width: int = 46) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    args = parse_args()
    task_dir = args.dataset_dir / "real_response"
    if not task_dir.is_dir():
        raise SystemExit(f"{task_dir} がありません。先にデータセットを作ってください。")

    annotations: dict[str, list] = {}
    silent = 0
    shown = 0

    for case_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        meta_path = case_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        responded = bool((meta.get("human_reference") or {}).get("responded"))
        if responded:
            if not args.all:
                continue
        else:
            silent += 1

        src = meta["source"]
        annotation = src["annotation"]
        if annotation not in annotations:
            annotations[annotation] = read_annotation(Path(annotation))
        utterances = annotations[annotation]
        target = src["user_speaker"]
        user_end = float(src["user_end_sec"])

        next_user = next(
            (u.start for u in sorted(utterances, key=lambda x: x.start)
             if u.speaker.lower() == target.lower() and u.start > float(src["user_start_sec"])),
            None,
        )
        # 応答窓は次の User 発話では閉じない(gold_response の既定)。次の User
        # 発話の位置は参考として出すだけ。
        width = args.window_sec
        closed_by = f"上限 {args.window_sec:.0f}s"
        if next_user is not None:
            closed_by += f" / 次の User 発話は {next_user:.2f}s(窓は閉じない)"

        nearby = [
            u for u in sorted(utterances, key=lambda x: x.start)
            if u.speaker.lower() != target.lower()
            and u.end > user_end and u.start < user_end + args.window_sec
        ]

        shown += 1
        print(f"--- {meta['case_id']}  responded={responded}")
        print(f"    User  {src['user_start_sec']:8.2f}-{user_end:8.2f}  {clip(src.get('user_text', ''))}")
        print(f"    窓    幅 {width:6.2f}s  閉じた理由: {closed_by}")
        if nearby:
            for u in nearby:
                mark = "  <- 窓内" if u.start < user_end + width else "  <- 窓の外"
                print(f"    Staff {u.start:8.2f}-{u.end:8.2f}  {clip(u.text)}{mark}")
        else:
            print(f"    Staff  後続 {args.window_sec:.0f}s に発話なし(本当に無応答)")
        print()

    print(f"表示 {shown} ケース / responded=false は {silent} ケース")


if __name__ == "__main__":
    main()
