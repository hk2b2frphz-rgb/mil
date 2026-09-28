#!/usr/bin/env python3
"""eval loss が良い上位 K 個だけ残して、full-FT の ZeRO checkpoint を掃除する。

なぜ要るか:
    full-FT の checkpoint は 1 つで数十 GB ある。step ごとに貯めると walltime
    より先にディスクが尽きて、学習ではなく書き込みで落ちる。かといって
    --keep_best_only（trainer 側の機能）だと最良の 1 つと直近しか残らないので、
    その 1 つが壊れていたときに戻る先が無い。上位 K 個あれば、1 つ落としても
    次に戻れる。

残すもの:
    - 一番新しい step。次のジョブがここから再開するので、metric の良し悪しに
      関わらず必ず残す。まだ eval を通っていないので metric が無いことも多い。
    - metric がある step のうち、値の良い順に K 個。

削除の安全側:
    - 書き込み途中のディレクトリを消さないよう、更新が新しいものは触らない
      (--min-age-sec)。
    - 消せなかったものは warning を出して続ける。掃除の失敗で学習を落とさない。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_best_checkpoint import load_fullft_eval_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", required=True, type=Path,
                        help="step_<N>/ が並んでいるディレクトリ")
    parser.add_argument("--log-file", required=True, type=Path, nargs="+",
                        help="MILTO_METRICS 行を含む実験ログ。再開して走らせた"
                             "ぶんだけログが分かれるので、複数渡せる。渡し漏れ"
                             "ると、そのログにしか metric が無い step が"
                             "「metric 無し」と見なされて消える")
    parser.add_argument("--keep", type=int, default=3,
                        help="metric の良い順に残す数（既定 3）")
    parser.add_argument("--metric-key", default="loss/total")
    parser.add_argument("--tag", default="pytorch_model",
                        help="step_<N>/<tag>/ があるものだけを checkpoint と見なす")
    parser.add_argument("--min-age-sec", type=float, default=120.0,
                        help="これより新しい更新のディレクトリは触らない")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.keep < 1:
        parser.error("--keep must be >= 1")
    return args


def step_dirs(checkpoints_dir: Path, tag: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not checkpoints_dir.is_dir():
        return out
    for path in checkpoints_dir.glob("step_*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue
        if not (path / tag).exists():
            # 書き出し途中か、別物。触らない。
            continue
        out[step] = path
    return out


def keep_set(steps: dict[int, Path], metrics: dict[int, float], keep: int) -> set[int]:
    if not steps:
        return set()
    # 再開する先なので、metric の有無に関わらず最新は必ず残す。
    survivors = {max(steps)}
    scored = sorted(
        ((metrics[s], s) for s in steps if s in metrics),
        key=lambda pair: (pair[0], -pair[1]),
    )
    for _value, step in scored[:keep]:
        survivors.add(step)
    return survivors


def main() -> int:
    args = parse_args()
    steps = step_dirs(args.checkpoints_dir, args.tag)
    if not steps:
        print(f"[prune] no checkpoint under {args.checkpoints_dir}")
        return 0

    metrics: dict[int, float] = {}
    for log_file in args.log_file:
        if not log_file.is_file():
            continue
        for step, value in load_fullft_eval_points(log_file, args.metric_key):
            # 同じ step が複数回出たら最後の値を採る。
            metrics[step] = value

    if not metrics:
        # metric が 1 つも読めていない状態で消すと、最新以外を全部落とすことに
        # なる。ログの書式が変わった / まだ最初の eval が来ていないだけ、の
        # どちらでも起こるので、掃除はせず次の機会に回す。
        print(f"[prune] {[str(f) for f in args.log_file]} から "
              f"{args.metric_key} を 1 つも読めません。"
              "何も消さずに終わります。")
        return 0

    survivors = keep_set(steps, metrics, args.keep)
    now = time.time()
    removed = 0
    freed = 0

    for step in sorted(steps):
        if step in survivors:
            continue
        path = steps[step]
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age < args.min_age_sec:
            print(f"[prune] skip step_{step}: 更新が新しい ({age:.0f}s)")
            continue
        size = 0
        try:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except OSError:
            pass
        if args.dry_run:
            print(f"[prune] would remove step_{step} ({size / 1e9:.1f} GB)")
            continue
        try:
            shutil.rmtree(path)
        except OSError as exc:
            # 掃除の失敗で学習を止めない。
            print(f"[prune] WARN: could not remove {path}: {exc}", file=sys.stderr)
            continue
        removed += 1
        freed += size

    kept = sorted(survivors)
    print(f"[prune] kept {kept} / removed {removed} dir(s) / freed {freed / 1e9:.1f} GB")
    missing = [s for s in kept if s not in metrics]
    if missing:
        print(f"[prune] （うち metric 未記録は {missing}。最新を残す分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
