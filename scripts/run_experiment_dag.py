#!/usr/bin/env python3
"""
実験全体を1つの YAML で管理し、ステージごとに qsub を依存付きで投入するドライバ。

master YAML でステージ(学習データ生成 / 学習 / 評価 …)を定義し、`run:` で
「どれを実行するか」を選ぶ。各ステージは別々の PBS ジョブとして qsub され、
`depends_on` に応じて `-W depend=afterok:<jobid>` で連鎖する。

このスクリプトは**投入ノード(ログインノード)で実行**する。qsub を呼ぶだけで
GPU は使わない。

使い方:
    # 何が投入されるか確認(投入しない)
    uv run python scripts/run_experiment_dag.py experiments/exp01.yaml --dry-run
    # 実際に投入
    uv run python scripts/run_experiment_dag.py experiments/exp01.yaml
    # 一部ステージだけ
    uv run python scripts/run_experiment_dag.py experiments/exp01.yaml --stages eval

YAML 概略:
    experiment: exp01
    out_root: data/runs/exp01
    proxy: {http_proxy: "", https_proxy: "", no_proxy: ""}
    run: [traindata, train, eval]          # 実行するステージ(省略時は全部)
    stages:
      - name: traindata
        pbs: scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs
        env: {OUT_ROOT: "${out_root}"}
      - name: train
        pbs: scripts/run_experiment.pbs
        depends_on: [traindata]
        env: {SRC_RUN_DIR: "${out_root}", BASE_EXP: lora_base_config}
      - name: eval
        pbs: scripts/run_full_duplex_eval.pbs
        depends_on: [train]
        env: {MODEL_ID: lora_h01, MODEL_WEIGHT: "${out_root}/checkpoints/consolidated.safetensors"}

env 値の型: 文字列/数値はそのまま文字列化。true は "1"、false/空は「その変数を渡さない」。
${name} は out_root / experiment / vars ブロックのスカラーを展開する。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_VAR_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise SystemExit(f"config が見つかりません: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config はマッピングである必要があります: {path}")
    return data


def build_context(cfg: dict[str, Any]) -> dict[str, str]:
    """${...} 展開に使うスカラー辞書。experiment / out_root / vars: を集める。"""
    ctx: dict[str, str] = {}
    for key in ("experiment", "out_root"):
        if cfg.get(key) not in (None, ""):
            ctx[key] = str(cfg[key])
    for key, value in (cfg.get("vars") or {}).items():
        if isinstance(value, (str, int, float)):
            ctx[key] = str(value)
    return ctx


def interp(value: str, ctx: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in ctx:
            raise SystemExit(f"${{{name}}} を展開できません(out_root / vars 未定義)")
        return ctx[name]

    return _VAR_RE.sub(repl, value)


def stage_env(stage: dict[str, Any], proxy: dict[str, Any],
              ctx: dict[str, str]) -> dict[str, str]:
    """このステージの qsub に渡す環境変数を組み立てる。"""
    env: dict[str, str] = {}
    # proxy を先に(ステージ env で上書き可能)。
    for key in ("http_proxy", "https_proxy", "no_proxy"):
        val = proxy.get(key) if isinstance(proxy, dict) else None
        if val:
            env[key] = str(val)
            env[key.upper()] = str(val)
    for key, value in (stage.get("env") or {}).items():
        if value is False or value is None:
            continue  # 渡さない = 未設定扱い
        if value is True:
            env[key] = "1"
        else:
            env[key] = interp(str(value), ctx)
    return env


def resolve_run_order(cfg: dict[str, Any], cli_stages: list[str] | None) -> list[str]:
    stages = cfg.get("stages") or []
    defined = [s["name"] for s in stages]
    if cli_stages:
        unknown = [s for s in cli_stages if s not in defined]
        if unknown:
            raise SystemExit(f"未定義のステージ: {unknown}(定義: {defined})")
        want = set(cli_stages)
    elif cfg.get("run"):
        unknown = [s for s in cfg["run"] if s not in defined]
        if unknown:
            raise SystemExit(f"run に未定義のステージ: {unknown}")
        want = set(cfg["run"])
    else:
        want = set(defined)
    # 定義順を維持して選択。
    return [name for name in defined if name in want]


def submit(qsub_cmd: list[str], env: dict[str, str]) -> str:
    full_env = {**os.environ, **env}
    proc = subprocess.run(qsub_cmd, env=full_env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"qsub 失敗 (exit {proc.returncode}):\n"
            f"  cmd: {' '.join(qsub_cmd)}\n  stderr: {proc.stderr.strip()}"
        )
    job_id = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
    if not job_id:
        raise SystemExit(f"qsub が job id を返しませんでした: {proc.stdout!r}")
    return job_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path, help="master 実験 YAML")
    ap.add_argument("--stages", default=None,
                    help="実行するステージをカンマ区切りで指定(config の run より優先)")
    ap.add_argument("--dry-run", action="store_true",
                    help="qsub を実行せず、投入内容だけ表示")
    ap.add_argument("--strict", action="store_true",
                    help="依存先が実行対象外のときエラーにする(既定は依存を落として続行)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ctx = build_context(cfg)
    proxy = cfg.get("proxy") or {}
    stages_by_name = {s["name"]: s for s in (cfg.get("stages") or [])}
    cli_stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    order = resolve_run_order(cfg, cli_stages)
    if not order:
        raise SystemExit("実行対象のステージがありません(run / --stages を確認)")

    if not args.dry_run and shutil.which("qsub") is None:
        raise SystemExit("qsub が PATH にありません。投入ノードで実行してください "
                         "(--dry-run なら qsub 不要)。")

    print(f"experiment: {cfg.get('experiment', '<none>')}")
    print(f"stages to run: {order}")
    print(f"mode: {'DRY-RUN' if args.dry_run else 'SUBMIT'}\n")

    submitted: dict[str, str] = {}
    for name in order:
        stage = stages_by_name[name]
        pbs = stage.get("pbs")
        if not pbs:
            raise SystemExit(f"ステージ {name} に pbs: がありません")
        pbs = interp(str(pbs), ctx)
        if not Path(pbs).is_file():
            raise SystemExit(f"ステージ {name} の PBS が見つかりません: {pbs}")

        # 依存: 今回の実行に含まれる依存先のみを afterok に入れる。
        dep_names = stage.get("depends_on") or []
        dep_ids: list[str] = []
        for dep in dep_names:
            if dep in submitted:
                dep_ids.append(submitted[dep])
            elif dep in order:
                # 実行対象だが未投入 = 定義順の矛盾。
                raise SystemExit(f"ステージ {name} の依存 {dep} が先に投入されていません")
            else:
                msg = f"ステージ {name} の依存 {dep} は実行対象外"
                if args.strict:
                    raise SystemExit(msg + "(--strict)")
                print(f"  [warn] {msg} → 依存を無視して投入します")

        env = stage_env(stage, proxy, ctx)
        qsub_cmd = ["qsub"]
        if dep_ids:
            qsub_cmd += ["-W", "depend=afterok:" + ":".join(dep_ids)]
        qsub_cmd.append(pbs)

        print(f"=== stage: {name} ===")
        print(f"  pbs:   {pbs}")
        if dep_ids:
            print(f"  after: {dep_ids}")
        if env:
            print("  env:")
            for k in sorted(env):
                print(f"    {k}={env[k]}")

        if args.dry_run:
            print(f"  cmd:   {' '.join(qsub_cmd)}   (dry-run, not submitted)\n")
            submitted[name] = f"<dry:{name}>"
        else:
            job_id = submit(qsub_cmd, env)
            submitted[name] = job_id
            print(f"  submitted: {job_id}\n")

    if not args.dry_run:
        out = {
            "experiment": cfg.get("experiment"),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "stages": submitted,
        }
        rec = args.config.parent / f"_submitted_{datetime.now():%Y%m%d_%H%M%S}.json"
        rec.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"投入記録: {rec}")
        print("進捗確認: qstat -u $USER")


if __name__ == "__main__":
    main()
