#!/usr/bin/env python3
"""
実験パイプラインを1つの YAML で管理し、ステージを「自己連鎖」で回すドライバ。

各ステージは独立した PBS ジョブとして走り、**自分の処理が成功したら末尾で次段を
qsub する**(`next:`)。中央の司令塔ジョブを持たないので、常に走っているのは
「今のステージ」だけ = 無駄な待機リソースが無い。依存(-W depend)も使わない。

このスクリプトの2つの顔:
  - `--start` / `--start-at <stage>` : 最初(または指定)のステージを qsub して
    連鎖を開始する。一瞬で終わるのでログインノードで実行してよい。
  - `--run-stage <stage>`            : ステージjob(_dag_stage.pbs)の中から呼ばれ、
    本体を実行し、成功したら `next:` を qsub する。

使い方:
    # 連鎖の中身を確認(投入しない)
    uv run python scripts/run_experiment_dag.py --start --dry-run
    # dialogue から連鎖開始
    uv run python scripts/run_experiment_dag.py --start
    # eval だけ(次段を撃たない)
    uv run python scripts/run_experiment_dag.py --start-at eval --no-chain

config は引数 > $EXPERIMENT_CONFIG > 既定候補 から自動で読む。

YAML 概略:
    experiment: exp01
    out_root: data/runs/exp01
    proxy: {http_proxy: "", ...}
    start: dialogue                    # --start の起点(省略時は最初の stage)
    default_resources: {queue: xvn_s, select: "1:res=small", walltime: "24:00:00"}
    stages:
      - name: dialogue
        pbs: scripts/run_dialogues_qwen_1000.pbs
        resources: {queue: xvn_s, select: "1:res=small", walltime: "24:00:00"}
        env: {BATCH_ID: "${source_batch_id}"}
        next: tts
      - name: tts
        pbs: scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs
        resources: {select: "1:res=middle2", walltime: "24:00:00"}
        next: train
      - name: train  { ... next: eval }
      - name: eval   { ... }            # next 無し = 連鎖の終端
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_VAR_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")
_WRAPPER_PBS = "scripts/_dag_stage.pbs"

_CONFIG_CANDIDATES = (
    "configs/experiment.local.yaml",
    "configs/experiment.yaml",
    "experiments/experiment.yaml",
)


def discover_config(cli_config: str | None) -> Path:
    if cli_config:
        return Path(cli_config)
    env = os.environ.get("EXPERIMENT_CONFIG")
    if env:
        return Path(env)
    for cand in _CONFIG_CANDIDATES:
        if Path(cand).is_file():
            return Path(cand)
    raise SystemExit(
        "config を特定できません。引数で渡すか、EXPERIMENT_CONFIG を設定するか、"
        f"次のいずれかを用意してください: {', '.join(_CONFIG_CANDIDATES)}"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise SystemExit(f"config が見つかりません: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config はマッピングである必要があります: {path}")
    return data


def build_context(cfg: dict[str, Any]) -> dict[str, str]:
    ctx: dict[str, str] = {}
    for key in ("experiment", "out_root"):
        if cfg.get(key) not in (None, ""):
            ctx[key] = str(cfg[key])
    for key, value in (cfg.get("vars") or {}).items():
        if isinstance(value, (str, int, float)):
            ctx[key] = str(value)
    # paths: 各ステージの入出力先。${out_root} 等を先に展開して ctx に足す。
    # これにより env で ${dialogue_out} のように参照でき、途中ステージから
    # 開始しても前段の出力先を config から解決できる。
    for key, value in (cfg.get("paths") or {}).items():
        if isinstance(value, (str, int, float)):
            ctx[key] = interp(str(value), ctx)
    return ctx


def interp(value: str, ctx: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in ctx:
            raise SystemExit(f"${{{name}}} を展開できません(out_root / vars 未定義)")
        return ctx[name]

    return _VAR_RE.sub(repl, value)


def stages_by_name(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in (cfg.get("stages") or [])}


def start_stage_name(cfg: dict[str, Any], cli_start_at: str | None) -> str:
    if cli_start_at:
        return cli_start_at
    if cfg.get("start"):
        return str(cfg["start"])
    stages = cfg.get("stages") or []
    if not stages:
        raise SystemExit("stages が空です")
    return stages[0]["name"]


def resources_for(cfg: dict[str, Any], stage: dict[str, Any]) -> dict[str, str]:
    res = dict(cfg.get("default_resources") or {})
    res.update(stage.get("resources") or {})
    return {k: str(v) for k, v in res.items() if v not in (None, "")}


def stage_env(cfg: dict[str, Any], stage: dict[str, Any],
              ctx: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    proxy = cfg.get("proxy") or {}
    for key in ("http_proxy", "https_proxy", "no_proxy"):
        val = proxy.get(key) if isinstance(proxy, dict) else None
        if val:
            env[key] = str(val)
            env[key.upper()] = str(val)
    for key, value in (stage.get("env") or {}).items():
        if value is False or value is None:
            continue
        env[key] = "1" if value is True else interp(str(value), ctx)
    return env


def qsub_command(config_path: Path, name: str, res: dict[str, str]) -> list[str]:
    cmd = ["qsub", "-N", f"dag_{name}"]
    if res.get("queue"):
        cmd += ["-q", res["queue"]]
    if res.get("select"):
        cmd += ["-l", f"select={res['select']}"]
    if res.get("walltime"):
        cmd += ["-l", f"walltime={res['walltime']}"]
    cmd += ["-j", "oe", "-V",
            "-v", f"STAGE={name},CONFIG={config_path}",
            _WRAPPER_PBS]
    return cmd


def submit_stage(cfg: dict[str, Any], config_path: Path, name: str,
                 dry: bool) -> str:
    stage = stages_by_name(cfg).get(name)
    if stage is None:
        raise SystemExit(f"未定義のステージ: {name}")
    if not (stage.get("pbs") or stage.get("cmd")):
        raise SystemExit(f"ステージ {name} に pbs: も cmd: もありません")
    res = resources_for(cfg, stage)
    cmd = qsub_command(config_path, name, res)
    if dry:
        print(f"  qsub: {' '.join(cmd)}")
        return f"<dry:{name}>"
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"qsub 失敗 (exit {proc.returncode}): {proc.stderr.strip()}")
    job_id = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
    if not job_id:
        raise SystemExit(f"qsub が job id を返しませんでした: {proc.stdout!r}")
    print(f"  submitted {name}: {job_id}")
    return job_id


def print_chain(cfg: dict[str, Any], config_path: Path, start: str) -> None:
    by_name = stages_by_name(cfg)
    seen: set[str] = set()
    name: str | None = start
    n = 0
    while name:
        if name in seen:
            print(f"  [warn] 循環検出: {name} で停止")
            break
        seen.add(name)
        stage = by_name.get(name)
        if stage is None:
            raise SystemExit(f"未定義のステージ: {name}")
        n += 1
        res = resources_for(cfg, stage)
        target = stage.get("pbs") or f"cmd: {stage.get('cmd')}"
        print(f"  {n}. {name}  [{res.get('queue', '?')} {res.get('select', '?')} "
              f"{res.get('walltime', '?')}]  -> {target}")
        submit_stage(cfg, config_path, name, dry=True)
        name = stage.get("next")


def check_chain(cfg: dict[str, Any], ctx: dict[str, str], start: str) -> int:
    """連鎖に沿って各ステージの expect: ファイルの存在を検証する。

    ステージ完了後(特に PBS 自己連鎖の後)に「各ステージが通ったか」を確認する。
    全て揃えば 0、欠けがあれば 1。"""
    by_name = stages_by_name(cfg)
    name: str | None = start
    seen: set[str] = set()
    ok = 0
    miss = 0
    while name and name not in seen:
        seen.add(name)
        stage = by_name.get(name)
        if stage is None:
            raise SystemExit(f"未定義のステージ: {name}")
        expects = stage.get("expect") or []
        if not expects:
            print(f"  [skip] {name}: expect 未定義")
        for raw in expects:
            path = interp(str(raw), ctx)
            if Path(path).exists():
                print(f"  [ok]   {name}: {path}")
                ok += 1
            else:
                print(f"  [MISS] {name}: {path}")
                miss += 1
        name = stage.get("next")
    print(f"\n{'[CHECK OK]' if miss == 0 else '[CHECK FAILED]'} ok={ok} missing={miss}")
    return 0 if miss == 0 else 1


def run_stage(cfg: dict[str, Any], config_path: Path, name: str,
              ctx: dict[str, str], no_chain: bool, local: bool = False) -> int:
    """ステージ本体を実行し、成功したら next を進める。

    local=False: next を qsub(自己連鎖・本番)。
    local=True : next を同一プロセスで inline 実行(qsub 不要・smoke/単一ノード用)。
    """
    by_name = stages_by_name(cfg)
    stage = by_name.get(name)
    if stage is None:
        raise SystemExit(f"未定義のステージ: {name}")

    env = {**os.environ, **stage_env(cfg, stage, ctx)}
    shell = os.environ.get("DAG_SHELL", "bash")  # クラスタ/OS 差異用の逃げ道
    if stage.get("cmd"):
        cmd = [shell, "-c", interp(str(stage["cmd"]), ctx)]
    else:
        pbs = interp(str(stage["pbs"]), ctx)
        if not Path(pbs).is_file():
            raise SystemExit(f"ステージ {name} の PBS が見つかりません: {pbs}")
        cmd = [shell, pbs]

    print(f"===== stage {name}: 実行開始 =====", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        print(f"[{name}] FAIL (exit {rc}) -- next は進めません", flush=True)
        return rc
    print(f"[{name}] OK", flush=True)

    nxt = stage.get("next")
    if not nxt:
        print(f"[{name}] 連鎖の終端", flush=True)
        return 0
    if no_chain:
        print(f"[{name}] --no-chain のため next '{nxt}' は進めません", flush=True)
        return 0
    if local:
        return run_stage(cfg, config_path, nxt, ctx, no_chain, local=True)
    print(f"[{name}] 成功 → next '{nxt}' を投入します", flush=True)
    submit_stage(cfg, config_path, nxt, dry=False)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", default=None,
                    help="master 実験 YAML(省略時は EXPERIMENT_CONFIG / 既定候補を自動探索)")
    ap.add_argument("--config", dest="config_opt", default=None,
                    help="config を明示指定(位置引数と同義)")
    ap.add_argument("--start", action="store_true", help="連鎖を先頭から開始")
    ap.add_argument("--start-at", default=None, help="指定ステージから連鎖開始")
    ap.add_argument("--run-stage", default=None,
                    help="(内部用)ステージjob から本体を実行し next を投入")
    ap.add_argument("--no-chain", action="store_true",
                    help="next を進めない(単発実行)")
    ap.add_argument("--local", action="store_true",
                    help="qsub を使わず、連鎖を同一プロセスで inline 実行(smoke/単一ノード)")
    ap.add_argument("--dry-run", action="store_true",
                    help="start 時: 連鎖内容を表示するだけ(投入しない)")
    ap.add_argument("--check", action="store_true",
                    help="各ステージの expect: 成果物が揃っているか検証(連鎖後の確認用)")
    args = ap.parse_args()

    config_path = discover_config(args.config or args.config_opt).resolve()
    cfg = load_yaml(config_path)
    ctx = build_context(cfg)
    print(f"config: {config_path}")

    if args.run_stage:
        rc = run_stage(cfg, config_path, args.run_stage, ctx, args.no_chain, args.local)
        sys.exit(rc)

    # start / start-at
    start = start_stage_name(cfg, args.start_at)

    if args.check:
        print(f"check from: {start}\n")
        sys.exit(check_chain(cfg, ctx, start))

    mode = "DRY-RUN" if args.dry_run else ("LOCAL" if args.local else "SUBMIT")
    print(f"experiment: {cfg.get('experiment', '<none>')}")
    print(f"start: {start}   mode: {mode}\n")

    if args.dry_run:
        print("chain:")
        print_chain(cfg, config_path, start)
        return

    if args.local:
        rc = run_stage(cfg, config_path, start, ctx, args.no_chain, local=True)
        if rc == 0:
            print("\n[SMOKE/LOCAL OK] 全ステージ通過")
        else:
            print(f"\n[FAILED] exit {rc}")
        sys.exit(rc)

    if shutil.which("qsub") is None:
        raise SystemExit("qsub が PATH にありません。投入ノードで実行してください "
                         "(--dry-run なら qsub 不要)。")
    print("投入:")
    submit_stage(cfg, config_path, start, dry=False)
    print("\n以降は各ステージが成功するたびに次段を自動投入します。")
    print("進捗確認: qstat -u $USER")


if __name__ == "__main__":
    main()
