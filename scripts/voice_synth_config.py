#!/usr/bin/env python3
"""
音声合成テスト(clone + CustomVoice)の YAML 設定ローダ。

2つの使われ方をする:
  - clone_voice_examples.py が --config で読み込み、合成パラメータの既定値にする
    (CLI フラグが最優先。config がそれに次ぐ。ハードコード既定が最後)。
  - run_clone_voice_examples.pbs が --shell / --prefetch で使い、proxy の export・
    シャード数など orchestration 値の取り出し・モデルの事前DLを行う。

config は configs/voice_synth.example.yaml をコピーして編集する運用
(実体は .gitignore 済み)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# YAML キー -> clone_voice_examples.py の argparse dest。ここにあるキーだけを
# 合成側の既定値に反映する(proxy / cuda_visible_devices など orchestration 用の
# キーは合成側には渡さない)。
_ARG_KEYS = {
    "analysis_dir", "ref_dir", "transcripts", "speaker", "out_dir", "examples_file", "language", "dtype",
    "attn_impl", "device", "max_new_tokens", "gen_batch_size", "num_refs",
    "modes", "xvector_all", "xvector_max_sec", "xvector_gap_sec",
    "min_ref_sec", "max_ref_sec", "ref_wav", "ref_text", "model",
    "clone_enabled", "customvoice", "customvoice_model", "customvoice_speaker",
    "customvoice_instruct", "num_shards",
}

DEFAULT_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CUSTOMVOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"config が見つかりません: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config はマッピングである必要があります: {p}")
    return data


def arg_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """config -> argparse 既定値。空文字/None は「未指定」とみなし飛ばす
    (本来の既定を空で上書きしないため)。"""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if key not in _ARG_KEYS or value in ("", None):
            continue
        if key == "modes" and isinstance(value, list):
            value = ",".join(str(v) for v in value)
        out[key] = value
    return out


def _sh_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def emit_shell(cfg: dict[str, Any]) -> str:
    """proxy と orchestration 値を shell export として出力(PBS が eval する)。"""
    lines: list[str] = []
    proxy = cfg.get("proxy") or {}
    for key in ("http_proxy", "https_proxy", "no_proxy"):
        val = proxy.get(key) if isinstance(proxy, dict) else None
        if val:
            lines.append(f"export {key}={_sh_quote(val)}")
            lines.append(f"export {key.upper()}={_sh_quote(val)}")
    for key, envname in (
        ("num_shards", "CFG_NUM_SHARDS"),
        ("cuda_visible_devices", "CFG_CUDA_VISIBLE_DEVICES"),
        ("out_dir", "CFG_OUT_DIR"),
    ):
        val = cfg.get(key)
        if isinstance(val, list):
            val = ",".join(str(v) for v in val)
        if val not in (None, ""):
            lines.append(f"export {envname}={_sh_quote(val)}")
    return "\n".join(lines)


def prefetch_models(cfg: dict[str, Any]) -> None:
    """config で使うモデルを事前 DL(並列シャードの同時初回DL競合を回避)。"""
    try:
        import qwen_tts  # noqa: F401
    except Exception as exc:  # pragma: no cover - 実機のみ
        raise SystemExit(f"qwen_tts import failed before launch: {exc}")
    from huggingface_hub import snapshot_download

    repos: list[str] = []
    if cfg.get("clone_enabled", True):
        repos.append(cfg.get("model") or DEFAULT_CLONE_MODEL)
    if cfg.get("customvoice", True):
        repos.append(cfg.get("customvoice_model") or DEFAULT_CUSTOMVOICE_MODEL)
    for repo in repos:
        print(f"prefetch {repo} ...", flush=True)
        snapshot_download(repo)
    print("preflight ok", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="YAML 設定ファイル")
    ap.add_argument("--shell", action="store_true",
                    help="proxy + orchestration の shell export を出力")
    ap.add_argument("--prefetch", action="store_true",
                    help="config で使うモデルを事前ダウンロード")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.shell:
        print(emit_shell(cfg))
    elif args.prefetch:
        prefetch_models(cfg)
    else:
        for key, value in arg_defaults(cfg).items():
            print(f"{key}={value}", file=sys.stderr)


if __name__ == "__main__":
    main()
