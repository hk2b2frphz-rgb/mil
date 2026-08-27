#!/usr/bin/env python3
"""Minimal terminal chat against the HPC-hosted model, through the SSH tunnel.

Stdlib only -- no venv, no pip. Talks the OpenAI-compatible side of the LiteLLM
proxy (``/v1/chat/completions``) and streams tokens as they arrive.

    # after the tunnel is up
    python agent_hpc/local/chat.py

Connection settings are resolved in this order:
  1. command line flags
  2. environment (OPENAI_BASE_URL / OPENAI_API_KEY / ANTHROPIC_MODEL)
  3. agent_env.json written next to this script by connect.ps1
  4. http://127.0.0.1:8787/v1 with model "qwen-agent"

Commands inside the chat: /reset (forget history), /system <text>, /exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8787/v1"
DEFAULT_MODEL = "qwen-agent"
ENV_JSON = Path(__file__).with_name("agent_env.json")


def load_saved() -> dict:
    """Read the file connect.ps1 drops next to this script, if it is there."""
    try:
        return json.loads(ENV_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def post(url: str, api_key: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def stream_reply(base_url: str, api_key: str, model: str, messages: list,
                 sampling: dict, max_tokens: int, timeout: float) -> str:
    """Stream one assistant turn to stdout and return the full text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        **sampling,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    chunks: list[str] = []
    with post(f"{base_url}/chat/completions", api_key, payload, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Some servers stream the thinking trace separately; show it dimmed
            # rather than dropping it, so a stall is visibly the model thinking.
            reasoning = delta.get("reasoning_content")
            if reasoning:
                sys.stdout.write(f"\033[2m{reasoning}\033[0m")
                sys.stdout.flush()
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
    print()
    return "".join(chunks)


def check(base_url: str, api_key: str) -> int:
    url = f"{base_url}/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', 'replace')[:400]}")
        return 1
    except OSError as exc:
        print(f"cannot reach {url}: {exc}")
        print("is the SSH tunnel up? see agent_hpc/README.md")
        return 1
    names = [m.get("id") for m in data.get("data", [])]
    print(f"reachable: {url}")
    print("models:", ", ".join(n for n in names if n) or "(none reported)")
    return 0


def main() -> int:
    saved = load_saved()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", saved.get("openai_base_url", DEFAULT_BASE_URL)))
    parser.add_argument("--api-key", default=os.environ.get(
        "OPENAI_API_KEY", saved.get("api_key", "")))
    parser.add_argument("--model", default=os.environ.get(
        "ANTHROPIC_MODEL", saved.get("model_alias", DEFAULT_MODEL)))
    parser.add_argument("--system", default=None, help="system prompt")
    # Qwen3.6-27B's "precise coding" preset from its own model card. Its general
    # preset is temperature 1.0; coding wants the tighter one.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = server default")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--check", action="store_true",
                        help="just verify the tunnel and list served models")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.check:
        return check(base_url, args.api_key)

    if check(base_url, args.api_key) != 0:
        return 1

    # top_k is not in the OpenAI spec; vLLM honours it and the proxy drops it
    # if it cannot pass it through, so sending it costs nothing either way.
    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"model: {args.model}   (/reset, /system <text>, /exit)")
    print("-" * 60)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if user == "/reset":
            keep = [m for m in messages if m["role"] == "system"]
            messages = keep
            print("(history cleared)")
            continue
        if user.startswith("/system "):
            messages = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": user[len("/system "):]})
            print("(system prompt set)")
            continue

        messages.append({"role": "user", "content": user})
        print(f"\n{args.model}> ", end="", flush=True)
        try:
            reply = stream_reply(base_url, args.api_key, args.model, messages,
                                 sampling, args.max_tokens, args.timeout)
        except KeyboardInterrupt:
            print("\n(interrupted; dropping that turn)")
            messages.pop()
            continue
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            print(f"\nHTTP {exc.code}: {detail}")
            messages.pop()
            continue
        except OSError as exc:
            print(f"\nconnection error: {exc}")
            print("the tunnel or the PBS job may have died; check qstat on the HPC")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
