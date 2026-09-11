#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, TextIO


BATCH_ENV_MARKERS = (
    "PBS_JOBID",
    "PBS_JOBNAME",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "LSB_JOBID",
)


def guard_local_only(allow_server: bool) -> None:
    if allow_server:
        return
    markers = [name for name in BATCH_ENV_MARKERS if os.environ.get(name)]
    if markers:
        joined = ", ".join(markers)
        raise SystemExit(
            "Refusing to call OpenAI/Azure OpenAI from a batch/server environment "
            f"({joined} is set). Copy the packed JSONL to the local PC and run "
            "the judge there. Pass --allow-server only if you really intend this."
        )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(path: Path) -> set[str]:
    """Return durable row IDs from a prior append-only judge output."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in iter_jsonl(path):
        value = row.get("id")
        if value is not None:
            ids.add(str(value))
    return ids


def open_jsonl_append(path: Path, resume: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a" if resume else "w", encoding="utf-8")


def append_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    """Persist each completed API call before requesting the next one."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_client(provider: str, model: str | None):
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The local judge requires the OpenAI Python SDK. Install it on the "
            "local PC with: pip install openai"
        ) from exc

    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if not api_key or not endpoint or not deployment:
            raise SystemExit(
                "Azure judge requires AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, "
                "and AZURE_OPENAI_DEPLOYMENT, unless --model is provided."
            )
        endpoint = endpoint.rstrip("/")
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version or "2024-12-01-preview",
        )
        return client, deployment

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        openai_model = model or os.environ.get("OPENAI_MODEL")
        if not api_key or not openai_model:
            raise SystemExit("OpenAI judge requires OPENAI_API_KEY and OPENAI_MODEL or --model.")
        return OpenAI(api_key=api_key), openai_model

    raise ValueError(f"Unknown provider: {provider}")


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Judge did not return a JSON object: {text[:200]}")
    return json.loads(match.group(0))


def _response_format_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "response_format",
            "json_object",
            "unsupported parameter",
            "unknown parameter",
        )
    )


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class UsageTracker:
    """Accumulates OpenAI/Azure token usage across a judge run so cost stays
    visible -- these API calls are billed per token, not free."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0

    def add(self, usage: dict[str, int]) -> None:
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.cached_prompt_tokens += usage.get("cached_prompt_tokens", 0)

    def summary(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def print_summary(self, label: str) -> None:
        s = self.summary()
        print(
            f"[{label}] token usage: calls={s['calls']} "
            f"prompt={s['prompt_tokens']} (cached={s['cached_prompt_tokens']}) "
            f"completion={s['completion_tokens']} total={s['total_tokens']}",
            file=sys.stderr,
        )


def chat_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    retry: int = 3,
    retry_sleep: float = 5.0,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Returns (parsed_json, usage) where usage has prompt/completion/cached/total token counts."""
    last_exc: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                if not _response_format_unsupported(exc):
                    raise
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            content = response.choices[0].message.content or "{}"
            parsed = extract_json_object(content)
            if validator is not None:
                validator(parsed)
            return parsed, _usage_dict(response)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retry:
                break
            print(f"[judge] retry {attempt}/{retry} after error: {exc}", file=sys.stderr)
            time.sleep(retry_sleep * attempt)
    assert last_exc is not None
    raise last_exc


def score_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)
