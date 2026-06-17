#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


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


def load_client(provider: str, model: str | None):
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The local judge requires the OpenAI Python SDK. Install it on the "
            "local PC with: pip install openai"
        ) from exc

    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if not api_key or not endpoint or not deployment:
            raise SystemExit(
                "Azure judge requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "and AZURE_OPENAI_DEPLOYMENT, unless --model is provided."
            )
        endpoint = endpoint.rstrip("/")
        if api_version:
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        else:
            base_url = endpoint if endpoint.endswith("/openai/v1") else f"{endpoint}/openai/v1/"
            client = OpenAI(api_key=api_key, base_url=base_url)
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


def chat_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    retry: int = 3,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
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
            except Exception:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            content = response.choices[0].message.content or "{}"
            return extract_json_object(content)
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
