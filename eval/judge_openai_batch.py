#!/usr/bin/env python3
"""Asynchronous OpenAI Batch API path for ``judge_openai.py`` inputs.

Azure deployments intentionally stay on the synchronous local-judge path.
This CLI has explicit submit/status/collect phases so an interrupted local
machine never loses the server-side batch identifier or completed responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from judge_openai import SYSTEM_PROMPT, build_user_prompt, normalize_judge, validate_judge
from openai_judge_common import append_jsonl_row, completed_ids, extract_json_object, guard_local_only, iter_jsonl, open_jsonl_append


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["submit", "status", "collect"], required=True)
    parser.add_argument("--provider", choices=["openai", "azure"], default="openai")
    parser.add_argument("--input", type=Path, help="Source response JSONL; required for submit.")
    parser.add_argument("--out", type=Path, help="Judged JSONL output; required for submit/collect.")
    parser.add_argument("--state", required=True, type=Path, help="Durable batch state JSON.")
    parser.add_argument("--model", help="OpenAI model; required for submit.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    return parser.parse_args()


def client(provider: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK on the local PC: pip install openai") from exc
    import os

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is required for OpenAI Batch API judging.")
        return OpenAI(api_key=key)
    key = os.environ.get("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        raise SystemExit("Azure Batch requires AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.")
    # Azure Batch uses the OpenAI-compatible v1 client surface, but the base
    # URL belongs to the Azure resource and --model is the deployment name.
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=key, base_url=base_url)


def custom_id(index: int, row: dict[str, Any]) -> str:
    source_id = str(row.get("id") or f"row-{index}")
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return f"judge-{index:06d}-{digest}"


def submit(args: argparse.Namespace) -> int:
    if not args.input or not args.out or not args.model:
        raise SystemExit("submit requires --input, --out, and --model.")
    rows = list(iter_jsonl(args.input))
    if not rows:
        raise SystemExit("Input JSONL contains no rows.")
    requests_path = args.state.with_suffix(".requests.jsonl")
    mapping: dict[str, str] = {}
    with requests_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            request_id = custom_id(index, row)
            mapping[request_id] = str(row.get("id") or "")
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions" if args.provider == "openai" else "/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    api = client(args.provider)
    with requests_path.open("rb") as handle:
        uploaded = api.files.create(file=handle, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions" if args.provider == "openai" else "/chat/completions",
        completion_window="24h",
        metadata={"purpose": "miltoka_local_judge", "rows": str(len(rows))},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "input": str(args.input.resolve()),
        "out": str(args.out.resolve()),
        "model": args.model,
        "provider": args.provider,
        "request_ids": mapping,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[openai-batch] submitted {len(rows)} requests: {batch.id}")
    return 0


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Batch state not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def status(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    batch = client(state.get("provider", args.provider)).batches.retrieve(state["batch_id"])
    print(json.dumps({"id": batch.id, "status": batch.status, "request_counts": batch.request_counts}, default=str))
    return 0 if batch.status == "completed" else 2


def content_text(file_response: Any) -> str:
    if hasattr(file_response, "text"):
        return str(file_response.text)
    if hasattr(file_response, "read"):
        value = file_response.read()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if hasattr(file_response, "content"):
        value = file_response.content
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    raise TypeError("Unsupported OpenAI file content response.")


def usage_from_body(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_prompt_tokens": int(details.get("cached_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def collect(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    output = Path(args.out or state["out"])
    source = {str(row.get("id") or ""): row for row in iter_jsonl(Path(state["input"]))}
    provider = state.get("provider", args.provider)
    api = client(provider)
    batch = api.batches.retrieve(state["batch_id"])
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch {batch.id} is not ready (status={batch.status}).")
    done = completed_ids(output)
    failures: list[dict[str, Any]] = []
    written = 0
    with open_jsonl_append(output, bool(done)) as handle:
        for line in content_text(api.files.content(batch.output_file_id)).splitlines():
            result = json.loads(line)
            source_id = state["request_ids"].get(result.get("custom_id"))
            row = source.get(str(source_id))
            if row is None or str(source_id) in done:
                continue
            response = result.get("response") or {}
            body = response.get("body") or {}
            if result.get("error") or response.get("status_code") != 200:
                failures.append({"id": source_id, "batch_result": result})
                continue
            content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}")
            raw = extract_json_object(content)
            validate_judge(raw)
            judged = dict(row)
            judged["judge"] = {
                "provider": f"{provider}_batch",
                "model": state["model"],
                **normalize_judge(raw),
                "usage": usage_from_body(body),
                "batch_id": batch.id,
            }
            append_jsonl_row(handle, judged)
            written += 1
    error_path = output.with_suffix(output.suffix + ".batch_errors.json")
    error_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[openai-batch] collected {written} rows; failures={len(failures)} -> {output}")
    return 0 if not failures else 1


def main() -> int:
    args = parse_args()
    guard_local_only(False)
    if args.action == "submit":
        return submit(args)
    if args.action == "status":
        return status(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
