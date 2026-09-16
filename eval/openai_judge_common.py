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
from urllib.parse import unquote, urlsplit


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
        # Explicit rather than relying on the SDK's own OPENAI_BASE_URL lookup,
        # so the judge and the realtime evaluator provably share one endpoint.
        return OpenAI(api_key=api_key, base_url=openai_base_url()), openai_model

    raise ValueError(f"Unknown provider: {provider}")


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# The model id served for realtime, on api.openai.com and on the in-house
# gateway alike. Azure is the exception: there the name is a deployment the
# resource owner chose, so it has no default.
DEFAULT_REALTIME_MODEL = "gpt-realtime"


def openai_base_url() -> str:
    """Base URL for the OpenAI-compatible surface, judge and realtime alike.

    An in-house gateway such as https://api.rdg-genai.crl.hitachi.co.jp/v1 is
    OpenAI-compatible rather than Azure-shaped, so it is configured here and not
    through AZURE_OPENAI_ENDPOINT.  The OpenAI SDK would read OPENAI_BASE_URL by
    itself, but resolving it here keeps the judge and the realtime WebSocket --
    which has no SDK to read it -- pointed at one endpoint.
    """
    raw = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not raw:
        return DEFAULT_OPENAI_BASE_URL
    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw


def _to_websocket_scheme(url: str) -> str:
    """Rewrite an http(s) endpoint to its ws(s) form, leaving any base path."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    if url.startswith(("ws://", "wss://")):
        return url
    return "wss://" + url


def _extra_realtime_headers() -> list[str]:
    """Headers a corporate gateway may require on top of authentication.

    Read from OPENAI_REALTIME_EXTRA_HEADERS as a JSON object, e.g.
    '{"X-Tenant-Id": "abc"}'.  Routing or cost-centre headers are site-specific,
    so they are configuration rather than code.
    """
    raw = os.environ.get("OPENAI_REALTIME_EXTRA_HEADERS")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"OPENAI_REALTIME_EXTRA_HEADERS is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("OPENAI_REALTIME_EXTRA_HEADERS must be a JSON object of header names to values.")
    return [f"{name}: {value}" for name, value in parsed.items()]


def realtime_proxy_options(url: str) -> dict[str, Any]:
    """Proxy settings for websocket-client, which ignores HTTPS_PROXY itself.

    urllib and the OpenAI SDK pick the proxy up from the environment on their
    own, so without this the judge would reach a corporate proxy and the
    realtime socket would not -- the same environment behaving two ways.
    """
    target = urlsplit(url).hostname or ""
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for entry in (item.strip().lstrip(".").lower() for item in no_proxy.split(",")):
        if entry and (entry == "*" or target.lower() == entry or target.lower().endswith("." + entry)):
            return {}
    raw = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )
    if not raw:
        return {}
    parsed = urlsplit(raw if "://" in raw else f"http://{raw}")
    if not parsed.hostname:
        return {}
    options: dict[str, Any] = {
        "http_proxy_host": parsed.hostname,
        "http_proxy_port": parsed.port or (443 if parsed.scheme == "https" else 80),
    }
    if parsed.username:
        options["http_proxy_auth"] = (unquote(parsed.username), unquote(parsed.password or ""))
    return options


def resolve_provider(provider: str) -> str:
    """Pick the API surface from the environment when --provider is 'auto'.

    AZURE_OPENAI_ENDPOINT is Azure-shaped (/openai/deployments/...) while
    OPENAI_BASE_URL is OpenAI-compatible (.../v1); they are not interchangeable,
    so the one that is configured decides.
    """
    if provider != "auto":
        return provider
    azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    openai_compatible = bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"))
    if azure and openai_compatible:
        raise SystemExit(
            "Both AZURE_OPENAI_ENDPOINT and OPENAI_BASE_URL are set; pass --provider "
            "azure or --provider openai to say which endpoint to use."
        )
    if openai_compatible:
        return "openai"
    if azure:
        return "azure"
    raise SystemExit(
        "Set OPENAI_BASE_URL (OpenAI-compatible gateway, e.g. "
        "https://api.rdg-genai.crl.hitachi.co.jp/v1) with OPENAI_API_KEY, or "
        "AZURE_OPENAI_ENDPOINT with AZURE_OPENAI_KEY."
    )


def load_realtime_connection(provider: str, model: str | None) -> tuple[str, list[str], str]:
    """Return (url, websocket headers, resolved model/deployment) for the Realtime API.

    The Realtime API is a WebSocket, so it cannot go through the AzureOpenAI
    client used by the judges -- but it must read the same environment, because
    it is the same API subscription.  Keeping the variable names identical to
    load_client() is the point: one set of credentials configures both.
    """
    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        # The judge deployment is a text model, so a realtime deployment name
        # is read first; AZURE_OPENAI_DEPLOYMENT is only the last resort, for
        # resources where the two happen to be the same deployment.
        deployment = (
            model
            or os.environ.get("AZURE_OPENAI_REALTIME_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        )
        if not api_key or not endpoint or not deployment:
            raise SystemExit(
                "Azure Realtime requires AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and "
                "AZURE_OPENAI_REALTIME_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT), unless "
                "--model is provided."
            )
        # An in-house gateway may publish the realtime socket at a path that
        # neither Azure shape below can produce. AZURE_OPENAI_REALTIME_URL takes
        # the whole URL verbatim so such a deployment does not need a code
        # change; {deployment} in it is filled in.
        override = os.environ.get("AZURE_OPENAI_REALTIME_URL")
        if override:
            url = _to_websocket_scheme(override.replace("{deployment}", deployment))
        else:
            # Base paths are preserved, so a gateway mounted at
            # https://gw.example/azure-openai works the same way it does for the
            # judge's AzureOpenAI(azure_endpoint=...) client.
            host = _to_websocket_scheme(endpoint.rstrip("/"))
            # Deliberately not falling back to AZURE_OPENAI_API_VERSION: the
            # judge's dated chat-completions version does not necessarily serve
            # /realtime, and inheriting it would turn a working judge setup into
            # a 404 here.
            api_version = os.environ.get("AZURE_OPENAI_REALTIME_API_VERSION", "v1")
            if api_version == "v1":
                # The v1 surface takes the GA session schema this evaluator sends.
                url = f"{host}/openai/v1/realtime?model={deployment}"
            else:
                url = f"{host}/openai/realtime?api-version={api_version}&deployment={deployment}"
        # Azure authenticates the WebSocket with api-key rather than a bearer
        # token, but a gateway in front of it may want a different header name
        # (Authorization, Ocp-Apim-Subscription-Key, ...), so the name is
        # configurable while the value stays the one key the judge also uses.
        auth_header = os.environ.get("AZURE_OPENAI_REALTIME_AUTH_HEADER", "api-key")
        value = f"Bearer {api_key}" if auth_header.lower() == "authorization" else api_key
        headers = [f"{auth_header}: {value}"] + _extra_realtime_headers()
        return url, headers, deployment

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        # Not falling back to OPENAI_MODEL: that is the judge's text model, and
        # silently judging with it here would fail in a confusing way.
        openai_model = model or os.environ.get("OPENAI_REALTIME_MODEL") or DEFAULT_REALTIME_MODEL
        if not api_key:
            raise SystemExit("OpenAI Realtime requires OPENAI_API_KEY.")
        override = os.environ.get("OPENAI_REALTIME_URL")
        if override:
            url = _to_websocket_scheme(override.replace("{model}", openai_model))
        else:
            # Same base URL as the judge: an OpenAI-compatible gateway serves
            # the realtime socket next to chat/completions under /v1.
            url = f"{_to_websocket_scheme(openai_base_url())}/realtime?model={openai_model}"
        auth_header = os.environ.get("OPENAI_REALTIME_AUTH_HEADER", "Authorization")
        value = f"Bearer {api_key}" if auth_header.lower() == "authorization" else api_key
        return url, [f"{auth_header}: {value}"] + _extra_realtime_headers(), openai_model

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
