"""Full-Duplex-Bench-JA judge as a GRPO reward, served by local vLLM.

The judge runs behind vLLM's OpenAI-compatible server rather than in-process.
The previous in-process version called ``LLM(...)`` and freed it once per
segment; loading a 27B model takes minutes, so a run that judges every segment
never finished. A resident server on its own GPU is loaded once for the whole
job and answers over HTTP, which also keeps the judge's memory out of the
training process's allocator.

Prompts, validation and normalization come from scripts/grpo/judge_prompt.py,
which re-exports eval/judge_full_duplex_azure.py: the reward and the reported
Azure evaluation score the same rubric by construction.
"""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from scripts.grpo.judge_prompt import (
    SYSTEM_PROMPT,
    guided_json_schema,
    normalize,
    user_prompt,
    validate,
)
# judge_prompt puts eval/ on sys.path, so the Azure judge's own JSON salvage
# path is reused here rather than reimplemented.
from openai_judge_common import extract_json_object  # noqa: E402

# Returned for a row the judge could not score. Mid-scale on every axis with no
# flags set is the neutral choice: it neither rewards nor punishes the rollout,
# and after per-axis z-normalization within the group it pulls the advantage
# toward zero instead of inventing a gradient from a failed API call.
NEUTRAL_SCORE = 3.0


class JudgeUnavailable(RuntimeError):
    """The judge endpoint did not answer. Training must stop, not guess."""


class FullDuplexJudge:
    """Scores rollouts with the Full-Duplex-Bench-JA rubric via vLLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        concurrency: int = 8,
        retry: int = 3,
        retry_sleep: float = 2.0,
        guided_decoding: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)
        self.retry = retry
        self.retry_sleep = retry_sleep
        self.guided_decoding = guided_decoding
        self._client: Any = None
        self.failures = 0

    # -- lifecycle ---------------------------------------------------

    def wait_until_ready(self, timeout_sec: float = 1800.0, poll_sec: float = 10.0) -> None:
        """Block until the vLLM server answers /health.

        A 27B server takes minutes to load. Without this the first epoch dies
        on connection-refused after the training process has already paid for
        loading Moshi.
        """
        health = self.base_url.rsplit("/v1", 1)[0] + "/health"
        deadline = time.time() + timeout_sec
        last: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health, timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = exc
            time.sleep(poll_sec)
        raise JudgeUnavailable(
            f"Judge server at {health} was not ready within {timeout_sec:.0f}s: {last}"
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # -- scoring -----------------------------------------------------

    def _score_one(self, row: dict[str, Any]) -> dict[str, Any] | None:
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
        ]
        task = row.get("task")
        extra_body: dict[str, Any] = {}
        if self.guided_decoding:
            # vLLM can enforce the whole schema, not just "some JSON" as
            # Azure's json_object mode does, so a malformed reply cannot cost
            # a rollout its reward.
            extra_body["guided_json"] = guided_json_schema(task)

        last: Exception | None = None
        for attempt in range(1, self.retry + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body or None,
                )
                content = response.choices[0].message.content or "{}"
                parsed = extract_json_object(content)
                validate(parsed, task)
                return normalize(parsed)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < self.retry:
                    time.sleep(self.retry_sleep * attempt)
        self.failures += 1
        print(f"[grpo-judge] giving up on row {row.get('id')}: {last}", flush=True)
        return None

    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a group of rollouts, one request each, issued concurrently.

        The G rollouts of a group are independent, so they go out together;
        vLLM batches them server-side. Order is preserved so a result lines up
        with the rollout it came from.
        """
        if not rows:
            return []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(rows))) as pool:
            results = list(pool.map(self._score_one, rows))
        return [r if r is not None else neutral_judgement(row.get("task"))
                for r, row in zip(results, rows)]


def neutral_judgement(task: str | None) -> dict[str, Any]:
    from scripts.grpo.judge_prompt import FLAG_KEYS, SCORE_KEYS, action_required

    return {
        "scores": {key: NEUTRAL_SCORE for key in SCORE_KEYS},
        "flags": {key: False for key in FLAG_KEYS},
        "overlap_action": "C_UNKNOWN" if action_required(task) else None,
        "reason": "judge unavailable",
    }


def judge_from_config(cfg: Any) -> FullDuplexJudge:
    """Build the judge from a GRPOConfig, with env overrides for the PBS job."""
    base_url = os.environ.get("GRPO_JUDGE_BASE_URL") or cfg.judge_base_url
    model = os.environ.get("GRPO_JUDGE_MODEL") or cfg.judge_model
    return FullDuplexJudge(
        base_url=base_url,
        model=model,
        temperature=cfg.judge_temperature,
        max_tokens=cfg.judge_max_tokens,
        concurrency=cfg.judge_concurrency,
        guided_decoding=cfg.judge_guided_decoding,
    )
