"""Single source for the Full-Duplex-Bench-JA judge prompt.

The GRPO reward and the reported evaluation must score the same thing, so this
module does not restate the rubric -- it imports the exact SYSTEM_PROMPT,
rubric, validator and normalizer that eval/judge_full_duplex_azure.py uses for
the Azure run. Editing the rubric there changes the reward here, which is the
point: a divergence between the two is a silent experiment bug.

eval/ modules import each other by flat module name (``from
full_duplex_judge_input import ...``), so eval/ has to be on sys.path before
they can be imported from outside that directory.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from full_duplex_judge_input import (  # noqa: E402
    ACTION_DEFINITIONS,
    ACTION_LABELS,
    action_required,
    compact_event_segments,
    event_interval,
)
from judge_full_duplex_azure import (  # noqa: E402
    FLAG_KEYS,
    FLAGS,
    RUBRIC,
    SCORE_KEYS,
    SYSTEM_PROMPT,
    build_judge_input,
    normalize,
    prompt as user_prompt,
    validate,
)

__all__ = [
    "ACTION_DEFINITIONS",
    "ACTION_LABELS",
    "FLAGS",
    "FLAG_KEYS",
    "RUBRIC",
    "SCORE_KEYS",
    "SYSTEM_PROMPT",
    "action_required",
    "build_judge_input",
    "compact_event_segments",
    "event_interval",
    "guided_json_schema",
    "normalize",
    "user_prompt",
    "validate",
]


def guided_json_schema(task: str | None) -> dict[str, Any]:
    """JSON schema for vLLM guided decoding, derived from the shared rubric.

    Azure enforces the shape with ``response_format={"type": "json_object"}``,
    which only guarantees *some* JSON. vLLM can enforce the full schema, so a
    local judge never wastes a rollout on a malformed reply. The schema is
    generated from SCORE_KEYS/FLAG_KEYS rather than written out, so it cannot
    drift from what validate() accepts.
    """
    action_values: list[Any] = list(ACTION_LABELS)
    if not action_required(task):
        action_values = [None]
    else:
        # validate() rejects null when the task needs an action, so the schema
        # must not offer it as an escape hatch.
        pass
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    key: {"type": "integer", "minimum": 1, "maximum": 5}
                    for key in SCORE_KEYS
                },
                "required": list(SCORE_KEYS),
                "additionalProperties": False,
            },
            "flags": {
                "type": "object",
                "properties": {key: {"type": "boolean"} for key in FLAG_KEYS},
                "required": list(FLAG_KEYS),
                "additionalProperties": False,
            },
            "overlap_action": {"enum": action_values},
            "reason": {"type": "string"},
        },
        "required": ["scores", "flags", "overlap_action", "reason"],
        "additionalProperties": False,
    }
