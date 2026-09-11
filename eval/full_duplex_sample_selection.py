from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def select_samples(
    samples: Iterable[dict[str, Any]],
    selected_tasks: set[str] | None,
    cases_per_task: int | None,
) -> list[dict[str, Any]]:
    """Filter tasks and deterministically keep the first N cases per task."""
    if cases_per_task is not None and cases_per_task < 1:
        raise ValueError("cases_per_task must be >= 1")

    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for sample in samples:
        task = str(sample["task"])
        if selected_tasks is not None and task not in selected_tasks:
            continue
        if cases_per_task is not None and counts[task] >= cases_per_task:
            continue
        selected.append(sample)
        counts[task] += 1
    return selected
