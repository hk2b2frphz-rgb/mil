"""Benchmark case schema and manifest I/O.

A *base item* is one slot-bearing user utterance (from MASSIVE ja-JP or the
underspecified template arm). A *case* is a base item rendered under one
corruption condition. The manifest is JSONL (one case per line) plus a
``manifest.json`` header recording the construction profile, so an
evaluation run can verify it is reading the dataset it thinks it is.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

PROFILE_NAME = "clarify_fd_ja"
PROFILE_VERSION = 1


@dataclass
class BaseItem:
    """One slot-bearing utterance before condition expansion."""

    base_id: str
    arm: str                    # "acoustic" | "underspecified"
    intent: str
    slot_type: str
    slot_value: str             # gold surface form; "" for underspecified
    utterance_text: str         # full user utterance (what is spoken)
    pre_text: str               # utterance split for span alignment:
    slot_text: str              #   utterance == pre + slot + post
    post_text: str
    repair_text: str            # scripted user repair if the model asks
    source: str = "massive_ja"  # provenance
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.arm == "acoustic":
            joined = self.pre_text + self.slot_text + self.post_text
            if joined != self.utterance_text:
                raise ValueError(
                    f"{self.base_id}: pre+slot+post != utterance "
                    f"({joined!r} vs {self.utterance_text!r})"
                )
            if not self.slot_text:
                raise ValueError(f"{self.base_id}: acoustic arm needs slot_text")
        elif self.arm == "underspecified":
            if self.slot_text:
                raise ValueError(
                    f"{self.base_id}: underspecified arm must not carry slot_text"
                )
        else:
            raise ValueError(f"{self.base_id}: unknown arm {self.arm!r}")
        if not self.repair_text:
            raise ValueError(f"{self.base_id}: repair_text required")


@dataclass
class BenchmarkCase:
    """One (base item x condition) evaluation input."""

    case_id: str
    base: BaseItem
    condition: str
    expected_behavior: str          # act | ask | graded
    audio_path: str                 # relpath under dataset dir (corrupted)
    clean_audio_path: str           # relpath (uncorrupted render)
    repair_audio_path: str          # relpath (pre-synthesized repair turn)
    span_start_sec: float | None    # slot span in the rendered audio
    span_end_sec: float | None
    sample_rate: int
    seed_base: int                  # per-case corruption seed
    tts: dict[str, Any] = field(default_factory=dict)
    asr_oracle: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["base"] = dataclasses.asdict(self.base)
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "BenchmarkCase":
        base = BaseItem(**d.pop("base"))
        return BenchmarkCase(base=base, **d)


def case_seed(base_id: str, condition: str) -> int:
    """Stable 32-bit seed for a case's corruption rng."""
    digest = hashlib.sha256(f"{base_id}/{condition}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def write_manifest(
    dataset_dir: Path,
    cases: list[BenchmarkCase],
    profile_extra: dict[str, Any] | None = None,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "n_cases": len(cases),
        "conditions": sorted({c.condition for c in cases}),
        "arms": sorted({c.base.arm for c in cases}),
    }
    if profile_extra:
        header.update(profile_extra)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(header, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (dataset_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_json(), ensure_ascii=False) + "\n")


def read_manifest(dataset_dir: Path) -> tuple[dict[str, Any], list[BenchmarkCase]]:
    header = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if header.get("profile") != PROFILE_NAME:
        raise ValueError(
            f"dataset at {dataset_dir} has profile {header.get('profile')!r}; "
            f"expected {PROFILE_NAME!r}"
        )
    cases = []
    with (dataset_dir / "cases.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(BenchmarkCase.from_json(json.loads(line)))
    if len(cases) != header.get("n_cases"):
        raise ValueError(
            f"manifest header says {header.get('n_cases')} cases but "
            f"cases.jsonl has {len(cases)}"
        )
    return header, cases


def iter_trials(
    cases: list[BenchmarkCase], seeds: list[int]
) -> Iterator[tuple["BenchmarkCase", int]]:
    for case in cases:
        for seed in seeds:
            yield case, seed
