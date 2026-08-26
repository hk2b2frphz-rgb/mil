from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alignment_words import (
    DEFAULT_MAX_WORD_CHARS,
    _segment_fallback,
    distribute_word_times,
    get_segmenter_name,
    segment_text,
    split_utterance_alignments,
)


def test_segment_text_is_lossless() -> None:
    text = "もしもし、こちら孤独孤立相談窓口になります。"
    words = segment_text(text)
    assert "".join(words) == text
    assert len(words) >= 3
    assert all(len(w) <= DEFAULT_MAX_WORD_CHARS for w in words)


def test_segment_fallback_is_lossless_and_chunked() -> None:
    text = "十五年もお一人で頑張ってこられて、お辛かったですね…。"
    words = _segment_fallback(text)
    assert "".join(words) == text


def test_punctuation_attaches_to_previous_word() -> None:
    words = segment_text("はい、そうです。")
    assert "".join(words) == "はい、そうです。"
    # 句読点だけの独立エントリが無いこと
    assert all(any(c not in "、。！？…" for c in w) for w in words)


def test_distribute_word_times_proportional_and_monotonic() -> None:
    words = ["ああ", "いいいい", "うう"]  # 2:4:2 文字
    timed = distribute_word_times(words, 10.0, 18.0)
    assert timed[0] == ("ああ", 10.0, 12.0)
    assert timed[1] == ("いいいい", 12.0, 16.0)
    assert timed[2] == ("うう", 16.0, 18.0)
    for (_, s, e) in timed:
        assert s < e


def test_split_utterance_alignments_expands_and_keeps_labels() -> None:
    alignments = [
        ["もしもし、こちら孤独孤立相談窓口になります。", [0.5, 4.5], "SPEAKER_MAIN"],
        ["最近ずっと眠れなくて。", [5.0, 8.0], "SPEAKER_USER"],
    ]
    out, stats = split_utterance_alignments(alignments)
    assert stats["utterances"] == 2
    assert stats["passthrough"] == 0
    assert len(out) == stats["words"] > 4
    assert all(entry[2] in {"SPEAKER_MAIN", "SPEAKER_USER"} for entry in out)
    # 発話区間の内側で単調・連続
    main = [e for e in out if e[2] == "SPEAKER_MAIN"]
    assert main[0][1][0] == 0.5
    assert main[-1][1][1] == 4.5
    for prev, cur in zip(main, main[1:]):
        assert prev[1][1] == cur[1][0]
    # テキストは無劣化
    assert "".join(e[0] for e in main) == alignments[0][0]


def test_split_utterance_alignments_passthrough_invalid() -> None:
    alignments = [
        ["", [0.0, 1.0], "SPEAKER_MAIN"],        # 空テキスト
        ["ゼロ長", [2.0, 2.0], "SPEAKER_MAIN"],   # end <= start
        "not-a-list",
    ]
    out, stats = split_utterance_alignments(alignments)
    assert stats["utterances"] == 0
    assert stats["passthrough"] == 3
    assert out == alignments


def _write_sidecar(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "alignments": [
                    ["もしもし、こちら相談窓口になります。", [0.5, 3.5], "SPEAKER_MAIN"],
                    ["眠れない日が続いていて。", [4.0, 7.0], "SPEAKER_USER"],
                ],
                "metadata": {"sample_rate": 24000},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_cli_converts_in_place_and_is_idempotent(tmp_path: Path) -> None:
    sidecar = tmp_path / "sample_001.json"
    _write_sidecar(sidecar)
    script = ROOT / "scripts" / "split_alignments_to_words.py"

    first = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "converted=1" in first.stdout

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["metadata"]["alignments_granularity"] == "word"
    assert len(data["alignments"]) > 2
    assert len(data["alignments_utterance"]) == 2
    joined = "".join(e[0] for e in data["alignments"] if e[2] == "SPEAKER_MAIN")
    assert joined == "もしもし、こちら相談窓口になります。"

    second = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "converted=0" in second.stdout
    assert "skipped(already word-level)=1" in second.stdout


def test_get_segmenter_name_reports_active_segmenter() -> None:
    name = get_segmenter_name()
    assert name in {"pyopenjtalk", "regex-chunk"}
    # キャッシュされても同じ値を返す
    assert get_segmenter_name() == name


def test_cli_require_pyopenjtalk_flag(tmp_path: Path) -> None:
    sidecar = tmp_path / "sample_003.json"
    _write_sidecar(sidecar)
    script = ROOT / "scripts" / "split_alignments_to_words.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path),
         "--require-pyopenjtalk", "--dry-run"],
        capture_output=True, text=True,
    )
    if get_segmenter_name() == "pyopenjtalk":
        assert result.returncode == 0
        assert "segmenter=pyopenjtalk" in result.stdout
    else:
        assert result.returncode == 2
        assert "pyopenjtalk" in result.stderr


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    sidecar = tmp_path / "sample_002.json"
    _write_sidecar(sidecar)
    before = sidecar.read_text(encoding="utf-8")
    script = ROOT / "scripts" / "split_alignments_to_words.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "mode=dry-run" in result.stdout
    assert sidecar.read_text(encoding="utf-8") == before
