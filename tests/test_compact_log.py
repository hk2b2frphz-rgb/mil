"""scripts/compact_log.py のテスト。

full FT の pbs_logs が開けない大きさになっていた対処。圧縮しても
「何が起きたか」が読めなくならないことを確かめる。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compact_log.py"


def run(stdin_text: str, raw_path: Path | None = None) -> str:
    args = [sys.executable, "-u", str(SCRIPT)]
    if raw_path is not None:
        args.append(str(raw_path))
    result = subprocess.run(
        args, input=stdin_text.encode("utf-8"),
        capture_output=True, check=True,
    )
    return result.stdout.decode("utf-8")


def test_carriage_return_progress_is_collapsed(tmp_path: Path) -> None:
    # tqdm は端末でないと 1 イテレーションごとに書き出す。これがログ肥大の主因。
    progress = "".join(f"\r {i}/1000" for i in range(1, 1001))
    out = run(f"start\n{progress}\ndone\n")

    assert "start" in out
    assert "done" in out
    # 最終状態は残ること(進捗が消えてしまっては困る)。
    assert "1000/1000" in out
    # 中間状態は残らないこと。
    assert "500/1000" not in out
    assert "畳みました" in out


def test_repeated_lines_are_folded_with_count() -> None:
    out = run("head\n" + "same warning\n" * 5000 + "tail\n")

    assert out.count("same warning") == 1
    # 何回繰り返されたかは失わない。
    assert "4999" in out
    assert "head" in out
    assert "tail" in out


def test_small_repeats_are_kept_verbatim() -> None:
    # 2 回程度の繰り返しまで畳むと、普通のログが読みにくくなる。
    out = run("x\nx\ny\n")
    assert out.count("x") == 2
    assert "繰り返されました" not in out


def test_distinct_lines_are_untouched() -> None:
    # 圧縮対象でない行は 1 文字も変えない。障害調査は本文をそのまま読むため。
    body = "".join(f"step {i} loss {i / 10:.3f}\n" for i in range(1, 21))
    out = run(body)
    for i in range(1, 21):
        assert f"step {i} loss {i / 10:.3f}" in out


def test_raw_log_keeps_everything(tmp_path: Path) -> None:
    raw = tmp_path / "job.raw.log"
    noisy = "".join(f"\r {i}/300" for i in range(1, 301))
    stdin_text = f"start\n{noisy}\n" + "dup\n" * 100 + "end\n"

    out = run(stdin_text, raw_path=raw)

    # バイトで読む。read_text は \r を \n に潰すので、進捗が保全されている
    # ことを確かめられない。
    raw_text = raw.read_bytes().decode("utf-8")
    # 生ログ側は一切間引かれていないこと。
    assert raw_text == stdin_text
    assert "150/300" in raw_text
    assert raw_text.count("dup") == 100

    # 圧縮側は小さく、かつ生ログの場所を案内していること。
    assert len(out) < len(stdin_text) / 5
    assert str(raw) in out


def test_raw_log_appends_across_runs(tmp_path: Path) -> None:
    # ジョブ再投入で前回分を消さないこと。
    raw = tmp_path / "job.raw.log"
    run("first\n", raw_path=raw)
    run("second\n", raw_path=raw)
    raw_text = raw.read_text(encoding="utf-8")
    assert "first" in raw_text
    assert "second" in raw_text
