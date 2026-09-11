#!/usr/bin/env python3
"""ジョブログを「開けるサイズ」に圧縮しながら中継するフィルタ。

標準入力をそのまま生ログへ書き出しつつ、標準出力へは圧縮版を流す。
PBS スクリプトから次のように挟んで使う。

    exec > >(python3 -u scripts/compact_log.py "$RAW_LOG" | tee -a "$LOG") 2>&1

こうすると
  - $RAW_LOG          全部入り。障害調査はこちらを見る
  - $LOG と PBS の .o  圧縮版。人間が開くのはこちら
になる。

圧縮は 2 つだけ。どちらも「情報を捨てない」ことを優先している。

1. キャリッジリターンによる進捗の上書きを畳む
   tqdm などは出力が端末でないと 1 イテレーションごとに書き出す。端末なら
   同じ行を上書きして見えないが、ファイルに落ちると全部残る。full FT の
   ログが開けなくなる主因はこれ。\\r で区切られた中間状態は捨て、各行の
   最終状態だけを残す。

2. 連続する同一行をまとめる
   同じ警告が何万行も続くケースを 1 行 + 反復回数に畳む。畳んだ事実は
   必ず出力するので、繰り返しが起きていたこと自体は失われない。

行の中身は書き換えない。フィルタしたりレベルで落としたりもしない。
「実際のエラー本文を読む」ために生ログを別途残しているので、圧縮版で
判断がつかないときは $RAW_LOG を見ればよい。
"""

from __future__ import annotations

import sys
from pathlib import Path

# これ以上同じ行が続いたらまとめる。2 だと普通の繰り返しまで畳んでしまうので
# 少し余裕を持たせる。
REPEAT_THRESHOLD = 3


def main() -> int:
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raw = None
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # 追記。ジョブが再投入されても前の分を消さない。
        raw = raw_path.open("ab")

    out = sys.stdout.buffer
    stdin = sys.stdin.buffer

    pending: bytes | None = None
    repeat = 0
    dropped_progress = 0

    def flush_pending() -> None:
        nonlocal pending, repeat
        if pending is None:
            return
        out.write(pending + b"\n")
        if repeat >= REPEAT_THRESHOLD:
            note = f"[compact-log] 直前の行がさらに {repeat - 1} 回繰り返されました"
            out.write(note.encode("utf-8", "replace") + b"\n")
        elif repeat > 1:
            for _ in range(repeat - 1):
                out.write(pending + b"\n")
        out.flush()
        pending = None
        repeat = 0

    buffer = b""
    while True:
        chunk = stdin.readline()
        if not chunk:
            break
        if raw is not None:
            raw.write(chunk)
            raw.flush()

        buffer = chunk.rstrip(b"\n")

        # \r で上書きされた中間状態を捨て、最後の状態だけを採る。
        if b"\r" in buffer:
            parts = [p for p in buffer.split(b"\r") if p.strip()]
            dropped = max(0, len(parts) - 1)
            dropped_progress += dropped
            buffer = parts[-1] if parts else b""
            if not buffer:
                continue

        if buffer == pending:
            repeat += 1
            continue

        flush_pending()
        pending = buffer
        repeat = 1

    flush_pending()

    if dropped_progress:
        note = (
            f"[compact-log] 進捗表示の中間状態を {dropped_progress} 件畳みました"
        )
        out.write(note.encode("utf-8", "replace") + b"\n")
    if raw_path is not None:
        note = f"[compact-log] 全文ログ: {raw_path}"
        out.write(note.encode("utf-8", "replace") + b"\n")
    out.flush()

    if raw is not None:
        raw.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # 下流の tee が先に落ちても、学習側を巻き込まない。
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
