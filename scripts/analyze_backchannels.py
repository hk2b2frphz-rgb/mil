#!/usr/bin/env python3
from __future__ import annotations
"""
Analyze LLM-jp-Moshi output for Japanese backchannels (相槌).

response_recorder.py で生成した out_dir 配下の各トライアル
(<input_stem>/seed_<n>/{meta.json, transcript.jsonl}) を走査し、
Moshi 側ストリームに出た相槌を検出・分類する。

分類:
  - listening : user 発話中 (time_sec < input_duration_sec) に出た相槌
                = 真の傾聴オーバーラップ相槌。これが出れば「会話音声を
                  プロンプトに与えて自然な相槌を作る」狙いが成立する。
  - response  : user 発話後 (time_sec >= input_duration_sec) に出た相槌
                = ターン冒頭の受け応え。

出力:
  - 標準出力にトライアル別 + 全体サマリ
  - out_dir/backchannel_report.json に集計を保存

これは合成データ品質の "検証" 用であって学習データそのものではない。
相槌が listening 区間に十分出ることを確認できたら、本番では
response_recorder の transcript/audio から該当区間を切り出して
miltoka の学習データに回す、という流れを想定している。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

# 代表的な日本語相槌。表記ゆれ・伸ばし・繰り返しを正規表現で吸収する。
# 文字列を連結したトークン列に対して検索するため、語境界は厳密にしない。
AIZUCHI_PATTERNS: list[tuple[str, str]] = [
    ("un", r"う+ん+"),
    ("hai", r"はい(?:はい)?"),
    ("ee", r"え+ぇ*|ええ"),
    ("sou", r"そう(?:そう|です(?:ね|よね)?|なん(?:です(?:ね)?|だ)?|なんですか)?|そっか|そうか"),
    ("naruhodo", r"なるほど(?:ね)?"),
    ("hee", r"へ[ぇえー]+|ほ[ーお]+"),
    ("fun", r"ふ[んー]+|ふんふん"),
    ("aa", r"あ[ぁあーっ]+|ああ"),
    ("wakaru", r"わかり(?:ます|ました)|わかる"),
    ("tashika", r"たしかに|確かに"),
    ("oo", r"お[おーっ]+"),
    ("warai", r"(?:あは|うふ|えへ|ふふ)+|笑"),
]

COMPILED = [(name, re.compile(pat)) for name, pat in AIZUCHI_PATTERNS]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_trial_dirs(out_dir: Path) -> list[Path]:
    """meta.json を持つディレクトリ = 1 トライアルとして収集する。"""
    return sorted(p.parent for p in out_dir.rglob("meta.json"))


def classify_events(
    events: list[dict[str, Any]],
    input_duration_sec: float,
) -> dict[str, Any]:
    """text_events 列を相槌マッチ + listening/response に分類する。"""
    # 連続トークンを 1 つの発話塊にまとめてから相槌判定する。
    # ピースをそのまま検索すると "そ"+"う" のように分割され取りこぼす。
    listening_hits: list[dict[str, Any]] = []
    response_hits: list[dict[str, Any]] = []

    # 時刻順に並べ、近接ピースを連結窓にまとめる
    events = sorted(events, key=lambda e: e.get("time_sec", 0.0))
    window_text = ""
    window_start: float | None = None
    last_t = 0.0

    def flush(end_t: float) -> None:
        nonlocal window_text, window_start
        text = window_text.strip()
        if not text or window_start is None:
            window_text = ""
            window_start = None
            return
        for name, pat in COMPILED:
            for m in pat.finditer(text):
                hit = {
                    "kind": name,
                    "surface": m.group(0),
                    "time_sec": round(window_start, 3),
                    "window_text": text,
                }
                if window_start < input_duration_sec:
                    listening_hits.append(hit)
                else:
                    response_hits.append(hit)
        window_text = ""
        window_start = None

    for ev in events:
        piece = str(ev.get("piece", ""))
        t = float(ev.get("time_sec", 0.0))
        if not piece.strip():
            # 空白/句切りでウィンドウを flush
            flush(last_t)
            last_t = t
            continue
        if window_start is None:
            window_start = t
        # 0.8s 以上間が空いたら別塊
        if t - last_t > 0.8:
            flush(last_t)
            window_start = t
        window_text += piece
        last_t = t
    flush(last_t)

    transcript = "".join(str(e.get("piece", "")) for e in events).strip()
    return {
        "listening": listening_hits,
        "response": response_hits,
        "n_listening": len(listening_hits),
        "n_response": len(response_hits),
        "transcript": transcript,
        "transcript_len": len(transcript),
    }


def analyze_trial(trial_dir: Path) -> dict[str, Any] | None:
    meta_path = trial_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    events = load_jsonl(trial_dir / "transcript.jsonl")
    input_dur = float(meta.get("input_duration_sec") or 0.0)
    result = classify_events(events, input_dur)
    result.update(
        {
            "trial": str(trial_dir),
            "prompt_text": meta.get("prompt_text"),
            "input_duration_sec": round(input_dur, 3),
            "seed": meta.get("seed"),
            "first_response_latency_sec": meta.get("first_response_latency_sec"),
            "audible_response_start_sec": meta.get("audible_response_start_sec"),
        }
    )
    return result


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    n_trials = len(trials)
    n_with_listening = sum(1 for t in trials if t["n_listening"] > 0)
    n_with_any = sum(1 for t in trials if t["n_listening"] + t["n_response"] > 0)
    kind_counts: dict[str, int] = {}
    for t in trials:
        for hit in t["listening"] + t["response"]:
            kind_counts[hit["kind"]] = kind_counts.get(hit["kind"], 0) + 1
    return {
        "n_trials": n_trials,
        "n_trials_with_listening_backchannel": n_with_listening,
        "n_trials_with_any_backchannel": n_with_any,
        "listening_coverage": round(n_with_listening / n_trials, 3) if n_trials else 0.0,
        "total_listening_hits": sum(t["n_listening"] for t in trials),
        "total_response_hits": sum(t["n_response"] for t in trials),
        "backchannel_kind_counts": dict(sorted(kind_counts.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="response_recorder.py の出力ルート (run_metadata.json がある所)。",
    )
    parser.add_argument(
        "--report-name",
        default="backchannel_report.json",
        help="サマリの保存ファイル名。",
    )
    args = parser.parse_args()

    trial_dirs = find_trial_dirs(args.out_dir)
    if not trial_dirs:
        raise SystemExit(f"No trials (meta.json) found under {args.out_dir}")

    trials: list[dict[str, Any]] = []
    for d in trial_dirs:
        res = analyze_trial(d)
        if res is not None:
            trials.append(res)

    print(f"=== backchannel analysis: {args.out_dir} ===")
    print(f"{'trial':<52} {'in_sec':>7} {'listen':>7} {'resp':>5}  prompt")
    for t in trials:
        rel = t["trial"]
        if len(rel) > 50:
            rel = "..." + rel[-47:]
        prompt = (t.get("prompt_text") or "")[:30]
        print(
            f"{rel:<52} {t['input_duration_sec']:>7.2f} "
            f"{t['n_listening']:>7} {t['n_response']:>5}  {prompt}"
        )

    summary = summarize(trials)
    print("\n--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # listening 相槌の実例を少し見せる（質の目視確認用）
    print("\n--- listening backchannel examples (max 15) ---")
    shown = 0
    for t in trials:
        for hit in t["listening"]:
            print(
                f"[{hit['time_sec']:>6.2f}s] {hit['kind']:<9} "
                f"surface={hit['surface']!r}  ctx={hit['window_text'][:40]!r}"
            )
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break
    if shown == 0:
        print("(listening 区間の相槌は検出されず。temp/プロンプト長/silence を調整して再検証)")

    report = {"summary": summary, "trials": trials}
    report_path = args.out_dir / args.report_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
