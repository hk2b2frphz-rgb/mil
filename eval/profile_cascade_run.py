#!/usr/bin/env python3
"""
カスケード実行の時間内訳を出す。どこが遅いのかを推測せずに特定するための道具。

eval/run_local_baseline_full_duplex.py は 1 ケースごとに段別の実測値を
output.meta.json へ書いている。それを集めて合計・中央値・1 ケースあたりを出す。

  asr_wall_time_sec         入力の書き起こし
  llm_wall_time_sec         応答文の生成
  tts_wall_time_sec         音声合成
  output_asr_wall_time_sec  出力音声の再書き起こし(アライメント用)

使い方:
    uv run python eval/profile_cascade_run.py \\
        --run-dir eval_runs/real_response/<run>/inference
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf

STAGES = (
    ("asr_wall_time_sec", "ASR(入力)"),
    ("llm_wall_time_sec", "LLM"),
    ("tts_wall_time_sec", "TTS"),
    ("output_asr_wall_time_sec", "ASR(出力アライメント)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top", type=int, default=5,
                        help="遅かったケースを何件まで出すか")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metas = sorted(args.run_dir.glob("*/*/seed_*/output.meta.json"))
    if not metas:
        raise SystemExit(f"{args.run_dir} に output.meta.json がありません。")

    rows = []
    for path in metas:
        row = json.loads(path.read_text(encoding="utf-8"))
        # 応答がどれだけ長いか。TTS が遅いのか、長い文を喋らせているのかを
        # 分けるために要る。
        text_path = path.parent / "output.json"
        if text_path.is_file():
            data = json.loads(text_path.read_text(encoding="utf-8"))
            row["_text"] = data.get("generated_text") or data.get("text") or ""
        else:
            row["_text"] = ""
        wav_path = path.parent / "output.wav"
        row["_response_sec"] = 0.0
        if wav_path.is_file():
            try:
                info = sf.info(str(wav_path))
                row["_response_sec"] = max(
                    0.0, info.duration - float(row.get("input_duration_sec") or 0.0)
                    - float(row.get("wall_time_sec") or 0.0)
                )
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
    stages = {key: [float(r.get(key) or 0.0) for r in rows] for key, _ in STAGES}
    totals = [float(r.get("wall_time_sec") or 0.0) for r in rows]
    inputs = [float(r.get("input_duration_sec") or 0.0) for r in rows]

    # 壁時計は 2 つある。応答の置き位置に使われるのは wall_time_sec で、これは
    # ASR(入力)+LLM+TTS のみ。出力アライメント ASR は応答が鳴った後の処理なので
    # 応答速度には入らないが、ジョブの実時間には効く。
    placement = sum(totals)
    grand = sum(sum(v) for v in stages.values())
    print(f"ケース数:       {len(rows)}")
    print(f"入力音声の合計: {sum(inputs) / 60:.1f} 分")
    print(f"ジョブの処理時間 合計: {grand / 60:.1f} 分 "
          f"(1 ケース平均 {grand / len(rows):.2f} 秒)")
    print(f"うち応答速度に入る分: {placement / 60:.1f} 分 "
          f"(1 ケース平均 {placement / len(rows):.2f} 秒)")
    print()
    print(f"{'段':<24} {'合計(分)':>9} {'1件平均':>9} {'中央値':>9} {'割合':>7}")
    print("-" * 62)
    for key, label in STAGES:
        values = stages[key]
        total = sum(values)
        share = total / grand * 100 if grand else 0.0
        print(f"{label:<24} {total / 60:>9.1f} {total / len(values):>9.2f} "
              f"{statistics.median(values):>9.2f} {share:>6.1f}%")
    print()
    print("ASR(出力アライメント)は応答が鳴った後の処理なので応答速度には入らない。")
    print("時間を削りたいだけなら CASCADE_SKIP_OUTPUT_ALIGNMENT=1 で省ける")
    print("(応答テキストは LLM 側に残る。失うのは時刻つきチャンクだけ)。")
    print()

    chars = [len(r["_text"]) for r in rows]
    resp_secs = [r["_response_sec"] for r in rows]
    print(f"応答テキスト長: 中央値 {statistics.median(chars):.0f} 文字 "
          f"/ 最長 {max(chars)} 文字")
    print(f"応答音声長:     中央値 {statistics.median(resp_secs):.1f} 秒 "
          f"/ 最長 {max(resp_secs):.1f} 秒")
    truncated = [r for r in rows if r.get("response_looks_truncated")]
    if truncated:
        print(f"言い切らずに終わった応答: {len(truncated)}/{len(rows)} 件")
        print("  max_new_tokens で切られた疑い。上限を上げるか、プロンプトで")
        print("  さらに短く指示すること(上限を下げても文の途中で切れるだけ)。")
        for row in truncated[:3]:
            print(f"    {row.get('case_id')}: 「{row['_text'][-40:]}」")
    if statistics.median(chars) > 80:
        print("  応答が長い。--llm-max-new-tokens を下げると TTS ごと短くなる")
        print("  (CASCADE_LLM_MAX_NEW_TOKENS、既定 200)。")
    print()

    ratio = placement / sum(inputs) if sum(inputs) else 0.0
    print(f"実時間比: 入力 1 秒あたり {ratio:.2f} 秒")
    if ratio > 1.0:
        print("  1 を超えている = 実時間より遅い。対話システムとしては成立しない")
        print("  速度だが、評価としては応答速度に正直に反映されるだけで問題ない。")

    print()
    print(f"遅かったケース 上位 {args.top}:")
    for row in sorted(rows, key=lambda r: -(r.get("wall_time_sec") or 0.0))[:args.top]:
        parts = " ".join(
            f"{label}={float(row.get(key) or 0.0):.2f}" for key, label in STAGES
        )
        print(f"  {row.get('case_id')}: 合計 {row.get('wall_time_sec')}s  {parts}")
        print(f"      応答 {len(row['_text'])} 文字 / 音声 {row['_response_sec']:.1f} 秒"
              f"  「{row['_text'][:40]}」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
