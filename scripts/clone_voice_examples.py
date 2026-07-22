#!/usr/bin/env python3
"""
実対話の切り出しセグメントを参照に、Qwen3-TTS のゼロショット・ボイスクローンで
例文をいくつか合成する(聴き比べ用)。

狙い: Qwen3-TTS CustomVoice のプリセット話者は感情/テンションが固定で不自然な
ことがあるため、実相談員/相談者の声を「参照音声 + その書き起こし」で in-context
クローンし、本人の声・韻律・落ち着いたテンションのまま例文を喋らせる。

入力は analyze_real_dialogue.py の出力ディレクトリ(<wav_stem>/ 配下):
  - timeline.jsonl        {speaker, start, end, text, overlap_sec, is_aizuchi}
  - segments/A/, segments/B/   発話ごとの切り出し WAV(ファイル名先頭が通し番号)
話者(A/B)はユーザーが --speaker で指定する。参照はその話者の
「重畳なし・相槌でない・十分な長さでテキストのある」区間から自動選別する
(--ref-wav / --ref-text で明示指定も可)。

出力 <out_dir>/ 配下:
  - example_00.wav ...    クローンで合成した例文
  - reference.wav         使用した参照音声(コピー)
  - manifest.json         参照・例文・設定の記録

クローン API(qwen-tts パッケージ):
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", ...)
    prompt = model.create_voice_clone_prompt(ref_audio=..., ref_text=..., x_vector_only_mode=False)
    wavs, sr = model.generate_voice_clone(text=[...], language=[...], voice_clone_prompt=prompt)

使い方:
    uv run python scripts/clone_voice_examples.py \
        --analysis-dir data/real_dialogue/<jobid>/<wav_stem> \
        --speaker A \
        --out-dir data/clone_examples/<name>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Heartbeat:
    """内部進捗を出さない長い処理(モデルDL/ロード・合成)の生存確認用。

    with ブロックの間、interval 秒ごとに経過秒をログへ出す。数値が増え続けて
    いれば実行中、増えなければハングと判別できる。daemon スレッド。
    """

    def __init__(self, label: str, interval: float = 15.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            logger.info("%s ... 実行中(%.0f 秒経過)", self.label, time.monotonic() - self._t0)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        logger.info("%s 完了(%.0f 秒)", self.label, time.monotonic() - self._t0)

CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# 参照音声の推奨長。短すぎると声質が不足、長すぎると in-context が重い。
DEFAULT_MIN_REF_SEC = 3.0
DEFAULT_MAX_REF_SEC = 12.0

# 既定の例文(孤独孤立相談窓口ドメイン)。長い共感応答から短い相槌まで混ぜ、
# クローンが本人のテンションを保てているか聴き分けやすくする。相槌は
# analyze_backchannels.AIZUCHI_PATTERNS のカテゴリ(はい/うん/ええ/そう系/
# なるほど/へえ/ふーん/ああ/わかる/たしかに)を、繰り返し・伸ばし・
# 組み合わせの自然な変異まで含めて広くカバーする。
DEFAULT_EXAMPLES = [
    # --- 長め: 挨拶・共感応答(対比用) ---
    "もしもし、こちら孤独孤立相談窓口になります。",
    "今日はどうされましたか。ゆっくりで大丈夫ですよ。",
    "そうだったんですね。それは、お辛かったですね。",
    "一人で抱え込まなくて大丈夫です。少しずつ、お話ししましょう。",
    "なるほど、そういうことがあったんですね。よく話してくださいました。",
    # --- 短い相槌: はい/うん系 ---
    "はい。",
    "はいはい。",
    "はい、聞いていますよ。",
    "うん。",
    "うんうん。",
    "うんうん、そうですよね。",
    # --- ええ/そう系 ---
    "ええ。",
    "ええ、ええ。",
    "そうですね。",
    "そうなんですね。",
    "そうそう。",
    "そっか。",
    # --- なるほど/わかる/たしかに ---
    "なるほど。",
    "なるほどですね。",
    "うんうん、なるほど。",
    "わかります。",
    "たしかに。",
    # --- 感嘆系: へえ/ふーん/ああ ---
    "へえ。",
    "ふーん。",
    "ああ、そうなんですね。",
]


def load_timeline(analysis_dir: Path) -> list[dict[str, Any]]:
    path = analysis_dir / "timeline.jsonl"
    if not path.is_file():
        raise SystemExit(f"timeline.jsonl が見つかりません: {path}")
    segs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        # 書き出しは 1 行 1 セグメントで空行なし。行番号 = segments/ の通し番号。
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["_index"] = i
            segs.append(d)
    return segs


def find_segment_wav(analysis_dir: Path, speaker: str, index: int) -> Path | None:
    seg_dir = analysis_dir / "segments" / speaker
    matches = sorted(seg_dir.glob(f"{index:04d}_*.wav"))
    return matches[0] if matches else None


def _dur(seg: dict[str, Any]) -> float:
    return float(seg["end"]) - float(seg["start"])


def select_reference(
    segs: list[dict[str, Any]], speaker: str, min_sec: float, max_sec: float
) -> dict[str, Any] | None:
    """参照に向く区間を選ぶ: 重畳なし・相槌でない・テキストありの中から、
    推奨長レンジ内で最長のもの(=最も文脈が豊富)。レンジ内が無ければ緩める。"""
    clean = [
        s for s in segs
        if s.get("speaker") == speaker
        and float(s.get("overlap_sec", 0.0)) == 0.0
        and not s.get("is_aizuchi", False)
        and str(s.get("text", "")).strip()
    ]
    in_range = [s for s in clean if min_sec <= _dur(s) <= max_sec]
    pool = in_range or clean
    if not pool:
        return None
    return max(pool, key=_dur)


def load_examples(args: argparse.Namespace) -> list[str]:
    if args.examples_file:
        lines = Path(args.examples_file).read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if ln.strip()]
        if not texts:
            raise SystemExit(f"例文ファイルが空です: {args.examples_file}")
        return texts
    return list(DEFAULT_EXAMPLES)


def resolve_reference(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any] | None]:
    """(ref_wav, ref_text, 選ばれた timeline セグメント or None) を返す。"""
    if args.ref_wav:
        ref_wav = Path(args.ref_wav)
        if not ref_wav.is_file():
            raise SystemExit(f"--ref-wav が存在しません: {ref_wav}")
        if not args.ref_text:
            raise SystemExit("--ref-wav を使うときは --ref-text も必須です")
        return ref_wav, args.ref_text.strip(), None

    if not args.analysis_dir:
        raise SystemExit("--analysis-dir か、--ref-wav/--ref-text のどちらかが必要です")
    analysis_dir = Path(args.analysis_dir)
    segs = load_timeline(analysis_dir)
    chosen = select_reference(segs, args.speaker, args.min_ref_sec, args.max_ref_sec)
    if chosen is None:
        raise SystemExit(
            f"話者 {args.speaker} に参照向きの区間が見つかりません"
            "(重畳なし・相槌でない・テキストありが必要)。--ref-wav/--ref-text で明示指定してください。"
        )
    ref_wav = find_segment_wav(analysis_dir, args.speaker, chosen["_index"])
    if ref_wav is None:
        raise SystemExit(
            f"選ばれた区間の WAV が見つかりません: "
            f"segments/{args.speaker}/{chosen['_index']:04d}_*.wav"
        )
    logger.info(
        "参照に選択: 話者%s idx=%d dur=%.1fs text=%r",
        args.speaker, chosen["_index"], _dur(chosen), chosen["text"],
    )
    return ref_wav, str(chosen["text"]).strip(), chosen


def build_model(args: argparse.Namespace):
    import torch
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(
            "qwen-tts パッケージが必要です(`uv sync` 済みのメイン環境で実行してください)。"
        ) from exc

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(args.dtype, torch.float16)
    device_map = "cuda:0" if args.device == "cuda" else args.device

    load_kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype}
    if args.attn_impl and args.attn_impl != "default":
        load_kwargs["attn_implementation"] = args.attn_impl

    logger.info("Qwen3-TTS(clone)読み込み中: %s (device=%s dtype=%s attn=%s)",
                args.model, device_map, args.dtype, args.attn_impl)
    logger.info("※初回は -Base モデル(数GB)のダウンロードで時間がかかります")
    try:
        with Heartbeat("モデルDL/ロード"):
            return Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)
    except Exception as exc:
        if load_kwargs.pop("attn_implementation", None) is not None:
            logger.warning("attn=%s の読み込みに失敗。既定の attention で再試行: %s",
                           args.attn_impl, exc)
            with Heartbeat("モデルDL/ロード(再試行)"):
                return Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--analysis-dir", type=str, default=None,
                        help="analyze_real_dialogue.py の出力(<wav_stem>/)。参照を自動選別")
    parser.add_argument("--speaker", default="A", help="参照にする話者 A / B(既定 A)")
    parser.add_argument("--ref-wav", default=None, help="参照音声を明示指定(--ref-text 必須)")
    parser.add_argument("--ref-text", default=None, help="--ref-wav の書き起こし")
    parser.add_argument("--out-dir", type=Path, required=True, help="出力先")
    parser.add_argument("--examples-file", default=None,
                        help="例文ファイル(1 行 1 文)。未指定なら既定の相談ドメイン例文")
    parser.add_argument("--language", default="Japanese", help="合成言語(既定 Japanese)")
    parser.add_argument("--model", default=CLONE_MODEL, help=f"クローン用モデル(既定 {CLONE_MODEL})")
    parser.add_argument("--device", default="cuda", help="cuda / cpu(既定 cuda)")
    parser.add_argument("--dtype", default="float16",
                        help="float16 / bfloat16 / float32(V100 は float16)")
    parser.add_argument("--attn-impl", default="default",
                        help="attention 実装(default / sdpa / flash_attention_2)")
    parser.add_argument("--min-ref-sec", type=float, default=DEFAULT_MIN_REF_SEC)
    parser.add_argument("--max-ref-sec", type=float, default=DEFAULT_MAX_REF_SEC)
    parser.add_argument("--x-vector-only", action="store_true",
                        help="話者埋め込みのみで高速クローン(既定は in-context で韻律保持)")
    args = parser.parse_args()

    import soundfile as sf

    ref_wav, ref_text, chosen = resolve_reference(args)
    examples = load_examples(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)

    logger.info("クローンプロンプト作成中(x_vector_only=%s)...", args.x_vector_only)
    with Heartbeat("クローンプロンプト作成"):
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(ref_wav),
            ref_text=ref_text,
            x_vector_only_mode=args.x_vector_only,
        )

    logger.info("例文 %d 文を合成中...", len(examples))
    with Heartbeat("例文合成"):
        wavs, sr = model.generate_voice_clone(
            text=examples,
            language=[args.language] * len(examples),
            voice_clone_prompt=prompt_items,
        )

    out_files: list[str] = []
    for i, (text, wav) in enumerate(zip(examples, wavs)):
        path = args.out_dir / f"example_{i:02d}.wav"
        sf.write(str(path), wav, sr)
        out_files.append(path.name)
        logger.info("  example_%02d.wav <- %r", i, text)

    shutil.copy(ref_wav, args.out_dir / "reference.wav")

    manifest = {
        "model": args.model,
        "language": args.language,
        "x_vector_only_mode": args.x_vector_only,
        "sample_rate": int(sr),
        "speaker": args.speaker,
        "reference": {
            "wav": str(ref_wav),
            "text": ref_text,
            "timeline_index": chosen["_index"] if chosen else None,
            "duration_sec": round(_dur(chosen), 2) if chosen else None,
        },
        "examples": [
            {"file": f, "text": t} for f, t in zip(out_files, examples)
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 耳で確認 ===", file=sys.stderr)
    print(f"{args.out_dir.resolve()}", file=sys.stderr)
    print("  reference.wav   使った参照(本人の声)", file=sys.stderr)
    print("  example_XX.wav  クローン合成した例文", file=sys.stderr)


if __name__ == "__main__":
    main()
