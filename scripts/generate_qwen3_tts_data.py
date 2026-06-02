#!/usr/bin/env python3
from __future__ import annotations
"""
Qwen3-TTS を使ってシンプルな対話音声データを生成し、
Moshi fine-tune フォーマット（ステレオWAV + JSONL manifest）で書き出す。

出力ディレクトリ構造:
  out_dir/
    synthetic_moshi_train.jsonl   ← Moshi fine-tune manifest
    dialogues.jsonl               ← 対話スクリプト
    data_stereo/
      sample_001_<id>.wav         ← ステレオWAV (左=moshi/相談員, 右=user/相談者)
      sample_001_<id>.json        ← アライメント / メタデータ

使い方:
  uv run python scripts/generate_qwen3_tts_data.py --out-dir ./output_qwen3
  uv run python scripts/generate_qwen3_tts_data.py --out-dir ./output_qwen3 \
      --model Qwen/Qwen3-TTS --device cuda --num-dialogues 5

Qwen3-TTS は trust_remote_code=True で AutoModel として呼び出します。
モデルが .inference(text, speaker) を持つ想定です（CosyVoice2 ベース）。
API が異なる場合は Qwen3TTS.synthesize() を修正してください。
"""

import argparse
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000  # Qwen3-TTS / Moshi どちらも 24 kHz


# ---------------------------------------------------------------------------
# 対話テンプレート（孤独・孤立相談窓口想定の短い日本語対話）
# ---------------------------------------------------------------------------

TEMPLATE_DIALOGUES: list[dict[str, Any]] = [
    {
        "id": "smalltalk_evening_001",
        "category": "smalltalk",
        "risk_level": "low",
        "title": "夜の雑談",
        "turns": [
            {"speaker": "user",  "text": "こんばんは。相談というほどでもないんですが、少し話してもいいですか。"},
            {"speaker": "moshi", "text": "もちろんです。来てくれてありがとうございます。どうぞゆっくり話してください。"},
            {"speaker": "user",  "text": "最近、夜になると少し寂しくなるんですよね。"},
            {"speaker": "moshi", "text": "そうですか。夜は特に静かになって、気持ちが大きくなることがありますよね。"},
            {"speaker": "user",  "text": "そうなんです。誰かと話すとちょっと楽になります。"},
            {"speaker": "moshi", "text": "ここで話してくれてよかったです。急がなくて大丈夫ですよ。"},
        ],
    },
    {
        "id": "holiday_loneliness_001",
        "category": "loneliness_light",
        "risk_level": "low",
        "title": "休日の孤独感",
        "turns": [
            {"speaker": "user",  "text": "休日に予定がないと、自分だけ誰にも呼ばれてない気がします。"},
            {"speaker": "moshi", "text": "その気持ち、わかります。休みの日って比べてしまうことがありますよね。"},
            {"speaker": "user",  "text": "SNSを見ると余計に落ち込んでしまいます。"},
            {"speaker": "moshi", "text": "見なくてもいいんですよ。今日はここでゆっくり話しましょう。"},
            {"speaker": "user",  "text": "そうですね。少し気持ちが楽になりました。"},
            {"speaker": "moshi", "text": "それはよかったです。いつでも話しに来てください。"},
        ],
    },
    {
        "id": "help_hesitation_001",
        "category": "loneliness_deep",
        "risk_level": "medium",
        "title": "助けを求めることへの躊躇",
        "turns": [
            {"speaker": "user",  "text": "助けてって言いたいんですけど、迷惑だと思われそうで言えません。"},
            {"speaker": "moshi", "text": "声を出すだけでも、とても勇気がいりますよね。"},
            {"speaker": "user",  "text": "はい。自分で解決しないといけないと思って。"},
            {"speaker": "moshi", "text": "ひとりで抱えてきたんですね。今ここで話してくれて、よかったです。"},
            {"speaker": "user",  "text": "少し楽になった気がします。"},
            {"speaker": "moshi", "text": "今夜、安全に過ごせそうですか？ひとこと聞かせてもらえますか。"},
            {"speaker": "user",  "text": "はい、大丈夫です。"},
            {"speaker": "moshi", "text": "よかった。また話しに来てくださいね。"},
        ],
    },
]


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    speaker: str
    text: str


@dataclass
class Dialogue:
    id: str
    category: str
    risk_level: str
    title: str
    turns: list[DialogueTurn]


@dataclass
class AudioSegment:
    speaker: str
    label: str
    text: str
    start_sec: float
    end_sec: float
    pcm: np.ndarray


# ---------------------------------------------------------------------------
# Qwen3-TTS ラッパー
# ---------------------------------------------------------------------------

class Qwen3TTS:
    """
    Qwen3-TTS (CosyVoice2 ベース) の薄いラッパー。

    Qwen3-TTS の推論 API は trust_remote_code=True で読み込んだ AutoModel を通じて
    model.inference(text, speaker) を呼ぶ形式を想定しています。
    実際の API が異なる場合はこのクラスの synthesize() を修正してください。
    """

    # Qwen3-TTS が提供するデフォルト話者（モデルカード記載のものを使用）
    SPEAKER_USER  = "Chelsie"   # 相談者側に使う声
    SPEAKER_MOSHI = "Cherry"    # 相談員側に使う声

    def __init__(self, model_id: str, device: str, dtype_str: str):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype_str
        self.model = None
        self.sample_rate = SAMPLE_RATE

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
            "auto":     "auto",
        }
        torch_dtype = dtype_map.get(self.dtype_str, "auto")

        logger.info("Qwen3-TTS を読み込み中: %s (device=%s, dtype=%s)", self.model_id, self.device, self.dtype_str)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Qwen3-TTS 読み込み完了")

    def synthesize(self, text: str, speaker_role: str) -> np.ndarray:
        """
        speaker_role: "user" | "moshi"
        返り値: float32 モノラル PCM (self.sample_rate Hz)
        """
        self.load()
        assert self.model is not None

        voice = self.SPEAKER_USER if speaker_role == "user" else self.SPEAKER_MOSHI

        import torch
        with torch.no_grad():
            # Qwen3-TTS (CosyVoice2) の標準 API
            # モデルカードの例: audio, sr = model.inference(text="...", speaker="...")
            try:
                result = self.model.inference(text=text, speaker=voice)
            except TypeError:
                # キーワード引数の形式が異なる場合の fallback
                result = self.model.inference(text, voice)

        # 結果が (audio_np, sample_rate) のタプルか、音声 Tensor かを正規化
        if isinstance(result, (tuple, list)):
            audio, sr = result[0], result[1]
        else:
            audio = result
            sr = self.sample_rate

        # Tensor → numpy
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()

        # リサンプリングが必要な場合
        sr = int(sr)
        if sr != self.sample_rate:
            audio = _resample(audio, sr, self.sample_rate)

        logger.info(
            "Qwen3-TTS 合成完了: speaker=%s voice=%s dur=%.2fs text=%r",
            speaker_role, voice, audio.size / self.sample_rate, text[:30],
        )
        return audio


def _resample(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import torchaudio
        import torch
        t = torch.from_numpy(pcm).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_sr, target_sr)
        return t.squeeze(0).numpy()
    except Exception:
        # 粗いリサンプル（精度より動作優先）
        ratio = target_sr / orig_sr
        n_out = int(len(pcm) * ratio)
        indices = (np.arange(n_out) / ratio).astype(np.int32)
        indices = np.clip(indices, 0, len(pcm) - 1)
        return pcm[indices]


# ---------------------------------------------------------------------------
# ステレオ合成ユーティリティ
# ---------------------------------------------------------------------------

def build_segments(dialogue: Dialogue, tts: Qwen3TTS, lead_in_sec: float, gap_sec: float) -> list[AudioSegment]:
    cursor = lead_in_sec
    segments: list[AudioSegment] = []
    for turn in dialogue.turns:
        pcm = tts.synthesize(turn.text, turn.speaker)
        start = cursor
        end = start + pcm.size / tts.sample_rate
        segments.append(AudioSegment(
            speaker=turn.speaker,
            label="SPEAKER_MAIN" if turn.speaker == "moshi" else "SPEAKER_USER",
            text=turn.text,
            start_sec=start,
            end_sec=end,
            pcm=pcm,
        ))
        cursor = end + gap_sec
    return segments


def render_stereo(segments: list[AudioSegment], sample_rate: int, tail_sec: float = 0.5) -> np.ndarray:
    duration = max((s.end_sec for s in segments), default=0.0) + tail_sec
    n = max(1, int(math.ceil(duration * sample_rate)))
    stereo = np.zeros((2, n), dtype=np.float32)
    for seg in segments:
        ch = 0 if seg.speaker == "moshi" else 1
        st = int(round(seg.start_sec * sample_rate))
        en = min(n, st + seg.pcm.size)
        if en > st:
            stereo[ch, st:en] += seg.pcm[: en - st]
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.99:
        stereo = stereo / peak * 0.99
    return stereo.astype(np.float32)


# ---------------------------------------------------------------------------
# I/O ユーティリティ
# ---------------------------------------------------------------------------

def write_wav(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    import sphn  # type: ignore[import]
    path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(path), stereo.astype(np.float32), sample_rate)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def safe_stem(text: str, fallback: str) -> str:
    stem = re.sub(r"\s+", "_", text.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)
    stem = re.sub(r"[^0-9A-Za-z_.\-぀-ヿ㐀-鿿]+", "_", stem)
    return (stem[:60].strip("._-") or fallback)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS で日本語対話音声データを生成し Moshi fine-tune フォーマットで保存する"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS", help="Qwen3-TTS モデル ID")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--num-dialogues", type=int, default=3, help="生成する対話数（最大 %d）" % len(TEMPLATE_DIALOGUES))
    parser.add_argument("--lead-in-sec", type=float, default=0.3)
    parser.add_argument("--gap-sec", type=float, default=0.4, help="ターン間の無音（秒）")
    parser.add_argument("--manifest-name", default="synthetic_moshi_train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.out_dir / "data_stereo"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / args.manifest_name
    dialogues_path = args.out_dir / "dialogues.jsonl"
    for p in (manifest_path, dialogues_path):
        if p.exists():
            p.unlink()

    tts = Qwen3TTS(model_id=args.model, device=args.device, dtype_str=args.dtype)
    tts.load()

    templates = TEMPLATE_DIALOGUES[: args.num_dialogues]
    for idx, tmpl in enumerate(templates, start=1):
        dialogue = Dialogue(
            id=safe_stem(tmpl["id"], f"dialogue_{idx:03d}"),
            category=tmpl["category"],
            risk_level=tmpl["risk_level"],
            title=tmpl["title"],
            turns=[DialogueTurn(t["speaker"], t["text"]) for t in tmpl["turns"]],
        )

        logger.info("[%d/%d] 対話 %s を合成中 ...", idx, len(templates), dialogue.id)
        t0 = time.time()
        segments = build_segments(dialogue, tts, args.lead_in_sec, args.gap_sec)
        stereo = render_stereo(segments, tts.sample_rate)
        elapsed = time.time() - t0

        stem = f"sample_{idx:03d}_{dialogue.id}"
        wav_path = data_dir / f"{stem}.wav"
        json_path = wav_path.with_suffix(".json")
        duration = stereo.shape[-1] / tts.sample_rate

        write_wav(wav_path, stereo, tts.sample_rate)

        alignments = [
            [seg.text, [round(seg.start_sec, 4), round(seg.end_sec, 4)], seg.label]
            for seg in segments
        ]
        write_json(json_path, {
            "alignments": alignments,
            "metadata": {
                "mode": "qwen3-tts-scripted",
                "sample_rate": tts.sample_rate,
                "duration_sec": round(duration, 4),
                "tts_model": args.model,
                "left_channel": "moshi",
                "right_channel": "user",
                "wall_time_sec": round(elapsed, 3),
                "dialogue": {
                    "id": dialogue.id,
                    "category": dialogue.category,
                    "risk_level": dialogue.risk_level,
                    "title": dialogue.title,
                    "turns": [asdict(t) for t in dialogue.turns],
                },
            },
        })

        append_jsonl(dialogues_path, {
            "id": dialogue.id,
            "category": dialogue.category,
            "risk_level": dialogue.risk_level,
            "title": dialogue.title,
            "turns": [asdict(t) for t in dialogue.turns],
        })
        append_jsonl(manifest_path, {
            "path": str(wav_path.relative_to(args.out_dir)).replace("\\", "/"),
            "duration": duration,
        })

        logger.info("保存完了: %s (%.2f 秒, wall %.1f 秒)", wav_path, duration, elapsed)

    logger.info("完了。manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
