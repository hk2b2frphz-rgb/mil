#!/usr/bin/env python3
from __future__ import annotations
"""
Qwen3-TTS を使ってシンプルな日本語対話の音声データを生成し、
Moshi fine-tune フォーマット（ステレオWAV + JSONL manifest）で書き出す。

出力ディレクトリ構造:
  out_dir/
    synthetic_moshi_train.jsonl   ← Moshi fine-tune manifest
    dialogues.jsonl               ← 対話スクリプト
    data_stereo/
      sample_001_<id>.wav         ← ステレオWAV (左=moshi/相談員, 右=user/相談者)
      sample_001_<id>.json        ← アライメント / メタデータ

使い方（GPUサーバー上で）:
  pip install -U qwen-tts        # または uv sync
  uv run python scripts/generate_qwen3_tts_data.py --out-dir ./output_qwen3

  # 話者やモデルを変えたい場合
  uv run python scripts/generate_qwen3_tts_data.py \
      --out-dir ./output_qwen3 \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --speaker-user Ono_Anna \
      --speaker-moshi Serena \
      --num-dialogues 3

Qwen3-TTS のプリセット話者 (CustomVoice モデル):
  Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
"""

import argparse
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 対話テンプレート（孤独・孤立相談窓口想定の短い日本語対話）
# ---------------------------------------------------------------------------

# 感情ラベル → Qwen3-TTS の instruct 文字列。
# 試験的なマッピング。最終的には fine-tune 結果を見て調整する。
EMOTION_PRESETS: dict[str, str] = {
    # 中立
    "neutral":      "自然で落ち着いたトーンで話して",
    # 相談者 (user) 側でよく使う感情
    "hesitant":     "少し言い淀みながら、ためらうように話して",
    "sad":          "声を少し落として、沈んだトーンで静かに話して",
    "lonely":       "寂しさが滲むような、低めの静かなトーンで話して",
    "anxious":      "不安が伝わるように、声を少し震わせ気味に話して",
    "relieved":     "少しほっとした、肩の力が抜けたような穏やかなトーンで話して",
    "grateful":     "感謝の気持ちを込めて、柔らかいトーンで話して",
    # 相談員 (moshi) 側でよく使う感情
    "warm":         "温かく、相手に寄り添うような穏やかなトーンで話して",
    "gentle":       "やさしく語りかけるように、ゆっくり丁寧に話して",
    "empathetic":   "共感を込めて、相手の気持ちを受け止めるように話して",
    "encouraging": "やさしく背中を押すように、明るすぎず前向きなトーンで話して",
    "concerned":    "心配する気持ちを込めて、落ち着いた声で慎重に話して",
    "reassuring":   "安心させるように、ゆっくり穏やかに話して",
}


TEMPLATE_DIALOGUES: list[dict[str, Any]] = [
    {
        "id": "smalltalk_evening_001",
        "category": "smalltalk",
        "risk_level": "low",
        "title": "夜の雑談",
        "turns": [
            {"speaker": "user",  "emotion": "hesitant",
             "text": "こんばんは。相談というほどでもないんですが、少し話してもいいですか。"},
            {"speaker": "moshi", "emotion": "warm",
             "text": "もちろんです。来てくれてありがとうございます。どうぞゆっくり話してください。"},
            {"speaker": "user",  "emotion": "lonely",
             "text": "最近、夜になると少し寂しくなるんですよね。"},
            {"speaker": "moshi", "emotion": "empathetic",
             "text": "そうですか。夜は特に静かになって、気持ちが大きくなることがありますよね。"},
            {"speaker": "user",  "emotion": "relieved",
             "text": "そうなんです。誰かと話すとちょっと楽になります。"},
            {"speaker": "moshi", "emotion": "gentle",
             "text": "ここで話してくれてよかったです。急がなくて大丈夫ですよ。"},
        ],
    },
    {
        "id": "holiday_loneliness_001",
        "category": "loneliness_light",
        "risk_level": "low",
        "title": "休日の孤独感",
        "turns": [
            {"speaker": "user",  "emotion": "sad",
             "text": "休日に予定がないと、自分だけ誰にも呼ばれてない気がします。"},
            {"speaker": "moshi", "emotion": "empathetic",
             "text": "その気持ち、わかります。休みの日って比べてしまうことがありますよね。"},
            {"speaker": "user",  "emotion": "anxious",
             "text": "SNSを見ると余計に落ち込んでしまいます。"},
            {"speaker": "moshi", "emotion": "reassuring",
             "text": "見なくてもいいんですよ。今日はここでゆっくり話しましょう。"},
            {"speaker": "user",  "emotion": "relieved",
             "text": "そうですね。少し気持ちが楽になりました。"},
            {"speaker": "moshi", "emotion": "encouraging",
             "text": "それはよかったです。いつでも話しに来てください。"},
        ],
    },
    {
        "id": "help_hesitation_001",
        "category": "loneliness_deep",
        "risk_level": "medium",
        "title": "助けを求めることへの躊躇",
        "turns": [
            {"speaker": "user",  "emotion": "hesitant",
             "text": "助けてって言いたいんですけど、迷惑だと思われそうで言えません。"},
            {"speaker": "moshi", "emotion": "empathetic",
             "text": "声を出すだけでも、とても勇気がいりますよね。"},
            {"speaker": "user",  "emotion": "sad",
             "text": "はい。自分で解決しないといけないと思って。"},
            {"speaker": "moshi", "emotion": "warm",
             "text": "ひとりで抱えてきたんですね。今ここで話してくれて、よかったです。"},
            {"speaker": "user",  "emotion": "relieved",
             "text": "少し楽になった気がします。"},
            {"speaker": "moshi", "emotion": "concerned",
             "text": "今夜、安全に過ごせそうですか？ひとこと聞かせてもらえますか。"},
            {"speaker": "user",  "emotion": "grateful",
             "text": "はい、大丈夫です。"},
            {"speaker": "moshi", "emotion": "reassuring",
             "text": "よかった。また話しに来てくださいね。"},
        ],
    },
    {
        "id": "user_silence_001",
        "category": "loneliness_light",
        "risk_level": "low",
        "title": "ユーザーが言葉に詰まる",
        "turns": [
            {"speaker": "user",  "emotion": "hesitant",
             "text": "あの…ちょっと聞いてもらえますか。"},
            {"speaker": "moshi", "emotion": "warm",
             "text": "はい、もちろんです。お話ししてくださいね。"},
            {"speaker": "user",  "emotion": "hesitant",
             "text": "うまく言えないんですけど…えっと…"},
            # ここから沈黙: ユーザーが言葉に詰まる
            {"speaker": "silence", "duration_sec": 3.5,
             "note": "ユーザーが言い淀んで黙ってしまう"},
            {"speaker": "moshi", "emotion": "gentle",
             "text": "ゆっくりで大丈夫ですよ。"},
            {"speaker": "silence", "duration_sec": 2.0,
             "note": "ユーザーがまだ考え中"},
            {"speaker": "moshi", "emotion": "reassuring",
             "text": "急がなくて大丈夫です。言葉が出てこない日もありますからね。"},
            {"speaker": "user",  "emotion": "relieved",
             "text": "…ありがとうございます。少し落ち着きました。"},
            {"speaker": "moshi", "emotion": "warm",
             "text": "よかったです。話せそうなところから、ぽつぽつで大丈夫ですよ。"},
        ],
    },
    {
        "id": "user_silence_002",
        "category": "loneliness_deep",
        "risk_level": "medium",
        "title": "ユーザーが長く沈黙する",
        "turns": [
            {"speaker": "user",  "emotion": "sad",
             "text": "もしもし…。"},
            {"speaker": "moshi", "emotion": "warm",
             "text": "もしもし、お電話ありがとうございます。聞いていますよ。"},
            # 長めの沈黙
            {"speaker": "silence", "duration_sec": 5.0,
             "note": "ユーザーが何も言わない"},
            {"speaker": "moshi", "emotion": "gentle",
             "text": "大丈夫ですよ。話さなくても、つながっているだけで大丈夫です。"},
            {"speaker": "silence", "duration_sec": 4.0,
             "note": "まだ沈黙が続く"},
            {"speaker": "moshi", "emotion": "concerned",
             "text": "もしよかったら、今いる場所が安全かだけ、ひとこと教えてもらえますか。"},
            {"speaker": "silence", "duration_sec": 2.5,
             "note": "ためらいの沈黙"},
            {"speaker": "user",  "emotion": "hesitant",
             "text": "…家にいます。大丈夫です。"},
            {"speaker": "moshi", "emotion": "reassuring",
             "text": "ありがとうございます。安心しました。"},
            {"speaker": "moshi", "emotion": "gentle",
             "text": "話したくなったら、いつでも声を出してくださいね。"},
        ],
    },
]


VALID_SPEAKERS = {
    "Vivian", "Serena", "Uncle_Fu", "Dylan",
    "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee",
}


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    speaker: str  # "user" | "moshi" | "silence"
    text: str = ""
    emotion: str | None = None
    instruct: str | None = None      # 解決後の Qwen3-TTS instruct 文字列（参照保存用）
    duration_sec: float | None = None  # speaker=="silence" 用
    note: str | None = None          # 沈黙の状況メモなど（学習対象外）


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
    Qwen3-TTS (CustomVoice) の薄いラッパー。

    実体は qwen-tts パッケージの Qwen3TTSModel:
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(...)
        wavs, sr = model.generate_custom_voice(
            text=..., language="Japanese", speaker="Vivian", instruct=...
        )
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype_str: str,
        attn_impl: str,
        speaker_user: str,
        speaker_moshi: str,
        language: str,
        instruct_user: str | None,
        instruct_moshi: str | None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype_str
        self.attn_impl = attn_impl
        self.speaker_user = speaker_user
        self.speaker_moshi = speaker_moshi
        self.language = language
        self.instruct_user = instruct_user
        self.instruct_moshi = instruct_moshi
        self.model = None
        self.sample_rate: int = 0

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "qwen-tts package is required. Install with `pip install -U qwen-tts` "
                "or add it to pyproject.toml and run `uv sync`."
            ) from exc

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        device_map = self.device
        if device_map == "cuda":
            device_map = "cuda:0"

        logger.info(
            "Qwen3-TTS を読み込み中: %s (device_map=%s, dtype=%s, attn=%s)",
            self.model_id, device_map, self.dtype_str, self.attn_impl,
        )

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": torch_dtype,
        }
        if self.attn_impl and self.attn_impl != "default":
            load_kwargs["attn_implementation"] = self.attn_impl

        try:
            self.model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
        except Exception as exc:
            # flash_attention_2 が未インストールの場合などのフォールバック
            if self.attn_impl == "flash_attention_2":
                logger.warning(
                    "flash_attention_2 の読み込みに失敗。デフォルトの attention で再試行します: %s",
                    exc,
                )
                load_kwargs.pop("attn_implementation", None)
                self.model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
            else:
                raise

        logger.info("Qwen3-TTS 読み込み完了")

    def resolve_instruct(self, speaker_role: str, turn_instruct: str | None) -> str | None:
        """ターン側で明示指定があればそれ、無ければ CLI 既定 (--instruct-*) を使う。"""
        if turn_instruct:
            return turn_instruct
        if speaker_role == "user":
            return self.instruct_user
        return self.instruct_moshi

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        """
        speaker_role: "user" | "moshi"
        instruct: そのターンに使うスタイル指示。None なら既定にフォールバック。
        speaker_override: ロールごとの既定話者の代わりに使う話者名（対話単位で
            user 話者を切り替えるための引数）。
        返り値: float32 モノラル PCM (self.sample_rate Hz)
        """
        self.load()
        assert self.model is not None

        if speaker_override:
            voice = speaker_override
        else:
            voice = self.speaker_user if speaker_role == "user" else self.speaker_moshi
        instruct = self.resolve_instruct(speaker_role, instruct)

        kwargs: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "speaker": voice,
        }
        if instruct:
            kwargs["instruct"] = instruct

        import torch
        with torch.no_grad():
            wavs, sr = self.model.generate_custom_voice(**kwargs)

        # wavs は numpy 配列のリスト
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()

        sr = int(sr)
        if self.sample_rate == 0:
            self.sample_rate = sr
            logger.info("Qwen3-TTS sample rate = %d Hz", sr)
        elif sr != self.sample_rate:
            audio = _resample(audio, sr, self.sample_rate)

        logger.info(
            "Qwen3-TTS 合成完了: role=%s speaker=%s instruct=%r dur=%.2fs text=%r",
            speaker_role, voice, (instruct or "")[:24],
            audio.size / self.sample_rate, text[:30],
        )
        return audio


def _resample(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return pcm
    try:
        import torch
        import torchaudio
        t = torch.from_numpy(pcm).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_sr, target_sr)
        return t.squeeze(0).numpy()
    except Exception:
        # 粗いリサンプル（動作優先）
        ratio = target_sr / orig_sr
        n_out = int(len(pcm) * ratio)
        indices = (np.arange(n_out) / ratio).astype(np.int32)
        indices = np.clip(indices, 0, len(pcm) - 1)
        return pcm[indices]


# ---------------------------------------------------------------------------
# ステレオ合成ユーティリティ
# ---------------------------------------------------------------------------

def build_segments(
    dialogue: Dialogue,
    tts: Qwen3TTS,
    lead_in_sec: float,
    gap_sec: float,
    user_speaker_override: str | None = None,
) -> tuple[list[AudioSegment], list[dict[str, Any]]]:
    """
    返り値: (音声セグメント列, 沈黙区間のメタデータ列)

    speaker=="silence" のターンは音声を生成せず、duration_sec ぶんだけカーソルを
    進めて両チャンネル無音とする。沈黙の前後では追加の gap_sec を入れない。
    """
    cursor = lead_in_sec
    segments: list[AudioSegment] = []
    silences: list[dict[str, Any]] = []
    prev_was_silence = False

    for turn in dialogue.turns:
        if turn.speaker == "silence":
            dur = float(turn.duration_sec or 0.0)
            silences.append({
                "start_sec": round(cursor, 4),
                "end_sec": round(cursor + dur, 4),
                "duration_sec": round(dur, 4),
                "note": turn.note or "",
            })
            cursor += dur
            prev_was_silence = True
            continue

        # 直前が沈黙だった場合は、その沈黙が既にギャップを兼ねているので追加 gap は付けない
        # （cursor は既に沈黙ぶん進んでいる）
        override = user_speaker_override if turn.speaker == "user" else None
        pcm = tts.synthesize(
            turn.text,
            turn.speaker,
            instruct=turn.instruct,
            speaker_override=override,
        )
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
        # 次のターンが沈黙の場合、そちら側で時間が積まれるので、ここでは常に gap_sec を足してよい
        cursor = end + gap_sec
        prev_was_silence = False

    _ = prev_was_silence  # 現状未使用だが将来の分岐用に保持
    return segments, silences


def resolve_emotion(emotion: str | None, emotion_map: dict[str, str]) -> str | None:
    if not emotion:
        return None
    instruct = emotion_map.get(emotion)
    if instruct is None:
        logger.warning(
            "未知の emotion=%r。EMOTION_PRESETS のキーを使うか --emotion-map-file を指定してください。",
            emotion,
        )
    return instruct


def load_emotion_map(path: Path | None) -> dict[str, str]:
    base = dict(EMOTION_PRESETS)
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError("--emotion-map-file は { 'label': 'instruct文字列' } の JSON を期待します。")
    base.update({str(k): str(v) for k, v in overrides.items()})
    return base


def load_dialogues_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """外部 (Gemma) の dialogues.jsonl を読む。TEMPLATE_DIALOGUES と同じ形に整える。"""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_turns = row.get("turns") or []
            turns: list[dict[str, Any]] = []
            for t in raw_turns:
                speaker = str(t.get("speaker", "")).strip().lower()
                if speaker == "silence":
                    try:
                        dur = float(t.get("duration_sec", 2.0))
                    except (TypeError, ValueError):
                        dur = 2.0
                    turns.append({
                        "speaker": "silence",
                        "duration_sec": dur,
                        "note": t.get("note", ""),
                    })
                    continue
                if speaker not in {"user", "moshi"}:
                    continue
                text = str(t.get("text", "")).strip()
                if not text:
                    continue
                emotion = t.get("emotion")
                turn = {"speaker": speaker, "text": text}
                if emotion:
                    turn["emotion"] = str(emotion)
                turns.append(turn)
            if not turns:
                logger.warning("対話 %d 行目: turns 抽出失敗、スキップ", i)
                continue
            out.append({
                "id": str(row.get("id") or row.get("source_use_case") or f"dialogue_{i:03d}"),
                "category": str(row.get("category") or "unknown"),
                "risk_level": str(row.get("risk_level") or "low"),
                "title": str(row.get("title") or row.get("id") or f"dialogue {i}"),
                "turns": turns,
            })
    return out


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
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS の HuggingFace モデル ID（CustomVoice 系を推奨）",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-impl",
        default="default",
        choices=["default", "flash_attention_2", "sdpa", "eager"],
        help="attn_implementation。flash_attention_2 は flash-attn が必要。",
    )
    parser.add_argument("--language", default="Japanese",
                        help="generate_custom_voice に渡す language 文字列")
    parser.add_argument("--speaker-user", default="Ono_Anna",
                        help=f"user 側プリセット話者（pool 未指定時の既定）。候補: {sorted(VALID_SPEAKERS)}")
    parser.add_argument(
        "--user-speaker-pool",
        default="Ono_Anna,Sohee,Vivian,Dylan,Eric,Aiden",
        help=(
            "user 側で使う話者のプール（カンマ区切り）。対話ごとにこの順で1人ずつ割り当てる。"
            "moshi 側は --speaker-moshi で固定のまま。空文字列 '' を渡すと --speaker-user で固定。"
        ),
    )
    parser.add_argument("--speaker-moshi", default="Serena",
                        help=f"moshi 側プリセット話者（固定）。候補: {sorted(VALID_SPEAKERS)}")
    parser.add_argument("--instruct-user", default=None,
                        help="user 発話の既定スタイル指示（ターン側 emotion が無い場合に使う）")
    parser.add_argument("--instruct-moshi", default=None,
                        help="moshi 発話の既定スタイル指示（ターン側 emotion が無い場合に使う）")
    parser.add_argument("--no-emotion", action="store_true",
                        help="ターンの emotion ラベルを無視してプレーンに合成する（A/B比較用）")
    parser.add_argument("--emotion-map-file", type=Path, default=None,
                        help="感情ラベル→instruct のオーバーライド JSON 辞書ファイル")
    parser.add_argument(
        "--dialogues-jsonl",
        type=Path,
        default=None,
        help=(
            "外部対話 JSONL（Gemma が出した dialogues.jsonl など）を読む。"
            "未指定なら内蔵 TEMPLATE_DIALOGUES を使う。"
        ),
    )
    parser.add_argument("--num-dialogues", type=int, default=None,
                        help="生成する対話数。未指定なら全件。")
    parser.add_argument("--lead-in-sec", type=float, default=0.3)
    parser.add_argument("--gap-sec", type=float, default=0.4,
                        help="ターン間の無音（秒）")
    parser.add_argument("--manifest-name", default="synthetic_moshi_train.jsonl")
    args = parser.parse_args()

    for role, name in [("--speaker-user", args.speaker_user),
                       ("--speaker-moshi", args.speaker_moshi)]:
        if name not in VALID_SPEAKERS:
            parser.error(
                f"{role}={name!r} は無効です。候補: {sorted(VALID_SPEAKERS)}"
            )

    pool_raw = args.user_speaker_pool.strip()
    if pool_raw:
        pool = [s.strip() for s in pool_raw.split(",") if s.strip()]
        invalid = [s for s in pool if s not in VALID_SPEAKERS]
        if invalid:
            parser.error(
                f"--user-speaker-pool に無効な話者: {invalid}。候補: {sorted(VALID_SPEAKERS)}"
            )
        args.user_speaker_pool_list = pool
    else:
        args.user_speaker_pool_list = [args.speaker_user]
    return args


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

    tts = Qwen3TTS(
        model_id=args.model,
        device=args.device,
        dtype_str=args.dtype,
        attn_impl=args.attn_impl,
        speaker_user=args.speaker_user,
        speaker_moshi=args.speaker_moshi,
        language=args.language,
        instruct_user=args.instruct_user,
        instruct_moshi=args.instruct_moshi,
    )
    tts.load()

    emotion_map = load_emotion_map(args.emotion_map_file)
    if args.no_emotion:
        logger.info("--no-emotion: ターン側 emotion ラベルを無視します")

    if args.dialogues_jsonl is not None:
        all_templates = load_dialogues_from_jsonl(args.dialogues_jsonl)
        logger.info("対話 %d 件を %s から読み込みました", len(all_templates), args.dialogues_jsonl)
    else:
        all_templates = TEMPLATE_DIALOGUES
    if args.num_dialogues is not None:
        templates = all_templates[: args.num_dialogues]
    else:
        templates = all_templates
    for idx, tmpl in enumerate(templates, start=1):
        turns: list[DialogueTurn] = []
        for t in tmpl["turns"]:
            speaker = t["speaker"]
            if speaker == "silence":
                turns.append(DialogueTurn(
                    speaker="silence",
                    text="",
                    duration_sec=float(t.get("duration_sec", 2.0)),
                    note=t.get("note"),
                ))
                continue
            emotion = None if args.no_emotion else t.get("emotion")
            instruct = resolve_emotion(emotion, emotion_map)
            turns.append(DialogueTurn(
                speaker=speaker,
                text=t["text"],
                emotion=emotion,
                instruct=instruct,
            ))
        dialogue = Dialogue(
            id=safe_stem(tmpl["id"], f"dialogue_{idx:03d}"),
            category=tmpl["category"],
            risk_level=tmpl["risk_level"],
            title=tmpl["title"],
            turns=turns,
        )

        user_voice = args.user_speaker_pool_list[(idx - 1) % len(args.user_speaker_pool_list)]
        logger.info(
            "[%d/%d] 対話 %s を合成中 (moshi=%s, user=%s) ...",
            idx, len(templates), dialogue.id, args.speaker_moshi, user_voice,
        )
        t0 = time.time()
        segments, silences = build_segments(
            dialogue, tts, args.lead_in_sec, args.gap_sec,
            user_speaker_override=user_voice,
        )
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
                "language": args.language,
                "speaker_user": user_voice,
                "speaker_user_pool": args.user_speaker_pool_list,
                "speaker_moshi": args.speaker_moshi,
                "left_channel": "moshi",
                "right_channel": "user",
                "wall_time_sec": round(elapsed, 3),
                "emotion_control": "off" if args.no_emotion else "on",
                "emotion_map_used": {
                    e: emotion_map[e]
                    for e in {t.emotion for t in dialogue.turns if t.emotion}
                },
                "silences": silences,
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
