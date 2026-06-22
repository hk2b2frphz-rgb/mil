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
    # 相談者 (user) 側の強い感情・声の状態
    "tearful":      "涙ぐんで、声を震わせながら話して",
    "sobbing":      "声を詰まらせ、すすり泣きながら、途切れ途切れに話して",
    "high_tension": "やや早口で、テンション高く勢いよく話して",
    "agitated":     "感情が高ぶって落ち着かない様子で、少し早口に話して",
    "withdrawn":    "消え入りそうな小さな声で、ぽつぽつと途切れがちに話して",
    "weary":        "疲れて気だるそうに、ゆっくり力なく話して",
    "irritable":    "苛立ちを抑えきれない、とげのあるトーンで話して",
    "laughing":     "軽く笑いながら、明るいトーンで話して",
    # 相談員 (moshi) 側でよく使う感情
    "warm":         "温かく、相手に寄り添うような穏やかなトーンで話して",
    "gentle":       "やさしく語りかけるように、ゆっくり丁寧に話して",
    "empathetic":   "共感を込めて、相手の気持ちを受け止めるように話して",
    "encouraging": "やさしく背中を押すように、明るすぎず前向きなトーンで話して",
    "concerned":    "心配する気持ちを込めて、落ち着いた声で慎重に話して",
    "reassuring":   "安心させるように、ゆっくり穏やかに話して",
    "soothing":     "なだめるように、とてもやわらかくゆっくり話して",
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
    timing: str = "sequential"       # "sequential" | "overlap_previous"
    start_after_previous_start_sec: float | None = None
    truncate_previous_after_sec: float | None = None
    gain: float = 1.0
    voice_role: str | None = None    # "user" | "other" | "background"
    event: str | None = None


@dataclass
class Dialogue:
    id: str
    category: str
    risk_level: str
    title: str
    turns: list[DialogueTurn]
    duplex_task: str | None = None


@dataclass
class AudioSegment:
    speaker: str
    label: str
    text: str
    start_sec: float
    end_sec: float
    pcm: np.ndarray
    event: str | None = None
    voice_role: str | None = None


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


class MossTTSD:
    """MOSS-TTSD per-utterance voice-cloning adapter."""

    ROLES = ("user", "moshi", "other", "background")
    DEFAULT_REF_TEXT = (
        "こんにちは、今日はよろしくお願いします。"
        "ゆっくりお話しできればと思います。"
    )
    DEFAULT_REF_SPEAKERS = {
        "moshi": "Serena",
        "user": "Ono_Anna",
        "other": "Dylan",
        "background": "Ryan",
    }

    def __init__(
        self,
        model_name: str,
        codec_model_name: str,
        ref_audio_paths: dict[str, Path],
        ref_texts: dict[str, str],
        device: str,
        dtype: str,
    ):
        self.model_name = model_name
        self.codec_model_name = codec_model_name
        self.ref_audio_paths = ref_audio_paths
        self.ref_texts = ref_texts
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.sample_rate: int = 0
        self._reference_codes: dict[str, Any] = {}
        self._prompt_audio_codes: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None:
            return

        import torch
        import torchaudio
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-TTSD requires transformers and its audio dependencies. "
                "Run `uv sync` and ensure ffmpeg is available on the cluster."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        device = "cuda:0" if self.device == "cuda" else self.device

        if device.startswith("cuda"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        logger.info(
            "Loading MOSS-TTSD: %s (codec=%s, device=%s, dtype=%s)",
            self.model_name, self.codec_model_name, device, self.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer = AutoModel.from_pretrained(
            self.codec_model_name,
            trust_remote_code=True,
        ).to(device)
        self.processor.audio_tokenizer.eval()
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()

        target_sr = int(self.processor.model_config.sampling_rate)
        for role in self.ROLES:
            wav, sr = torchaudio.load(str(self.ref_audio_paths[role]))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if int(sr) != target_sr:
                wav = torchaudio.functional.resample(wav, int(sr), target_sr)
            audio_codes = self.processor.encode_audios_from_wav(
                [wav], sampling_rate=target_sr,
            )
            self._reference_codes[role] = audio_codes
            self._prompt_audio_codes[role] = audio_codes[0]
        logger.info("MOSS-TTSD loaded; model sample rate = %d Hz", target_sr)

    def _resolve_role(
        self,
        speaker_role: str,
        speaker_override: str | None,
    ) -> str:
        if speaker_override in self.ROLES:
            return str(speaker_override)
        if speaker_role in self.ROLES:
            return speaker_role
        logger.warning("Unknown MOSS-TTSD role=%r; using user reference", speaker_role)
        return "user"

    @staticmethod
    def _tag_s1(text: str) -> str:
        text = re.sub(r"^\s*\[S[1-5]\]\s*", "", text).strip()
        return f"[S1] {text}"

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        import torch

        role = self._resolve_role(speaker_role, speaker_override)
        if instruct:
            logger.debug(
                "MOSS-TTSD ignores instruct=%r for role=%s",
                instruct[:48], role,
            )

        conversations = [[
            self.processor.build_user_message(
                text=(
                    f"{self._tag_s1(self.ref_texts[role])} "
                    f"{self._tag_s1(text)}"
                ),
                reference=self._reference_codes[role],
            ),
            self.processor.build_assistant_message(
                audio_codes_list=[self._prompt_audio_codes[role]],
            ),
        ]]
        batch = self.processor(conversations, mode="continuation")
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2000,
            )
        messages = self.processor.decode(outputs)
        if not messages or not messages[0].audio_codes_list:
            raise RuntimeError("MOSS-TTSD returned no decoded audio")
        audio = messages[0].audio_codes_list[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        output_sr = int(self.processor.model_config.sampling_rate)
        if self.sample_rate == 0:
            self.sample_rate = output_sr
            logger.info("MOSS-TTSD sample rate = %d Hz", output_sr)
        elif output_sr != self.sample_rate:
            audio = _resample(audio, output_sr, self.sample_rate)

        logger.info(
            "MOSS-TTSD synthesis complete role=%s dur=%.2fs text=%r",
            role, audio.size / self.sample_rate, text[:30],
        )
        return audio.astype(np.float32, copy=False)


def load_moss_references(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Load MOSS references from refs.json or explicit CLI arguments."""
    ref_audio_paths: dict[str, Path] = {}
    ref_texts: dict[str, str] = {}

    if args.moss_refs_json is not None:
        with args.moss_refs_json.open(encoding="utf-8") as f:
            refs_data = json.load(f)
        refs = refs_data.get("roles", refs_data)
        for role in MossTTSD.ROLES:
            entry = refs.get(role)
            if not isinstance(entry, dict):
                raise ValueError(f"Missing role {role!r} in {args.moss_refs_json}")
            path = Path(str(entry.get("path", "")))
            if not path.is_absolute():
                path = args.moss_refs_json.parent / path
            text = str(entry.get("transcript", "")).strip()
            if not path.is_file():
                raise FileNotFoundError(
                    f"MOSS reference WAV for {role!r} does not exist: {path}"
                )
            if not text:
                raise ValueError(
                    f"MOSS reference transcript for {role!r} is empty in "
                    f"{args.moss_refs_json}"
                )
            ref_audio_paths[role] = path
            ref_texts[role] = text
        return ref_audio_paths, ref_texts

    for role in MossTTSD.ROLES:
        path = getattr(args, f"moss_ref_{role}")
        text = getattr(args, f"moss_ref_text_{role}")
        if path is None or not text:
            raise ValueError(
                "MOSS-TTSD requires --moss-refs-json or a WAV and transcript "
                f"for every role; missing {role!r}"
            )
        ref_audio_paths[role] = path
        ref_texts[role] = text.strip()

    return ref_audio_paths, ref_texts


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

def truncate_alignment_text(text: str, kept_samples: int, total_samples: int) -> str:
    if not text or total_samples <= 0 or kept_samples >= total_samples:
        return text
    ratio = max(0.0, min(1.0, kept_samples / total_samples))
    keep_chars = max(1, min(len(text), int(math.ceil(len(text) * ratio))))
    shortened = text[:keep_chars].rstrip("、。！？,.!? ")
    return (shortened or text[:1]) + "…"


def build_segments(
    dialogue: Dialogue,
    tts: Qwen3TTS | MossTTSD,
    lead_in_sec: float,
    gap_sec: float,
    user_speaker_override: str | None = None,
    other_speaker_override: str | None = None,
    background_speaker_override: str | None = None,
) -> tuple[list[AudioSegment], list[dict[str, Any]]]:
    """
    返り値: (音声セグメント列, 沈黙区間のメタデータ列)

    speaker=="silence" のターンは音声を生成せず、duration_sec ぶんだけカーソルを
    進めて両チャンネル無音とする。沈黙の前後では追加の gap_sec を入れない。
    """
    cursor = lead_in_sec
    segments: list[AudioSegment] = []
    silences: list[dict[str, Any]] = []
    previous_segment: AudioSegment | None = None

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
            previous_segment = None
            continue

        override = None
        if turn.speaker == "user":
            if turn.voice_role == "background":
                override = background_speaker_override or user_speaker_override
            elif turn.voice_role == "other":
                override = other_speaker_override or user_speaker_override
            else:
                override = user_speaker_override
        pcm = tts.synthesize(
            turn.text,
            turn.speaker,
            instruct=turn.instruct,
            speaker_override=override,
        )
        gain = max(0.0, float(turn.gain))
        if gain != 1.0:
            pcm = np.asarray(pcm, dtype=np.float32) * gain
        if turn.timing == "overlap_previous" and previous_segment is not None:
            offset = max(0.0, float(turn.start_after_previous_start_sec or 0.0))
            latest_overlap_start = max(
                previous_segment.start_sec,
                previous_segment.end_sec - 0.05,
            )
            start = min(previous_segment.start_sec + offset, latest_overlap_start)
            if turn.truncate_previous_after_sec is not None:
                stop = min(
                    previous_segment.end_sec,
                    start + max(0.0, float(turn.truncate_previous_after_sec)),
                )
                original_samples = previous_segment.pcm.size
                keep = max(
                    1,
                    int(round((stop - previous_segment.start_sec) * tts.sample_rate)),
                )
                previous_segment.text = truncate_alignment_text(
                    previous_segment.text,
                    kept_samples=keep,
                    total_samples=original_samples,
                )
                previous_segment.pcm = previous_segment.pcm[:keep]
                previous_segment.end_sec = (
                    previous_segment.start_sec
                    + previous_segment.pcm.size / tts.sample_rate
                )
        else:
            start = cursor
        end = start + pcm.size / tts.sample_rate
        segment = AudioSegment(
            speaker=turn.speaker,
            label="SPEAKER_MAIN" if turn.speaker == "moshi" else "SPEAKER_USER",
            text=turn.text,
            start_sec=start,
            end_sec=end,
            pcm=pcm,
            event=turn.event,
            voice_role=turn.voice_role,
        )
        segments.append(segment)
        cursor = max(item.end_sec for item in segments) + gap_sec
        previous_segment = segment

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


def optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
                timing = str(t.get("timing") or "sequential").strip().lower()
                if timing in {"sequential", "overlap_previous"}:
                    turn["timing"] = timing
                for key in (
                    "start_after_previous_start_sec",
                    "truncate_previous_after_sec",
                    "gain",
                ):
                    value = optional_float(t.get(key))
                    if value is not None:
                        turn[key] = value
                for key in ("voice_role", "event"):
                    value = str(t.get(key) or "").strip()
                    if value:
                        turn[key] = value
                turns.append(turn)
            if not turns:
                logger.warning("対話 %d 行目: turns 抽出失敗、スキップ", i)
                continue
            out.append({
                "id": str(row.get("id") or row.get("source_use_case") or f"dialogue_{i:03d}"),
                "category": str(row.get("category") or "unknown"),
                "risk_level": str(row.get("risk_level") or "low"),
                "title": str(row.get("title") or row.get("id") or f"dialogue {i}"),
                "duplex_task": str(row.get("duplex_task") or "") or None,
                "turns": turns,
            })
    return out


def validate_duplex_dialogue(dialogue: dict[str, Any]) -> list[str]:
    task = str(dialogue.get("duplex_task") or "")
    if not task:
        return []
    turns = list(dialogue.get("turns") or [])
    overlaps = [
        turn for turn in turns if str(turn.get("timing") or "") == "overlap_previous"
    ]
    events = {str(turn.get("event") or ""): turn for turn in turns}
    errors: list[str] = []

    if task == "pause_handling":
        valid_pause = False
        for index, turn in enumerate(turns):
            if (
                turn.get("speaker") == "silence"
                and float(optional_float(turn.get("duration_sec"), 0.0) or 0.0) >= 2.0
                and 0 < index < len(turns) - 1
                and turns[index - 1].get("speaker") == "user"
                and turns[index + 1].get("speaker") == "user"
            ):
                valid_pause = True
                break
        if not valid_pause:
            errors.append(
                "pause_handling requires user -> silence>=2s -> user continuation"
            )
    elif task == "smooth_turn_taking":
        if overlaps:
            errors.append("smooth_turn_taking must not contain overlap_previous")
    elif task == "backchannel":
        turn = events.get("model_backchannel")
        if not turn or turn.get("speaker") != "moshi" or turn not in overlaps:
            errors.append("backchannel requires overlapping moshi event=model_backchannel")
        elif turns.index(turn) == 0 or turns[turns.index(turn) - 1].get("speaker") != "user":
            errors.append("model_backchannel must overlap a preceding user turn")
    elif task == "user_interruption":
        turn = events.get("user_interruption")
        if not turn or turn.get("speaker") != "user" or turn not in overlaps:
            errors.append("user_interruption requires an overlapping user event")
        elif turns.index(turn) == 0 or turns[turns.index(turn) - 1].get("speaker") != "moshi":
            errors.append("user_interruption must overlap a preceding moshi turn")
        elif optional_float(turn.get("truncate_previous_after_sec")) is None:
            errors.append("user_interruption requires truncate_previous_after_sec")
    elif task == "user_backchannel":
        turn = events.get("user_backchannel")
        if not turn or turn.get("speaker") != "user" or turn not in overlaps:
            errors.append("user_backchannel requires an overlapping user event")
        elif turns.index(turn) == 0 or turns[turns.index(turn) - 1].get("speaker") != "moshi":
            errors.append("user_backchannel must overlap a preceding moshi turn")
        elif optional_float(turn.get("truncate_previous_after_sec")) is not None:
            errors.append("user_backchannel must not truncate the model turn")
    elif task == "talking_to_other":
        turn = events.get("talking_to_other")
        if not turn or turn.get("speaker") != "user" or turn not in overlaps:
            errors.append("talking_to_other requires an overlapping user event")
        elif turns.index(turn) == 0 or turns[turns.index(turn) - 1].get("speaker") != "moshi":
            errors.append("talking_to_other must overlap a preceding moshi turn")
    elif task == "background_speech":
        turn = events.get("background_speech")
        if not turn or turn.get("speaker") != "user" or turn not in overlaps:
            errors.append("background_speech requires an overlapping user-channel event")
        elif turns.index(turn) == 0 or turns[turns.index(turn) - 1].get("speaker") != "moshi":
            errors.append("background_speech must overlap a preceding moshi turn")
        elif str(turn.get("voice_role") or "") != "background":
            errors.append("background_speech requires voice_role=background")
        elif float(optional_float(turn.get("gain"), 1.0) or 0.0) >= 0.6:
            errors.append("background_speech gain must be below 0.6")
    else:
        errors.append(f"unknown duplex_task={task!r}")
    return errors


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
        description="日本語対話音声データを生成し Moshi fine-tune フォーマットで保存する"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument(
        "--tts-backend",
        default="qwen3",
        choices=["qwen3", "moss-ttsd"],
        help="TTS backend. Default: qwen3.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS の HuggingFace モデル ID（CustomVoice 系を推奨）",
    )
    parser.add_argument(
        "--moss-model",
        default="OpenMOSS-Team/MOSS-TTSD-v1.0",
        help="MOSS-TTSD Hugging Face model ID.",
    )
    parser.add_argument(
        "--moss-codec-model",
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        help="MOSS audio-tokenizer model ID.",
    )
    parser.add_argument(
        "--moss-refs-json",
        type=Path,
        default=None,
        help="Reference manifest produced by build_moss_reference_voices.py.",
    )
    for role in MossTTSD.ROLES:
        parser.add_argument(
            f"--moss-ref-{role}",
            type=Path,
            default=None,
            help=f"MOSS-TTSD reference WAV for the {role} role.",
        )
        parser.add_argument(
            f"--moss-ref-text-{role}",
            default=None,
            help=f"Exact transcript of --moss-ref-{role}.",
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
    parser.add_argument(
        "--speaker-other",
        default="Dylan",
        help=f"第三者発話に使うプリセット話者。候補: {sorted(VALID_SPEAKERS)}",
    )
    parser.add_argument(
        "--speaker-background",
        default="Ryan",
        help=f"背景発話に使うプリセット話者。候補: {sorted(VALID_SPEAKERS)}",
    )
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
    parser.add_argument(
        "--allow-invalid-duplex",
        action="store_true",
        help="Do not fail when a duplex_task dialogue is missing its required timing event.",
    )
    args = parser.parse_args()

    if args.tts_backend == "qwen3":
        for role, name in [
            ("--speaker-user", args.speaker_user),
            ("--speaker-moshi", args.speaker_moshi),
            ("--speaker-other", args.speaker_other),
            ("--speaker-background", args.speaker_background),
        ]:
            if name not in VALID_SPEAKERS:
                parser.error(
                    f"{role}={name!r} は無効です。候補: {sorted(VALID_SPEAKERS)}"
                )

    pool_raw = args.user_speaker_pool.strip()
    if pool_raw and args.tts_backend == "qwen3":
        pool = [s.strip() for s in pool_raw.split(",") if s.strip()]
        invalid = [s for s in pool if s not in VALID_SPEAKERS]
        if invalid:
            parser.error(
                f"--user-speaker-pool に無効な話者: {invalid}。候補: {sorted(VALID_SPEAKERS)}"
            )
        args.user_speaker_pool_list = pool
    else:
        args.user_speaker_pool_list = [args.speaker_user]

    if args.tts_backend == "moss-ttsd":
        if args.moss_refs_json is not None and not args.moss_refs_json.is_file():
            parser.error(
                f"--moss-refs-json does not exist or is not a file: "
                f"{args.moss_refs_json}"
            )
        if args.moss_refs_json is not None:
            return args
        for role in MossTTSD.ROLES:
            path = getattr(args, f"moss_ref_{role}")
            text = getattr(args, f"moss_ref_text_{role}")
            if path is None:
                parser.error(
                    "--moss-refs-json or all four --moss-ref-* paths are "
                    "required with --tts-backend moss-ttsd"
                )
            if not path.is_file():
                parser.error(f"--moss-ref-{role} does not exist or is not a file: {path}")
            if not text or not text.strip():
                parser.error(
                    f"--moss-ref-text-{role} is required when --moss-ref-{role} is set"
                )
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
    if not args.allow_invalid_duplex:
        validation_errors: list[str] = []
        for template in templates:
            for error in validate_duplex_dialogue(template):
                validation_errors.append(f"{template.get('id', '<unknown>')}: {error}")
        if validation_errors:
            details = "\n".join(f"- {error}" for error in validation_errors[:20])
            raise ValueError(
                "Invalid full-duplex dialogue schema. Regenerate or fix dialogues.jsonl:\n"
                + details
            )
    if args.tts_backend == "moss-ttsd":
        ref_audio_paths, ref_texts = load_moss_references(args)
        tts: Qwen3TTS | MossTTSD = MossTTSD(
            model_name=args.moss_model,
            codec_model_name=args.moss_codec_model,
            ref_audio_paths=ref_audio_paths,
            ref_texts=ref_texts,
            device=args.device,
            dtype=args.dtype,
        )
    else:
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
                timing=str(t.get("timing") or "sequential"),
                start_after_previous_start_sec=optional_float(
                    t.get("start_after_previous_start_sec")
                ),
                truncate_previous_after_sec=optional_float(
                    t.get("truncate_previous_after_sec")
                ),
                gain=max(0.0, float(optional_float(t.get("gain"), 1.0) or 0.0)),
                voice_role=str(t.get("voice_role") or "") or None,
                event=str(t.get("event") or "") or None,
            ))
        dialogue = Dialogue(
            id=safe_stem(tmpl["id"], f"dialogue_{idx:03d}"),
            category=tmpl["category"],
            risk_level=tmpl["risk_level"],
            title=tmpl["title"],
            turns=turns,
            duplex_task=str(tmpl.get("duplex_task") or "") or None,
        )

        user_voice = args.user_speaker_pool_list[(idx - 1) % len(args.user_speaker_pool_list)]
        user_override = "user" if args.tts_backend == "moss-ttsd" else user_voice
        other_override = "other" if args.tts_backend == "moss-ttsd" else args.speaker_other
        background_override = (
            "background" if args.tts_backend == "moss-ttsd"
            else args.speaker_background
        )
        logger.info(
            "[%d/%d] 対話 %s を合成中 (moshi=%s, user=%s) ...",
            idx, len(templates), dialogue.id, args.speaker_moshi, user_voice,
        )
        t0 = time.time()
        segments, silences = build_segments(
            dialogue, tts, args.lead_in_sec, args.gap_sec,
            user_speaker_override=user_override,
            other_speaker_override=other_override,
            background_speaker_override=background_override,
        )
        if not segments or tts.sample_rate == 0:
            # 音声ターンが 0 件（沈黙のみ等）の対話。sample_rate が未確定で
            # ゼロ除算になるため、合成せずスキップする。
            logger.warning(
                "[%d/%d] 対話 %s は音声ターンが無いためスキップします",
                idx, len(templates), dialogue.id,
            )
            continue
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
                "mode": (
                    "moss-ttsd-scripted"
                    if args.tts_backend == "moss-ttsd"
                    else "qwen3-tts-scripted"
                ),
                "sample_rate": tts.sample_rate,
                "duration_sec": round(duration, 4),
                "tts_backend": args.tts_backend,
                "tts_model": (
                    args.moss_model if args.tts_backend == "moss-ttsd"
                    else args.model
                ),
                "tts_codec_model": (
                    args.moss_codec_model
                    if args.tts_backend == "moss-ttsd"
                    else None
                ),
                "language": args.language,
                "speaker_user": user_voice,
                "speaker_user_pool": args.user_speaker_pool_list,
                "speaker_moshi": args.speaker_moshi,
                "speaker_other": args.speaker_other,
                "speaker_background": args.speaker_background,
                "left_channel": "moshi",
                "right_channel": "user",
                "wall_time_sec": round(elapsed, 3),
                "emotion_control": "off" if args.no_emotion else "on",
                "emotion_map_used": {
                    e: emotion_map[e]
                    for e in {t.emotion for t in dialogue.turns if t.emotion}
                },
                "silences": silences,
                "duplex_task": dialogue.duplex_task,
                "duplex_events": [
                    {
                        "event": seg.event,
                        "speaker": seg.speaker,
                        "voice_role": seg.voice_role,
                        "start_sec": round(seg.start_sec, 4),
                        "end_sec": round(seg.end_sec, 4),
                    }
                    for seg in segments
                    if seg.event
                ],
                "dialogue": {
                    "id": dialogue.id,
                    "category": dialogue.category,
                    "risk_level": dialogue.risk_level,
                    "title": dialogue.title,
                    "duplex_task": dialogue.duplex_task,
                    "turns": [asdict(t) for t in dialogue.turns],
                },
            },
        })

        append_jsonl(dialogues_path, {
            "id": dialogue.id,
            "category": dialogue.category,
            "risk_level": dialogue.risk_level,
            "title": dialogue.title,
            "duplex_task": dialogue.duplex_task,
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
