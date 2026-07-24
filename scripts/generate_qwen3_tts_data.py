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
import hashlib
import json
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.alignment_words import get_segmenter_name, split_utterance_alignments
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from alignment_words import get_segmenter_name, split_utterance_alignments

try:
    from scripts.qwen3_tts_vllm_backend import (
        CloneReference,
        SynthesisRequest,
        VLLMQwen3TTS,
    )
except ImportError:
    from qwen3_tts_vllm_backend import (  # type: ignore[no-redef]
        CloneReference,
        SynthesisRequest,
        VLLMQwen3TTS,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 対話テンプレート（分野C想定の短い日本語対話）
# ---------------------------------------------------------------------------

# 感情ラベル → Qwen3-TTS の instruct 文字列。
# 試験的なマッピング。最終的には fine-tune 結果を見て調整する。
EMOTION_PRESETS: dict[str, str] = {
    # 中立
    "neutral":      "自然で落ち着いたトーンで話して",
    # 利用者 (user) 側でよく使う感情
    "hesitant":     "少し言い淀みながら、ためらうように話して",
    "sad":          "声を少し落として、沈んだトーンで静かに話して",
    "lonely":       "寂しさが滲むような、低めの静かなトーンで話して",
    "anxious":      "不安が伝わるように、声を少し震わせ気味に話して",
    "relieved":     "少しほっとした、肩の力が抜けたような穏やかなトーンで話して",
    "grateful":     "感謝の気持ちを込めて、柔らかいトーンで話して",
    # 利用者 (user) 側の強い感情・声の状態
    "tearful":      "涙ぐんで、声を震わせながら話して",
    "sobbing":      "声を詰まらせ、すすり泣きながら、途切れ途切れに話して",
    "high_tension": "やや早口で、テンション高く勢いよく話して",
    "agitated":     "感情が高ぶって落ち着かない様子で、少し早口に話して",
    "withdrawn":    "消え入りそうな小さな声で、ぽつぽつと途切れがちに話して",
    "weary":        "疲れて気だるそうに、ゆっくり力なく話して",
    "irritable":    "苛立ちを抑えきれない、とげのあるトーンで話して",
    "laughing":     "軽く笑いながら、明るいトーンで話して",
    # 応答者 (moshi) 側でよく使う感情
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


# Fixed opening line every training dialogue's moshi turns begin with (unless
# --no-opening-greeting). Moshi actually generating this itself -- rather
# than a canned clip played by some external layer -- is the point: emitting
# it establishes "domain-C window" as grounding
# context in its own generation history before anything else happens.
# Time-of-day-neutral by design (this is a fixed, always-on line, not a
# templated greeting picked per time of day).
OPENING_GREETING_TEXT = "もしもし、こちら孤独孤立相談窓口になります。"

# The greeting is always synthesized with this exact instruct and the fixed
# --speaker-moshi voice, independent of the per-dialogue style preset, so the
# audio is identical in every sample (and can be disk-cached / reused). The
# model should memorize ONE way of saying its opening line, not a
# per-dialogue-styled variant of it.
OPENING_GREETING_INSTRUCT = (
    "電話に出た相談員の第一声として、落ち着いた電話応対のトーンで、"
    "やわらかくはっきりと話して"
)


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


@dataclass
class DialogueRenderJob:
    index: int
    template: dict[str, Any]
    dialogue: Dialogue
    stem: str
    user_voice: str
    user_override: str | None
    other_override: str | None
    background_override: str | None
    instruct_user: str | None
    instruct_moshi: str | None
    resolved_preset_key: str | None
    log_progress: bool


WHOLE_UTTERANCE_STYLE_PRESETS: dict[str, dict[str, str]] = {
    "counseling_anxious": {
        "user": "不安そうに、少し声を震わせながら、ためらいがちにゆっくり話して",
        "moshi": "温かく、相手に寄り添うように、ゆっくり穏やかに話して",
    },
    "counseling_sad": {
        "user": "声を落として、沈んだトーンで静かにゆっくり話して",
        "moshi": "共感を込めて、相手の気持ちを受け止めるように、ゆっくり静かに話して",
    },
    "counseling_hesitant": {
        "user": "少し言い淀みながら、ためらうようにゆっくり話して",
        "moshi": "やさしく語りかけるように、ゆっくり丁寧に話して",
    },
    "counseling_tearful": {
        "user": "涙ぐんで声を震わせながら、途切れがちにゆっくり話して",
        "moshi": "なだめるように、とてもやわらかくゆっくり話して",
    },
    "counseling_weary": {
        "user": "疲れて気だるそうに、力なくゆっくり話して",
        "moshi": "安心させるように、ゆっくり穏やかに話して",
    },
    "counseling_neutral": {
        "user": "自然で落ち着いたトーンで、ゆっくり話して",
        "moshi": "穏やかで落ち着いたトーンで、ゆっくり話して",
    },
    # --- ミラーリング用プリセット。moshi 側の instruct は「相手のテンポ・
    # 感情に声を合わせる」ことを明示する（早口の相手にはテンポよく、沈んだ
    # 相手には静かにゆっくり）。
    "counseling_high_tension": {
        "user": "テンション高く、やや早口で明るく勢いよく話して",
        "moshi": "相手の明るさに合わせて、少し明るくテンポよく、丁寧に相づちを打つように話して",
    },
    "counseling_agitated": {
        "user": "感情が高ぶって落ち着かない様子で、少し早口に話して",
        "moshi": "落ち着いた声のまま、相手のテンポに遅れないように、しっかりと受け止めるトーンで話して",
    },
    "counseling_withdrawn": {
        "user": "消え入りそうな小さな声で、ぽつぽつと途切れがちに話して",
        "moshi": "とても静かに、小さめの声で、相手の間合いに合わせてゆっくり寄り添うように話して",
    },
}


# use case id に埋め込まれた「今日の状態」トークン（build_use_cases.py の
# EMOTIONAL_STATES）→ whole-utterance スタイルプリセット。--style-preset auto
# のときに対話ごとの感情に合った声（テンポ・トーンのミラーリング）を選ぶ。
EMOTIONAL_STATE_TO_PRESET: dict[str, str] = {
    "steady":       "counseling_neutral",
    "low":          "counseling_sad",
    "anxious":      "counseling_anxious",
    "tearful":      "counseling_tearful",
    "high_tension": "counseling_high_tension",
    "agitated":     "counseling_agitated",
    "withdrawn":    "counseling_withdrawn",
    # dialogues.jsonl の emotional_state は日本語ラベル
    # （build_use_cases.py の EMOTIONAL_STATES 第2要素）で入ることがある。
    "落ち着いている":   "counseling_neutral",
    "気分が沈んでいる": "counseling_sad",
    "不安が強い":       "counseling_anxious",
    "涙ぐんでいる":     "counseling_tearful",
    "ハイテンション":   "counseling_high_tension",
    "気持ちが高ぶる":   "counseling_agitated",
    "ふさぎ込んでいる": "counseling_withdrawn",
}

_AGE_BAND_SEGMENT = re.compile(r"^\d+代$")


def resolve_auto_style_preset(
    dialogue_id: str,
    emotional_state_token: str | None = None,
) -> str:
    """--style-preset auto: 対話の「今日の状態」からプリセットを決める。

    優先順位:
    1. dialogues.jsonl の emotional_state フィールド（新形式）
    2. use case id 形式 `{sit}_{conv}_{pers}_{state}_{age}_{gender}_{risk}_...`
       から、年齢帯セグメント（"30代" など）の直前を state トークンとして読む。
       （"low" はリスクレベルにも現れるため、単純な部分一致では判定しない。）
    3. どちらも取れなければ counseling_neutral。
    """
    token = (emotional_state_token or "").strip()
    if token in EMOTIONAL_STATE_TO_PRESET:
        return EMOTIONAL_STATE_TO_PRESET[token]
    segments = dialogue_id.split("_")
    for i, segment in enumerate(segments):
        if _AGE_BAND_SEGMENT.match(segment) and i > 0:
            # state トークンは "high_tension" のようにアンダースコアを含む
            # ことがあるので、直前 1〜2 セグメントを試す。
            for span in (2, 1):
                if i - span >= 0:
                    candidate = "_".join(segments[i - span : i])
                    if candidate in EMOTIONAL_STATE_TO_PRESET:
                        return EMOTIONAL_STATE_TO_PRESET[candidate]
            break
    return "counseling_neutral"


# 相手の発話に重ねてよい短いあいづちの検出セット。生成プロンプト側の推奨
# 語彙（generate_synthetic_moshi_training_data.py）より広い上位集合にして、
# 旧データ（なるほど。等）もそのまま重ねられるようにしている。
AIZUCHI_OVERLAP_TEXTS = frozenset({
    "はい。",
    "ええ。",
    "えぇ。",
    "ええ、ええ。",
    "あぁ…。",
    "あー…。",
    "そうでしたか。",
    "そうだったんですね。",
    "そうなんですね。",
    "なるほど。",
})
AIZUCHI_OVERLAP_START_AFTER_PREVIOUS_START_SEC = 999


def apply_aizuchi_overlap(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_turns: list[dict[str, Any]] = []
    previous_turn: dict[str, Any] | None = None
    for turn in turns:
        updated_turn = dict(turn)
        if (
            turn.get("speaker") == "moshi"
            and turn.get("text") in AIZUCHI_OVERLAP_TEXTS
            and previous_turn is not None
            and previous_turn.get("speaker") == "user"
        ):
            updated_turn["timing"] = "overlap_previous"
            updated_turn["start_after_previous_start_sec"] = (
                AIZUCHI_OVERLAP_START_AFTER_PREVIOUS_START_SEC
            )
            updated_turn.pop("truncate_previous_after_sec", None)
        updated_turns.append(updated_turn)
        previous_turn = turn
    return updated_turns


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

        # Per-call log: whole-utterance mode calls this many times per
        # dialogue (once per max-chars chunk per speaker), so at INFO this
        # dominates shard log size on large runs. debug-only; --verbose
        # re-enables it.
        logger.debug(
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

        # Per-call log; see the matching Qwen3-TTS.synthesize() note above.
        logger.debug(
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
# Forced Alignment (--whole-utterance mode)
# ---------------------------------------------------------------------------

class ForcedAlignmentError(RuntimeError):
    """Raised when CTC forced alignment fails and proportional fallback is disabled."""


class ForcedAligner:
    """CTC forced alignment using torchaudio MMS_FA for segment boundary detection.

    fallback_mode:
        "skip" (default): アライメント失敗時に ForcedAlignmentError を送出する。
            呼び出し側（対話ループ）が対話ごと破棄するので、境界が狂った
            （＝ターンが単語の途中で切れた）音声が学習データに混入しない。
        "proportional": 旧挙動。文字数比例の推定境界で続行する。境界品質が
            大きく落ちるため、デバッグ・スモーク用途のみ推奨。
    """

    def __init__(self, device: str, fallback_mode: str = "skip"):
        if fallback_mode not in {"skip", "proportional"}:
            raise ValueError(
                f"fallback_mode must be 'skip' or 'proportional', got {fallback_mode!r}"
            )
        self.fallback_mode = fallback_mode
        self.device = "cuda:0" if device == "cuda" else device
        self._model: Any = None
        self._tokenizer: Any = None
        self._aligner: Any = None
        self._sample_rate: int = 16000
        self._vocab: set[str] = set()
        self._romanizer: Any = None
        self._romanizer_kind: str = ""

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        self._model = bundle.get_model().to(self.device)
        self._model.eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        self._sample_rate = bundle.sample_rate
        self._vocab = set(bundle.get_dict().keys())
        logger.info(
            "ForcedAligner (MMS_FA) loaded, device=%s, sr=%d, vocab_size=%d",
            self.device, self._sample_rate, len(self._vocab),
        )

        try:
            import uroman as ur
            self._romanizer = ur.Uroman()
            self._romanizer_kind = "uroman"
            logger.info("Romanizer: uroman")
        except ImportError:
            try:
                import pykakasi
                self._romanizer = pykakasi.kakasi()
                self._romanizer_kind = "pykakasi"
                logger.info("Romanizer: pykakasi (uroman not found)")
            except ImportError:
                raise RuntimeError(
                    "--whole-utterance requires Japanese romanization. "
                    "Install: pip install uroman  or  pip install pykakasi"
                )

    def _romanize(self, text: str) -> str:
        if self._romanizer_kind == "uroman":
            return self._romanizer.romanize_string(text, lcode="jpn")
        result = self._romanizer.convert(text)
        return " ".join(item["hepburn"] for item in result)

    def _clean(self, romanized: str) -> str:
        text = romanized.lower().strip()
        return "".join(c for c in text if c in self._vocab)

    def _fallback_proportional(
        self,
        audio_dur: float,
        seg_char_counts: list[int],
    ) -> list[tuple[float, float]]:
        total_chars = max(sum(seg_char_counts), 1)
        cursor = 0.0
        result: list[tuple[float, float]] = []
        for count in seg_char_counts:
            seg_dur = audio_dur * max(count, 1) / total_chars
            result.append((cursor, cursor + seg_dur))
            cursor += seg_dur
        return result

    @staticmethod
    def expand_to_midpoints(
        tight: list[tuple[float, float]],
        audio_dur: float,
    ) -> list[tuple[float, float]]:
        """Expand tight speech boundaries to midpoints between adjacent segments."""
        expanded: list[tuple[float, float]] = []
        for i, (seg_start, seg_end) in enumerate(tight):
            if seg_start == seg_end:
                expanded.append((seg_start, seg_end))
                continue
            cut_start = 0.0 if i == 0 else (tight[i - 1][1] + seg_start) / 2
            cut_end = audio_dur if i == len(tight) - 1 else (seg_end + tight[i + 1][0]) / 2
            expanded.append((cut_start, cut_end))
        return expanded

    def align(
        self,
        audio: np.ndarray,
        audio_sr: int,
        segment_texts: list[str],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Given a single long audio produced from concatenated segment_texts,
        return (expanded, tight) boundary pairs for each segment.
        - expanded: midpoint-expanded boundaries (for slicing with natural pauses)
        - tight: speech-only boundaries (for internal positioning)
        """
        self.load()
        assert self._model is not None
        import torch
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        if audio_sr != self._sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, audio_sr, self._sample_rate,
            )

        cleaned_per_seg: list[str] = []
        for text in segment_texts:
            romanized = self._romanize(text)
            cleaned = self._clean(romanized)
            cleaned_per_seg.append(cleaned)
            logger.debug("FA: %r → %r", text[:30], cleaned[:40])

        seg_char_counts = [len(c) for c in cleaned_per_seg]
        full_cleaned = "".join(cleaned_per_seg)
        audio_dur = audio.size / audio_sr

        if not full_cleaned or sum(seg_char_counts) == 0:
            if self.fallback_mode != "proportional":
                raise ForcedAlignmentError(
                    "romanization produced no alignable characters"
                )
            logger.warning("FA: romanization produced no alignable characters, using proportional fallback")
            fb = self._fallback_proportional(audio_dur, seg_char_counts)
            return fb, fb

        tokens = self._tokenizer(full_cleaned)

        with torch.inference_mode():
            emission, _ = self._model(waveform.to(self.device))

        try:
            token_spans_grouped = self._aligner(emission[0], tokens)
        except Exception as exc:
            if self.fallback_mode != "proportional":
                raise ForcedAlignmentError(f"CTC alignment failed: {exc}") from exc
            logger.warning(
                "FA CTC alignment failed (%s), using proportional fallback",
                exc,
            )
            fb = self._fallback_proportional(audio_dur, seg_char_counts)
            return fb, fb

        all_spans = [span for group in token_spans_grouped for span in group]

        expected_total = sum(seg_char_counts)
        if len(all_spans) != expected_total:
            if self.fallback_mode != "proportional":
                raise ForcedAlignmentError(
                    f"span count mismatch: expected {expected_total}, got {len(all_spans)}"
                )
            logger.warning(
                "FA span count mismatch: expected %d, got %d; using proportional fallback",
                expected_total, len(all_spans),
            )
            fb = self._fallback_proportional(audio_dur, seg_char_counts)
            return fb, fb

        ratio = waveform.size(1) / emission.size(1) / self._sample_rate

        tight: list[tuple[float, float]] = []
        span_idx = 0
        for count in seg_char_counts:
            if count == 0:
                prev_end = tight[-1][1] if tight else 0.0
                tight.append((prev_end, prev_end))
                continue
            seg_spans = all_spans[span_idx : span_idx + count]
            span_idx += count
            start_sec = seg_spans[0].start * ratio
            end_sec = seg_spans[-1].end * ratio
            tight.append((start_sec, end_sec))

        expanded = self.expand_to_midpoints(tight, audio_dur)

        logger.info(
            "FA aligned %d segments, total_spans=%d, audio=%.2fs",
            len(expanded), len(all_spans), audio_dur,
        )
        return expanded, tight


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


def synthesize_opening_greeting(
    tts: Qwen3TTS | MossTTSD,
    text: str,
    instruct: str | None,
    cache_dir: Path | None,
    backend: str,
    model_id: str,
    speaker: str,
) -> np.ndarray:
    """固定の冒頭挨拶を1回だけ合成し、全対話で同じ音声を使い回す。

    声色を固定するため、対話ごとの style preset ではなく固定 instruct
    （OPENING_GREETING_INSTRUCT）と固定の moshi 話者で合成する。音声を
    変えうる全パラメータをキーにディスクへキャッシュするので、run や
    shard をまたいでもビット同一の挨拶になる。
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        key_src = "|".join([text, backend, model_id, speaker, instruct or ""])
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"greeting_{key}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            pcm = np.asarray(data["pcm"], dtype=np.float32)
            cached_rate = int(data["sample_rate"])
            if tts.sample_rate == 0:
                tts.sample_rate = cached_rate
            elif cached_rate != tts.sample_rate:
                pcm = _resample(pcm, cached_rate, tts.sample_rate)
            logger.info("冒頭挨拶: キャッシュを再利用 -> %s", cache_path)
            return pcm
    pcm = np.asarray(
        tts.synthesize(text, "moshi", instruct=instruct), dtype=np.float32
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, pcm=pcm, sample_rate=tts.sample_rate)
        logger.info("冒頭挨拶: 合成してキャッシュに保存 -> %s", cache_path)
    return pcm


def build_segments(
    dialogue: Dialogue,
    tts: Qwen3TTS | MossTTSD,
    lead_in_sec: float,
    gap_sec: float,
    user_speaker_override: str | None = None,
    other_speaker_override: str | None = None,
    background_speaker_override: str | None = None,
    opening_greeting_pcm: np.ndarray | None = None,
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
        if turn.event == "opening_greeting" and opening_greeting_pcm is not None:
            pcm = opening_greeting_pcm.copy()
        else:
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


def _find_energy_valley(
    pcm: np.ndarray,
    sample_rate: int,
    min_ratio: float = 0.5,
) -> float:
    """Return the time (in seconds from pcm start) of the lowest-energy
    50ms frame in the latter portion of *pcm*.  Used to place backchannels
    at a natural pause/breath point inside a preceding turn."""
    frame_len = int(0.05 * sample_rate)
    if pcm.size < frame_len * 3:
        return pcm.size / sample_rate * 0.7
    n_frames = pcm.size // frame_len
    energy = np.array([
        np.mean(pcm[i * frame_len : (i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    start_frame = max(1, int(n_frames * min_ratio))
    search = energy[start_frame:]
    if search.size == 0:
        return pcm.size / sample_rate * 0.7
    valley_frame = start_frame + int(np.argmin(search))
    return valley_frame * frame_len / sample_rate


def _chunk_indices_by_chars(
    indices: list[int],
    texts: list[str],
    max_chars: int,
) -> list[list[int]]:
    """Split *indices* into consecutive chunks whose concatenated text stays
    under *max_chars*. Keeps at least one index per chunk even if a single
    turn's text alone exceeds the limit. Chunking bounds how much text a
    single TTS call must render, which keeps the synthesized audio length
    in proportion to the text and avoids CTC forced-alignment failures
    ("target_length is too long for CTC") that occur when a very long
    concatenated utterance gets truncated by the TTS model's own generation
    limit relative to the full text."""
    if max_chars <= 0:
        return [list(indices)] if indices else []
    chunks: list[list[int]] = []
    current: list[int] = []
    current_len = 0
    for idx, text in zip(indices, texts):
        tlen = len(text)
        if current and current_len + tlen > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(idx)
        current_len += tlen
    if current:
        chunks.append(current)
    return chunks


def plan_whole_utterance_synthesis(
    dialogue: Dialogue,
    *,
    user_speaker_override: str | None = None,
    other_speaker_override: str | None = None,
    background_speaker_override: str | None = None,
    instruct_user: str | None = None,
    instruct_moshi: str | None = None,
    max_chars_per_synthesis: int = 150,
    opening_greeting_pcm: np.ndarray | None = None,
) -> list[SynthesisRequest]:
    """Return TTS calls in exactly the order used by whole-utterance rendering.

    vLLM-Omni evaluates these calls across several dialogues as a batch.  The
    resulting audio is replayed through ``build_segments_whole_utterance`` so
    all existing alignment, overlap, and metadata behavior stays unchanged.
    """
    concat_groups: dict[str, list[int]] = {"user": [], "moshi": []}
    per_segment_indices: list[int] = []
    greeting_indices: set[int] = set()

    for i, turn in enumerate(dialogue.turns):
        if turn.speaker == "silence":
            continue
        if turn.event == "opening_greeting":
            greeting_indices.add(i)
            per_segment_indices.append(i)
        elif turn.speaker in ("user", "moshi") and not turn.voice_role:
            concat_groups[turn.speaker].append(i)
        else:
            per_segment_indices.append(i)

    requests: list[SynthesisRequest] = []
    for speaker, indices in concat_groups.items():
        if not indices:
            continue
        all_texts = [dialogue.turns[i].text for i in indices]
        override = user_speaker_override if speaker == "user" else None
        instruct = instruct_user if speaker == "user" else instruct_moshi
        for chunk_indices in _chunk_indices_by_chars(
            indices, all_texts, max_chars_per_synthesis
        ):
            requests.append(
                SynthesisRequest(
                    text="".join(dialogue.turns[i].text for i in chunk_indices),
                    speaker_role=speaker,
                    instruct=instruct,
                    speaker_override=override,
                )
            )

    for i in per_segment_indices:
        turn = dialogue.turns[i]
        if i in greeting_indices:
            if opening_greeting_pcm is None:
                requests.append(
                    SynthesisRequest(
                        text=turn.text,
                        speaker_role="moshi",
                        instruct=OPENING_GREETING_INSTRUCT,
                    )
                )
            continue
        override = None
        if turn.speaker == "user":
            if turn.voice_role == "background":
                override = background_speaker_override or user_speaker_override
            elif turn.voice_role == "other":
                override = other_speaker_override or user_speaker_override
            else:
                override = user_speaker_override
        requests.append(
            SynthesisRequest(
                text=turn.text,
                speaker_role=turn.speaker,
                instruct=turn.instruct,
                speaker_override=override,
            )
        )
    return requests


class ReplayTTS:
    """TTS facade that replays an already-generated dialogue request list."""

    def __init__(
        self,
        requests: list[SynthesisRequest],
        audio: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        if len(requests) != len(audio):
            raise ValueError("request/audio count mismatch in batched TTS replay")
        self._requests = requests
        self._audio = audio
        self._position = 0
        self.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        if self._position >= len(self._requests):
            raise RuntimeError("whole-utterance render requested more audio than planned")
        actual = SynthesisRequest(text, speaker_role, instruct, speaker_override)
        expected = self._requests[self._position]
        if actual != expected:
            raise RuntimeError(
                "whole-utterance synthesis plan diverged during replay: "
                f"expected={expected!r}, actual={actual!r}"
            )
        audio = self._audio[self._position]
        self._position += 1
        return audio.copy()

    def assert_consumed(self) -> None:
        if self._position != len(self._requests):
            raise RuntimeError(
                f"whole-utterance replay left {len(self._requests) - self._position} audio item(s) unused"
            )


def build_segments_whole_utterance(
    dialogue: Dialogue,
    tts: Qwen3TTS | MossTTSD,
    aligner: ForcedAligner,
    lead_in_sec: float,
    gap_sec: float,
    user_speaker_override: str | None = None,
    other_speaker_override: str | None = None,
    background_speaker_override: str | None = None,
    instruct_user: str | None = None,
    instruct_moshi: str | None = None,
    max_chars_per_synthesis: int = 150,
    opening_greeting_pcm: np.ndarray | None = None,
) -> tuple[list[AudioSegment], list[dict[str, Any]]]:
    """
    Whole-utterance TTS with natural aizuchi overlap.

    Adjacent user turns bridged only by aizuchi are merged into a single
    continuous segment so the user's speech never stops unnaturally.
    The aizuchi is overlaid at the internal clause boundary.
    """
    # --- 1. Classify turns ------------------------------------------------
    concat_groups: dict[str, list[int]] = {"user": [], "moshi": []}
    per_segment_indices: list[int] = []
    greeting_indices: set[int] = set()

    for i, turn in enumerate(dialogue.turns):
        if turn.speaker == "silence":
            continue
        if turn.event == "opening_greeting":
            # 固定挨拶は moshi の連結合成に混ぜない。混ぜると対話ごとの
            # style preset で韻律が変わってしまい、「毎回同じ声・同じ
            # 言い方の第一声」という学習目標が崩れる。
            greeting_indices.add(i)
            per_segment_indices.append(i)
        elif turn.speaker in ("user", "moshi") and not turn.voice_role:
            concat_groups[turn.speaker].append(i)
        else:
            per_segment_indices.append(i)

    # --- 2. Identify aizuchi turns ----------------------------------------
    aizuchi_indices: set[int] = set()
    for i, turn in enumerate(dialogue.turns):
        if (
            turn.timing == "overlap_previous"
            and turn.speaker == "moshi"
            and turn.truncate_previous_after_sec is None
        ):
            aizuchi_indices.add(i)

    # --- 3. Concatenated TTS + alignment per speaker ----------------------
    speaker_audio: dict[str, np.ndarray] = {}
    turn_tight: dict[int, tuple[float, float]] = {}
    turn_expanded: dict[int, tuple[float, float]] = {}

    for speaker, indices in concat_groups.items():
        if not indices:
            continue
        all_texts = [dialogue.turns[i].text for i in indices]
        override = user_speaker_override if speaker == "user" else None
        instruct = instruct_user if speaker == "user" else instruct_moshi

        chunks = _chunk_indices_by_chars(indices, all_texts, max_chars_per_synthesis)
        # Fires once per speaker per dialogue; debug-only to keep shard logs
        # from growing unbounded on large sweeps (see synthesize() above).
        logger.debug(
            "whole-utterance TTS: speaker=%s, %d turns, %d chars, %d chunk(s), instruct=%r",
            speaker, len(all_texts), sum(len(t) for t in all_texts), len(chunks),
            (instruct or "")[:30],
        )

        audio_parts: list[np.ndarray] = []
        time_offset = 0.0
        for chunk_indices in chunks:
            chunk_texts = [dialogue.turns[i].text for i in chunk_indices]
            chunk_text = "".join(chunk_texts)
            audio = tts.synthesize(
                chunk_text, speaker, instruct=instruct, speaker_override=override,
            )
            audio_parts.append(audio)

            expanded, tight = aligner.align(audio, tts.sample_rate, chunk_texts)
            for idx, exp, tgt in zip(chunk_indices, expanded, tight):
                turn_expanded[idx] = (exp[0] + time_offset, exp[1] + time_offset)
                turn_tight[idx] = (tgt[0] + time_offset, tgt[1] + time_offset)
            time_offset += audio.size / tts.sample_rate

        speaker_audio[speaker] = (
            np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        )

    # --- 4. Build user merge groups (bridged by aizuchi) ------------------
    user_indices = concat_groups["user"]
    merge_groups: list[list[int]] = []
    current_group: list[int] = []
    for turn_idx in user_indices:
        if not current_group:
            current_group = [turn_idx]
            continue
        prev_idx = current_group[-1]
        has_silence = any(
            dialogue.turns[j].speaker == "silence"
            for j in range(prev_idx + 1, turn_idx)
        )
        bridged_by_aizuchi = (
            not has_silence
            and all(
                j in aizuchi_indices
                for j in range(prev_idx + 1, turn_idx)
                if dialogue.turns[j].speaker != "silence"
            )
        )
        if bridged_by_aizuchi:
            current_group.append(turn_idx)
        else:
            merge_groups.append(current_group)
            current_group = [turn_idx]
    if current_group:
        merge_groups.append(current_group)

    # --- 5. Build PCM slices with merge-aware slicing ---------------------
    pcm_slices: dict[int, np.ndarray] = {}
    merged_turns: set[int] = set()
    merged_text: dict[int, str] = {}
    aizuchi_overlay_sec: dict[int, float] = {}

    fade_samples = int(0.01 * tts.sample_rate)
    user_audio = speaker_audio.get("user")

    def _apply_fade(pcm: np.ndarray) -> np.ndarray:
        if pcm.size > 2 * fade_samples and fade_samples > 0:
            pcm[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
            pcm[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
        return pcm

    for group in merge_groups:
        if user_audio is None:
            break
        if len(group) == 1:
            idx = group[0]
            s, e = turn_expanded[idx]
            st = max(0, int(round(s * tts.sample_rate)))
            en = min(user_audio.size, int(round(e * tts.sample_rate)))
            pcm_slices[idx] = _apply_fade(user_audio[st:en].copy())
        else:
            primary = group[0]
            first_exp_start = turn_expanded[primary][0]
            last_exp_end = turn_expanded[group[-1]][1]
            st = max(0, int(round(first_exp_start * tts.sample_rate)))
            en = min(user_audio.size, int(round(last_exp_end * tts.sample_rate)))
            pcm_slices[primary] = _apply_fade(user_audio[st:en].copy())
            merged_text[primary] = "".join(
                dialogue.turns[idx].text for idx in group
            )
            for secondary in group[1:]:
                merged_turns.add(secondary)
            for gi in range(len(group) - 1):
                curr_idx, next_idx = group[gi], group[gi + 1]
                _, curr_tight_end = turn_tight[curr_idx]
                overlap_in_merged = curr_tight_end - first_exp_start
                for a in range(curr_idx + 1, next_idx):
                    if a in aizuchi_indices:
                        aizuchi_overlay_sec[a] = overlap_in_merged
            logger.debug(
                "merged user turns %s into one segment (%.2fs)",
                group, (last_exp_end - first_exp_start),
            )

    moshi_audio = speaker_audio.get("moshi")
    if moshi_audio is not None:
        for idx in concat_groups["moshi"]:
            s, e = turn_expanded[idx]
            st = max(0, int(round(s * tts.sample_rate)))
            en = min(moshi_audio.size, int(round(e * tts.sample_rate)))
            pcm_slices[idx] = _apply_fade(moshi_audio[st:en].copy())

    for i in per_segment_indices:
        turn = dialogue.turns[i]
        if i in greeting_indices:
            if opening_greeting_pcm is not None:
                pcm_slices[i] = opening_greeting_pcm.copy()
            else:
                # 事前合成が無い場合も声色は固定 instruct で合成する
                pcm_slices[i] = tts.synthesize(
                    turn.text, "moshi", instruct=OPENING_GREETING_INSTRUCT,
                )
            continue
        override = None
        if turn.speaker == "user":
            if turn.voice_role == "background":
                override = background_speaker_override or user_speaker_override
            elif turn.voice_role == "other":
                override = other_speaker_override or user_speaker_override
            else:
                override = user_speaker_override
        pcm_slices[i] = tts.synthesize(
            turn.text, turn.speaker,
            instruct=turn.instruct,
            speaker_override=override,
        )

    # --- 6. Timeline placement --------------------------------------------
    cursor = lead_in_sec
    segments: list[AudioSegment] = []
    silences: list[dict[str, Any]] = []
    previous_segment: AudioSegment | None = None

    for i, turn in enumerate(dialogue.turns):
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

        if i in merged_turns:
            continue

        pcm = pcm_slices.get(i)
        if pcm is None or pcm.size == 0:
            logger.warning("whole-utterance: turn %d has no audio, skipping", i)
            continue

        gain = max(0.0, float(turn.gain))
        if gain != 1.0:
            pcm = np.asarray(pcm, dtype=np.float32) * gain

        if turn.timing == "overlap_previous" and previous_segment is not None:
            if i in aizuchi_overlay_sec and previous_segment.speaker != "moshi":
                offset = aizuchi_overlay_sec[i]
                start = previous_segment.start_sec + offset
                logger.debug(
                    "aizuchi %r overlays at %.2fs into merged user segment",
                    turn.text[:10], offset,
                )
            else:
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
        text = merged_text.get(i, turn.text)
        segment = AudioSegment(
            speaker=turn.speaker,
            label="SPEAKER_MAIN" if turn.speaker == "moshi" else "SPEAKER_USER",
            text=text,
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
    """外部 (LLM) の dialogues.jsonl を読む。TEMPLATE_DIALOGUES と同じ形に整える。"""
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
                # --style-preset auto 用（新形式 dialogues.jsonl のみ持つ）
                "emotional_state": str(row.get("emotional_state") or "") or None,
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


def rebuild_from_completed(
    data_dir: Path,
    manifest_path: Path,
    dialogues_path: Path,
    done_stems: set[str],
) -> int:
    """Resume support: scan already-rendered stereo samples and rebuild the
    manifest / dialogues.jsonl from them so a re-run appends instead of
    starting over.

    A sample counts as complete only when both its WAV and sidecar JSON exist
    and the JSON parses (the JSON is written right after the WAV, and the
    manifest/dialogues rows are appended after that). Rebuilding the two JSONL
    files from the surviving sidecars makes them exactly consistent with the
    rendered audio regardless of where a previous run was killed (e.g. by a
    walltime limit) -- a half-written trailing manifest row is discarded and
    the corresponding sample is simply re-rendered. Populates ``done_stems``
    with the sample stems already on disk and returns their count."""
    out_dir = manifest_path.parent
    manifest_rows: list[tuple[str, str]] = []
    dialogue_rows: list[tuple[str, str]] = []
    for json_path in sorted(data_dir.glob("*.json")):
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            meta = payload["metadata"]
            duration = float(meta["duration_sec"])
            dialogue = meta["dialogue"]
        except (OSError, ValueError, KeyError):
            # Corrupt/partial sidecar -> not done; it will be re-rendered.
            continue
        stem = json_path.stem
        done_stems.add(stem)
        rel = str(wav_path.relative_to(out_dir)).replace("\\", "/")
        manifest_rows.append(
            (stem, json.dumps({"path": rel, "duration": duration}, ensure_ascii=False))
        )
        dialogue_rows.append((stem, json.dumps(dialogue, ensure_ascii=False)))

    manifest_rows.sort()
    dialogue_rows.sort()
    manifest_path.write_text(
        "".join(row + "\n" for _, row in manifest_rows), encoding="utf-8"
    )
    dialogues_path.write_text(
        "".join(row + "\n" for _, row in dialogue_rows), encoding="utf-8"
    )
    return len(done_stems)


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
        choices=["qwen3", "moss-ttsd", "kokoro"],
        help=(
            "TTS backend. Default: qwen3. kokoro は軽量・高速だが emotion/style "
            "instruct 非対応（無視される）。"
        ),
    )
    parser.add_argument(
        "--kokoro-model", default="hexgrad/Kokoro-82M",
        help="Kokoro の HuggingFace モデル ID",
    )
    parser.add_argument(
        "--kokoro-voice-moshi", default="jf_alpha",
        help="kokoro backend の moshi（相談員）話者",
    )
    parser.add_argument(
        "--kokoro-user-voices",
        default="jf_gongitsune,jf_nezumi,jf_tebukuro,jm_kumo",
        help=(
            "kokoro backend の user 話者プール（カンマ区切り）。対話ごとに"
            "ラウンドロビンで切り替える。日本語話者は jf_alpha / jf_gongitsune / "
            "jf_nezumi / jf_tebukuro / jm_kumo の5種。"
        ),
    )
    parser.add_argument(
        "--kokoro-voice-other", default="jf_gongitsune",
        help="kokoro backend の other（第三者）話者",
    )
    parser.add_argument(
        "--kokoro-voice-background", default="jf_tebukuro",
        help="kokoro backend の background 話者",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS の HuggingFace モデル ID（CustomVoice 系を推奨）",
    )
    parser.add_argument(
        "--qwen-engine",
        default="transformers",
        choices=["transformers", "vllm-omni"],
        help="Qwen3-TTS inference engine. vllm-omni enables cross-dialogue batching.",
    )
    parser.add_argument(
        "--tts-batch-size",
        type=int,
        default=16,
        help="vLLM-Omni requests per GPU batch; must be a power of two.",
    )
    parser.add_argument(
        "--dialogue-batch-size",
        type=int,
        default=16,
        help="Dialogues planned together so their TTS calls can share vLLM batches.",
    )
    parser.add_argument(
        "--vllm-stage-config",
        type=Path,
        default=Path("configs/qwen3_tts_v100_batch16.yaml"),
        help="vLLM-Omni stage/deploy YAML (resolved from the current working directory).",
    )
    parser.add_argument(
        "--vllm-max-new-tokens",
        type=int,
        default=2048,
        help="Maximum Qwen3-TTS talker tokens per request.",
    )
    parser.add_argument(
        "--qwen-clone-ref-audio-user",
        default=None,
        help="Voice-clone reference WAV for the user role (vllm-omni engine + "
             "Base model). Enables clone mode; both roles need a reference.",
    )
    parser.add_argument(
        "--qwen-clone-ref-text-user",
        default=None,
        help="Transcript of --qwen-clone-ref-audio-user (required unless "
             "--qwen-clone-x-vector-only).",
    )
    parser.add_argument(
        "--qwen-clone-ref-audio-moshi",
        default=None,
        help="Voice-clone reference WAV for the moshi role.",
    )
    parser.add_argument(
        "--qwen-clone-ref-text-moshi",
        default=None,
        help="Transcript of --qwen-clone-ref-audio-moshi (required unless "
             "--qwen-clone-x-vector-only).",
    )
    parser.add_argument(
        "--qwen-clone-x-vector-only",
        action="store_true",
        help="Clone timbre only from an x-vector (no in-context prosody transfer; "
             "ref texts become optional).",
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
            "外部対話 JSONL（LLM が出した dialogues.jsonl など）を読む。"
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
    parser.add_argument(
        "--auto-overlap-aizuchi",
        action="store_true",
        help="Automatically overlap short moshi backchannels with the preceding user turn.",
    )
    parser.add_argument(
        "--opening-greeting",
        default=OPENING_GREETING_TEXT,
        help=(
            "Fixed moshi turn prepended to every dialogue's turns before synthesis "
            "(training grounding for a memorized session-opening line). "
            "Pass '' or --no-opening-greeting to disable."
        ),
    )
    parser.add_argument(
        "--no-opening-greeting",
        action="store_true",
        help="Do not prepend the fixed opening-greeting moshi turn.",
    )
    parser.add_argument(
        "--opening-greeting-instruct",
        default=OPENING_GREETING_INSTRUCT,
        help=(
            "Fixed style instruct for the opening greeting. Always used instead "
            "of the per-dialogue style preset so the greeting voice never varies."
        ),
    )
    parser.add_argument(
        "--opening-greeting-cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / ".cache" / "opening_greeting",
        help=(
            "Synthesize the fixed greeting once and reuse the cached audio "
            "across dialogues, runs and shards (keyed by text/backend/model/"
            "speaker/instruct). Pass '' to disable caching."
        ),
    )
    parser.add_argument(
        "--whole-utterance",
        action="store_true",
        help=(
            "話者ごとに全発話を連結して1回のTTSで合成し、MMS_FA Forced Alignment で"
            "セグメント境界を復元する。韻律の一貫性が向上する。"
            "uroman または pykakasi が必要（pip install uroman）。"
        ),
    )
    _preset_names = sorted(WHOLE_UTTERANCE_STYLE_PRESETS.keys())
    parser.add_argument(
        "--style-preset",
        default=None,
        help=(
            "whole-utterance モード用のスタイルプリセット名。"
            f"候補: {_preset_names}, none（style instructなし、素の話速）, "
            "random（対話ごとにランダム選択）, "
            "auto（dialogues.jsonl の emotional_state または use case id の"
            "状態トークンから対話ごとに自動選択し、user と moshi の声の"
            "テンポ・感情をミラーリングさせる）。"
            "指定すると --instruct-user/--instruct-moshi より優先される。"
        ),
    )
    parser.add_argument(
        "--whole-utterance-max-chars",
        type=int,
        default=150,
        help=(
            "whole-utterance モードで話者ごとに連結するテキストの1回のTTS呼び出し"
            "あたりの最大文字数。長い対話を分割合成することで、TTSモデルの生成長"
            "上限による音声打ち切りとそれに伴う CTC alignment の "
            "'target_length is too long for CTC' 失敗を防ぐ。0以下で分割無効。"
        ),
    )
    parser.add_argument(
        "--fa-fallback",
        choices=["skip", "proportional"],
        default="skip",
        help=(
            "MMS_FA Forced Alignment 失敗時の挙動。skip（既定）は失敗した対話を"
            "破棄する（境界が狂った音声を学習データに混ぜない）。proportional は"
            "旧挙動で、文字数比例の推定境界のまま続行する（デバッグ用）。"
            "--success-target と併用すれば破棄分は予備対話で自動補充される。"
        ),
    )
    parser.add_argument(
        "--success-target",
        type=int,
        default=None,
        help=(
            "このシャードで成功させたい対話数。指定すると、個々の対話合成が"
            "例外で失敗してもシャード全体は継続し、次の対話（予備分含む）で"
            "補って目標数に到達し次第打ち切る。未指定時は渡された対話を"
            "全件処理する（失敗しても続行するが目標判定はしない）。"
            "全件処理後も目標に届かない場合はエラー終了する。"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "既存の out-dir を再利用し、data_stereo に WAV+JSON が揃っている"
            "対話をスキップして未完了分だけを合成する。manifest と "
            "dialogues.jsonl は完了済みサンプルから再構築してから追記するため、"
            "walltime 到達などで打ち切られたジョブを同じ out-dir に再投入すると"
            "途中から継続できる。未指定時は従来どおり out-dir を初期化して"
            "最初から生成する。"
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help=(
            "対話ごとの進捗ログ（開始/保存完了）を N 件おきに間引く"
            "（既定20）。1件目・最後の1件・失敗は常にログする。"
            "1以下を指定すると全件ログする。"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "TTS呼び出し単位の詳細ログ（合成完了/チャンク分割/ターン結合）"
            "をDEBUGからINFOに戻す。大規模実行では既定でDEBUGに落として"
            "shardログの肥大化を防いでいる。"
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    if args.qwen_engine == "vllm-omni":
        if args.tts_backend != "qwen3":
            parser.error("--qwen-engine vllm-omni requires --tts-backend qwen3")
        if not args.whole_utterance:
            parser.error("--qwen-engine vllm-omni currently requires --whole-utterance")
        if args.device != "cuda":
            parser.error("--qwen-engine vllm-omni requires --device cuda")
        if args.dtype != "float16":
            parser.error("V100 vLLM-Omni jobs require --dtype float16")
        if args.tts_batch_size < 1 or args.tts_batch_size & (args.tts_batch_size - 1):
            parser.error("--tts-batch-size must be a positive power of two")
        if args.dialogue_batch_size < 1:
            parser.error("--dialogue-batch-size must be positive")

    # Voice clone (vllm-omni Base task): validate the reference set.
    clone_args = [
        args.qwen_clone_ref_audio_user, args.qwen_clone_ref_text_user,
        args.qwen_clone_ref_audio_moshi, args.qwen_clone_ref_text_moshi,
    ]
    args.qwen_clone_enabled = any(a is not None for a in clone_args) or args.qwen_clone_x_vector_only
    if args.qwen_clone_enabled:
        if args.qwen_engine != "vllm-omni":
            parser.error("--qwen-clone-* requires --qwen-engine vllm-omni")
        if not args.qwen_clone_ref_audio_user or not args.qwen_clone_ref_audio_moshi:
            parser.error(
                "clone mode requires both --qwen-clone-ref-audio-user and "
                "--qwen-clone-ref-audio-moshi"
            )
        if "Base" not in args.model:
            parser.error(
                "clone mode requires a Base model, e.g. "
                "--model Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            )
        if not args.qwen_clone_x_vector_only:
            if not args.qwen_clone_ref_text_user or not args.qwen_clone_ref_text_moshi:
                parser.error(
                    "in-context clone requires --qwen-clone-ref-text-user/-moshi "
                    "(or use --qwen-clone-x-vector-only)"
                )
        for path in (args.qwen_clone_ref_audio_user, args.qwen_clone_ref_audio_moshi):
            if not Path(path).is_file():
                parser.error(f"clone reference WAV not found: {path}")
        if args.user_speaker_pool.strip():
            parser.error(
                "--user-speaker-pool cannot be combined with clone mode "
                "(preset speaker overrides have no clone reference)"
            )

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
    elif args.tts_backend == "kokoro":
        pool = [s.strip() for s in args.kokoro_user_voices.split(",") if s.strip()]
        invalid = [s for s in pool if not s.startswith(("jf_", "jm_"))]
        if not pool or invalid:
            parser.error(
                f"--kokoro-user-voices に無効な話者: {invalid or '(空)'}。"
                "日本語話者ID (jf_*/jm_*) をカンマ区切りで指定してください。"
            )
        args.user_speaker_pool_list = pool
    else:
        args.user_speaker_pool_list = [args.speaker_user]

    if args.style_preset is not None:
        if (
            args.style_preset not in {"none", "random", "auto"}
            and args.style_preset not in WHOLE_UTTERANCE_STYLE_PRESETS
        ):
            parser.error(
                f"--style-preset={args.style_preset!r} は無効です。"
                f"候補: {sorted(WHOLE_UTTERANCE_STYLE_PRESETS.keys())}, none, random, auto"
            )
        if not args.whole_utterance:
            parser.error("--style-preset は --whole-utterance と一緒に使ってください")

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


def prepare_dialogue_render_job(
    index: int,
    template: dict[str, Any],
    args: argparse.Namespace,
    emotion_map: dict[str, str],
    total_dialogues: int,
) -> DialogueRenderJob:
    turns: list[DialogueTurn] = []
    for raw_turn in template["turns"]:
        speaker = raw_turn["speaker"]
        if speaker == "silence":
            turns.append(
                DialogueTurn(
                    speaker="silence",
                    text="",
                    duration_sec=float(raw_turn.get("duration_sec", 2.0)),
                    note=raw_turn.get("note"),
                )
            )
            continue
        emotion = None if args.no_emotion else raw_turn.get("emotion")
        turns.append(
            DialogueTurn(
                speaker=speaker,
                text=raw_turn["text"],
                emotion=emotion,
                instruct=resolve_emotion(emotion, emotion_map),
                timing=str(raw_turn.get("timing") or "sequential"),
                start_after_previous_start_sec=optional_float(
                    raw_turn.get("start_after_previous_start_sec")
                ),
                truncate_previous_after_sec=optional_float(
                    raw_turn.get("truncate_previous_after_sec")
                ),
                gain=max(0.0, float(optional_float(raw_turn.get("gain"), 1.0) or 0.0)),
                voice_role=str(raw_turn.get("voice_role") or "") or None,
                event=str(raw_turn.get("event") or "") or None,
            )
        )

    if not args.no_opening_greeting and args.opening_greeting:
        turns = [
            DialogueTurn(
                speaker="moshi",
                text=args.opening_greeting,
                timing="sequential",
                event="opening_greeting",
            ),
            *turns,
        ]

    dialogue = Dialogue(
        id=safe_stem(template["id"], f"dialogue_{index:03d}"),
        category=template["category"],
        risk_level=template["risk_level"],
        title=template["title"],
        turns=turns,
        duplex_task=str(template.get("duplex_task") or "") or None,
    )
    user_voice = args.user_speaker_pool_list[
        (index - 1) % len(args.user_speaker_pool_list)
    ]
    user_override = "user" if args.tts_backend == "moss-ttsd" else user_voice
    other_override = (
        "other" if args.tts_backend in ("moss-ttsd", "kokoro") else args.speaker_other
    )
    background_override = (
        "background"
        if args.tts_backend in ("moss-ttsd", "kokoro")
        else args.speaker_background
    )
    log_progress = (
        args.log_every <= 1
        or index == 1
        or index == total_dialogues
        or index % args.log_every == 0
    )

    instruct_user = args.instruct_user
    instruct_moshi = args.instruct_moshi
    resolved_preset_key: str | None = None
    if args.whole_utterance and args.style_preset not in {None, "none"}:
        preset_key = args.style_preset
        if preset_key == "random":
            preset_key = random.choice(list(WHOLE_UTTERANCE_STYLE_PRESETS))
        elif preset_key == "auto":
            preset_key = resolve_auto_style_preset(
                dialogue.id, template.get("emotional_state")
            )
        resolved_preset_key = preset_key
        preset = WHOLE_UTTERANCE_STYLE_PRESETS[preset_key]
        instruct_user = preset["user"]
        instruct_moshi = preset["moshi"]

    return DialogueRenderJob(
        index=index,
        template=template,
        dialogue=dialogue,
        stem=f"sample_{index:03d}_{dialogue.id}",
        user_voice=user_voice,
        user_override=user_override,
        other_override=other_override,
        background_override=background_override,
        instruct_user=instruct_user,
        instruct_moshi=instruct_moshi,
        resolved_preset_key=resolved_preset_key,
        log_progress=log_progress,
    )


def iter_dialogue_render_jobs(
    templates: list[dict[str, Any]],
    args: argparse.Namespace,
    emotion_map: dict[str, str],
    tts: Any,
    greeting_pcm: np.ndarray | None,
    done_stems: set[str],
):
    """Yield jobs with either the live backend or per-dialogue batched replay."""
    pending: list[DialogueRenderJob] = []

    def flush():
        if not pending:
            return []
        if args.qwen_engine != "vllm-omni":
            return [(job, tts) for job in pending]

        request_groups = [
            plan_whole_utterance_synthesis(
                job.dialogue,
                user_speaker_override=job.user_override,
                other_speaker_override=job.other_override,
                background_speaker_override=job.background_override,
                instruct_user=job.instruct_user,
                instruct_moshi=job.instruct_moshi,
                max_chars_per_synthesis=args.whole_utterance_max_chars,
                opening_greeting_pcm=greeting_pcm,
            )
            for job in pending
        ]
        flat_requests = [request for group in request_groups for request in group]
        logger.info(
            "vLLM-Omni planning batch: dialogues=%d requests=%d batch_size=%d",
            len(pending),
            len(flat_requests),
            args.tts_batch_size,
        )
        flat_audio = tts.synthesize_many(flat_requests)
        rendered: list[tuple[DialogueRenderJob, ReplayTTS]] = []
        offset = 0
        for job, requests in zip(pending, request_groups):
            count = len(requests)
            replay = ReplayTTS(
                requests,
                flat_audio[offset : offset + count],
                sample_rate=tts.sample_rate,
            )
            rendered.append((job, replay))
            offset += count
        if offset != len(flat_audio):
            raise RuntimeError("batched TTS output count did not match the synthesis plan")
        return rendered

    for index, template in enumerate(templates, start=1):
        stem = f"sample_{index:03d}_{safe_stem(template['id'], f'dialogue_{index:03d}')}"
        if args.resume and stem in done_stems:
            continue
        pending.append(
            prepare_dialogue_render_job(
                index, template, args, emotion_map, len(templates)
            )
        )
        if len(pending) >= (
            args.dialogue_batch_size if args.qwen_engine == "vllm-omni" else 1
        ):
            yield from flush()
            pending.clear()
    if pending:
        yield from flush()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.out_dir / "data_stereo"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / args.manifest_name
    dialogues_path = args.out_dir / "dialogues.jsonl"
    done_stems: set[str] = set()
    if args.resume:
        resumed = rebuild_from_completed(
            data_dir, manifest_path, dialogues_path, done_stems
        )
        logger.info(
            "--resume: 完了済みサンプル %d 件を検出。manifest を再構築して"
            "未完了分のみ合成します。", resumed,
        )
    else:
        for p in (manifest_path, dialogues_path):
            if p.exists():
                p.unlink()

    emotion_map = load_emotion_map(args.emotion_map_file)
    if args.no_emotion:
        logger.info("--no-emotion: ターン側 emotion ラベルを無視します")

    if args.dialogues_jsonl is not None:
        all_templates = load_dialogues_from_jsonl(args.dialogues_jsonl)
        if args.auto_overlap_aizuchi:
            all_templates = [
                {**template, "turns": apply_aizuchi_overlap(template["turns"])}
                for template in all_templates
            ]
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
    tts: Any
    if args.tts_backend == "moss-ttsd":
        ref_audio_paths, ref_texts = load_moss_references(args)
        tts = MossTTSD(
            model_name=args.moss_model,
            codec_model_name=args.moss_codec_model,
            ref_audio_paths=ref_audio_paths,
            ref_texts=ref_texts,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.tts_backend == "kokoro":
        # 遅延 import: tts_comparison_backends は本モジュールを import している
        # ため、トップレベルで import すると循環になる。
        try:
            from scripts.tts_comparison_backends import KokoroTTS
        except ImportError:
            from tts_comparison_backends import KokoroTTS
        tts = KokoroTTS(
            device=args.device,
            voice_moshi=args.kokoro_voice_moshi,
            voice_user=args.user_speaker_pool_list[0],
            voice_other=args.kokoro_voice_other,
            voice_background=args.kokoro_voice_background,
            model_id=args.kokoro_model,
        )
    elif args.qwen_engine == "vllm-omni":
        clone_refs = None
        if args.qwen_clone_enabled:
            clone_refs = {
                "user": CloneReference(
                    ref_audio=args.qwen_clone_ref_audio_user,
                    ref_text=args.qwen_clone_ref_text_user,
                ),
                "moshi": CloneReference(
                    ref_audio=args.qwen_clone_ref_audio_moshi,
                    ref_text=args.qwen_clone_ref_text_moshi,
                ),
            }
        tts = VLLMQwen3TTS(
            model_id=args.model,
            dtype_str=args.dtype,
            speaker_user=args.speaker_user,
            speaker_moshi=args.speaker_moshi,
            language=args.language,
            instruct_user=args.instruct_user,
            instruct_moshi=args.instruct_moshi,
            batch_size=args.tts_batch_size,
            stage_configs_path=args.vllm_stage_config,
            max_new_tokens=args.vllm_max_new_tokens,
            clone_refs=clone_refs,
            clone_x_vector_only=args.qwen_clone_x_vector_only,
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

    # 固定挨拶は run の最初に1回だけ合成（またはキャッシュから読み込み）、
    # 全対話で同じ音声を使い回す。声色は --speaker-moshi と
    # --opening-greeting-instruct で固定される。
    greeting_pcm: np.ndarray | None = None
    if not args.no_opening_greeting and args.opening_greeting:
        greeting_cache_dir = (
            args.opening_greeting_cache_dir
            if str(args.opening_greeting_cache_dir) not in ("", ".")
            else None
        )
        greeting_pcm = synthesize_opening_greeting(
            tts,
            args.opening_greeting,
            args.opening_greeting_instruct,
            greeting_cache_dir,
            backend=args.tts_backend,
            model_id=(
                args.moss_model if args.tts_backend == "moss-ttsd"
                else args.kokoro_model if args.tts_backend == "kokoro"
                else args.model
            ),
            speaker=(
                "moshi" if args.tts_backend == "moss-ttsd"
                else args.kokoro_voice_moshi if args.tts_backend == "kokoro"
                else args.speaker_moshi
            ),
        )

    fa_aligner: ForcedAligner | None = None
    if args.whole_utterance:
        fa_aligner = ForcedAligner(
            device=args.device, fallback_mode=args.fa_fallback,
        )
        fa_aligner.load()

    completed_stems = {
        f"sample_{idx:03d}_{safe_stem(tmpl['id'], f'dialogue_{idx:03d}')}"
        for idx, tmpl in enumerate(templates, start=1)
    } & done_stems
    success_count = len(completed_stems) if args.resume else 0
    failed_count = 0
    fa_failed_count = 0
    if success_count:
        logger.info("resume: %d/%d dialogue(s) already complete", success_count, len(templates))

    if args.success_target is not None and success_count >= args.success_target:
        render_jobs = iter(())
        logger.info(
            "success-target %d is already satisfied; no TTS batches will run",
            args.success_target,
        )
    else:
        render_jobs = iter_dialogue_render_jobs(
            templates, args, emotion_map, tts, greeting_pcm, done_stems
        )

    for job, render_tts in render_jobs:
        idx = job.index
        tmpl = job.template
        dialogue = job.dialogue
        stem = job.stem
        user_voice = job.user_voice
        user_override = job.user_override
        other_override = job.other_override
        background_override = job.background_override
        log_progress = job.log_progress
        resolved_preset_key = job.resolved_preset_key
        try:
            if args.success_target is not None and success_count >= args.success_target:
                logger.info(
                    "success-target %d に到達（完了済み含む）したため打ち切ります",
                    args.success_target,
                )
                break
            if log_progress:
                logger.info(
                    "[%d/%d] 対話 %s を合成中 (moshi=%s, user=%s) ...",
                    idx, len(templates), dialogue.id, args.speaker_moshi, user_voice,
                )
                if resolved_preset_key is not None:
                    logger.info("style-preset: %s", resolved_preset_key)
            t0 = time.time()
            if args.whole_utterance and fa_aligner is not None:
                segments, silences = build_segments_whole_utterance(
                    dialogue, render_tts, fa_aligner, args.lead_in_sec, args.gap_sec,
                    user_speaker_override=user_override,
                    other_speaker_override=other_override,
                    background_speaker_override=background_override,
                    instruct_user=job.instruct_user,
                    instruct_moshi=job.instruct_moshi,
                    max_chars_per_synthesis=args.whole_utterance_max_chars,
                    opening_greeting_pcm=greeting_pcm,
                )
                if isinstance(render_tts, ReplayTTS):
                    render_tts.assert_consumed()
            else:
                segments, silences = build_segments(
                    dialogue, render_tts, args.lead_in_sec, args.gap_sec,
                    user_speaker_override=user_override,
                    other_speaker_override=other_override,
                    background_speaker_override=background_override,
                    opening_greeting_pcm=greeting_pcm,
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

            # stem was computed above for the resume check; reuse it so the
            # written filename matches the done_stems bookkeeping exactly.
            wav_path = data_dir / f"{stem}.wav"
            json_path = wav_path.with_suffix(".json")
            duration = stereo.shape[-1] / tts.sample_rate

            write_wav(wav_path, stereo, tts.sample_rate)

            # kyutai moshi-finetune の Interleaver は「1 エントリ = 1 単語」を
            # 前提にエントリ開始フレームへトークンを詰めるため、発話単位の
            # まま渡すとテキストストリームのペーシングが崩壊する。単語単位に
            # 分割したものを alignments として書き、発話単位の原本は
            # alignments_utterance に残す。
            utterance_alignments = [
                [seg.text, [round(seg.start_sec, 4), round(seg.end_sec, 4)], seg.label]
                for seg in segments
            ]
            alignments, _split_stats = split_utterance_alignments(utterance_alignments)
            write_json(json_path, {
                "alignments": alignments,
                "alignments_utterance": utterance_alignments,
                "metadata": {
                    "alignments_granularity": "word",
                    "alignments_word_split": {
                        "version": 1,
                        "method": "char-proportional",
                        "segmenter": get_segmenter_name(),
                    },
                    "mode": (
                        ("whole-utterance-" + args.tts_backend)
                        if args.whole_utterance
                        else (
                            "moss-ttsd-scripted"
                            if args.tts_backend == "moss-ttsd"
                            else "qwen3-tts-scripted"
                        )
                    ),
                    "sample_rate": tts.sample_rate,
                    "duration_sec": round(duration, 4),
                    "tts_backend": args.tts_backend,
                    "qwen_engine": (
                        args.qwen_engine if args.tts_backend == "qwen3" else None
                    ),
                    "tts_batch_size": (
                        args.tts_batch_size
                        if args.qwen_engine == "vllm-omni"
                        else 1
                    ),
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
                    "qwen_clone": (
                        {
                            "x_vector_only": args.qwen_clone_x_vector_only,
                            "ref_audio_user": args.qwen_clone_ref_audio_user,
                            "ref_text_user": args.qwen_clone_ref_text_user,
                            "ref_audio_moshi": args.qwen_clone_ref_audio_moshi,
                            "ref_text_moshi": args.qwen_clone_ref_text_moshi,
                        }
                        if getattr(args, "qwen_clone_enabled", False)
                        else None
                    ),
                    "left_channel": "moshi",
                    "right_channel": "user",
                    "wall_time_sec": round(elapsed, 3),
                    "emotion_control": "off" if args.no_emotion else "on",
                    "style_preset": getattr(args, "style_preset", None),
                    "style_preset_resolved": resolved_preset_key,
                    "opening_greeting": (
                        {
                            "text": args.opening_greeting,
                            "instruct": args.opening_greeting_instruct,
                            "reused_fixed_audio": greeting_pcm is not None,
                        }
                        if not args.no_opening_greeting and args.opening_greeting
                        else None
                    ),
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

            if log_progress:
                logger.info("保存完了: %s (%.2f 秒, wall %.1f 秒)", wav_path, duration, elapsed)
            success_count += 1
            if args.success_target is not None and success_count >= args.success_target:
                logger.info(
                    "success-target %d に到達したため打ち切ります (処理済み %d/%d, 失敗 %d 件)",
                    args.success_target, idx, len(templates), failed_count,
                )
                break
        except ForcedAlignmentError as exc:
            failed_count += 1
            fa_failed_count += 1
            logger.warning(
                "[%d/%d] 対話 %s: Forced Alignment 失敗のため破棄します"
                "（境界が狂った音声を学習データに混ぜない）: %s",
                idx, len(templates), tmpl.get("id", "<unknown>"), exc,
            )
            continue
        except Exception as exc:
            failed_count += 1
            logger.warning(
                "[%d/%d] 対話 %s の合成に失敗したためスキップします: %s",
                idx, len(templates), tmpl.get("id", "<unknown>"), exc,
                exc_info=True,
            )
            continue

    logger.info(
        "対話合成完了: 成功 %d 件, 失敗 %d 件 (うち FA 失敗による破棄 %d 件)",
        success_count, failed_count, fa_failed_count,
    )
    performance_stats = getattr(tts, "performance_stats", None)
    if callable(performance_stats):
        perf = performance_stats()
        logger.info(
            "vLLM-Omni performance summary: batches=%d requests=%d "
            "audio_sec=%.2f inference_wall_sec=%.2f audio_x=%.3f rtf=%.3f",
            perf["batches"],
            perf["requests"],
            perf["audio_sec"],
            perf["inference_wall_sec"],
            perf["audio_x"],
            perf["rtf"],
        )
    close_tts = getattr(tts, "close", None)
    if callable(close_tts):
        close_tts()
    if args.success_target is not None and success_count < args.success_target:
        raise SystemExit(
            f"success-target {args.success_target} に届きませんでした "
            f"(成功 {success_count} 件, 失敗 {failed_count} 件)。"
            "予備の対話を追加するか success-target を見直してください。"
        )
    logger.info("完了。manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
