#!/usr/bin/env python3
"""発話（ターン）単位の Moshi alignments を単語単位に分割する共通ヘルパー。

背景:
    kyutai moshi-finetune の Interleaver は「1 エントリ = 1 単語」を前提に、
    エントリ開始フレームから 1 フレーム 1 トークンでテキストを書き込む。
    ターン全文を 1 エントリで渡すと、(1) ターン冒頭に全文トークンが
    バースト配置されて音声とテキストの同期が崩れ、(2) 次のエントリ開始時に
    書き残しトークンが無警告で破棄される。

    このモジュールはターン文を単語（pyopenjtalk 形態素、無ければ正規表現
    チャンク）に分割し、ターンの [start, end] を文字数比例で各単語に配分する。
    nu-dialogue 版 tokenize_text.py の文字数等分と同じ近似で、音声の
    再合成なしに教師信号のペーシングを修復できる。

sidecar JSON の alignments 形式:
    [text, [start_sec, end_sec], "SPEAKER_MAIN" | "SPEAKER_USER"]
"""

from __future__ import annotations

import re
from typing import Any

# 単語 1 エントリの上限文字数。日本語の発話速度 ~7-8 文字/秒、Mimi 12.5Hz
# (80ms/frame)、1 トークン ~1-2 文字なので、8 文字 ≒ 1 秒分。これを超える
# エントリはバーストが目立つため均等分割する。
DEFAULT_MAX_WORD_CHARS = 8

# 正規表現フォールバック時のチャンク長。形態素境界が取れないぶん短めに切る。
FALLBACK_CHUNK_CHARS = 4

_PUNCT_CHARS = "、。！？：；…‥・「」『』（）［］〔〕〈〉《》,.!?:;()[]{}\"'　 \t"
_PUNCT_ONLY = re.compile(rf"^[{re.escape(_PUNCT_CHARS)}]+$")

_pyopenjtalk: Any = None
_pyopenjtalk_failed = False


def _try_import_pyopenjtalk() -> Any:
    global _pyopenjtalk, _pyopenjtalk_failed
    if _pyopenjtalk is not None or _pyopenjtalk_failed:
        return _pyopenjtalk
    try:
        import pyopenjtalk  # type: ignore

        _pyopenjtalk = pyopenjtalk
    except Exception:
        _pyopenjtalk_failed = True
    return _pyopenjtalk


def _segment_pyopenjtalk(text: str) -> list[str] | None:
    """pyopenjtalk の NJD 形態素で分割する。表層列を完全復元できなければ None。"""
    pyopenjtalk = _try_import_pyopenjtalk()
    if pyopenjtalk is None:
        return None
    try:
        features = pyopenjtalk.run_frontend(text)
    except Exception:
        return None
    words: list[str] = []
    for feat in features:
        if isinstance(feat, dict):
            surface = str(feat.get("string", ""))
        else:  # 旧 API (str の NJD feature 行)
            surface = str(feat).split(",", 1)[0]
        if surface:
            words.append(surface)
    # タイムスタンプを文字数比例で配るため、分割結果が原文を無劣化で
    # 復元できること（正規化・脱落なし）を必須条件にする。
    if "".join(words) != text:
        return None
    return words


_segmenter_name: str | None = None


def get_segmenter_name() -> str:
    """実際に使われる単語分割器の名前を返す（"pyopenjtalk" | "regex-chunk"）。

    日本語の代表文でプローブする（結果はプロセス内でキャッシュ）。
    pyopenjtalk が import できても表層復元に失敗する環境では regex-chunk を返す。
    """
    global _segmenter_name
    if _segmenter_name is None:
        probe = "もしもし、こちら相談窓口になります。"
        _segmenter_name = (
            "pyopenjtalk" if _segment_pyopenjtalk(probe) is not None else "regex-chunk"
        )
    return _segmenter_name


def _segment_fallback(text: str, chunk_chars: int = FALLBACK_CHUNK_CHARS) -> list[str]:
    """句読点境界 + 固定長チャンクによる無劣化分割。"""
    runs = re.findall(rf"[{re.escape(_PUNCT_CHARS)}]+|[^{re.escape(_PUNCT_CHARS)}]+", text)
    words: list[str] = []
    for run in runs:
        if _PUNCT_ONLY.match(run):
            words.append(run)
            continue
        for i in range(0, len(run), chunk_chars):
            words.append(run[i : i + chunk_chars])
    return words


def _merge_punct_into_previous(words: list[str]) -> list[str]:
    """句読点のみのトークンを直前の単語に併合する（先頭なら直後へ）。"""
    merged: list[str] = []
    pending_head = ""
    for word in words:
        if _PUNCT_ONLY.match(word):
            if merged:
                merged[-1] += word
            else:
                pending_head += word
        else:
            merged.append(pending_head + word)
            pending_head = ""
    if pending_head:
        if merged:
            merged[0] = pending_head + merged[0]
        else:
            merged.append(pending_head)
    return merged


def _enforce_max_chars(words: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    for word in words:
        if len(word) <= max_chars:
            out.append(word)
            continue
        n_pieces = -(-len(word) // max_chars)  # ceil
        size = -(-len(word) // n_pieces)
        out.extend(word[i : i + size] for i in range(0, len(word), size))
    return out


def segment_text(text: str, max_word_chars: int = DEFAULT_MAX_WORD_CHARS) -> list[str]:
    """テキストを単語列に分割する。常に ``"".join(結果) == text`` を保証する。"""
    if not text:
        return []
    words = _segment_pyopenjtalk(text)
    if words is None:
        words = _segment_fallback(text)
    words = _merge_punct_into_previous(words)
    words = _enforce_max_chars(words, max_word_chars)
    assert "".join(words) == text, "segmentation must be lossless"
    return words


def distribute_word_times(
    words: list[str], start: float, end: float
) -> list[tuple[str, float, float]]:
    """[start, end] を文字数比例で各単語に配分する。"""
    total_chars = sum(len(w) for w in words)
    if total_chars <= 0:
        return []
    duration = end - start
    out: list[tuple[str, float, float]] = []
    cum = 0
    for word in words:
        word_start = start + duration * cum / total_chars
        cum += len(word)
        word_end = start + duration * cum / total_chars
        out.append((word, round(word_start, 4), round(word_end, 4)))
    return out


def split_utterance_alignments(
    alignments: list[Any],
    max_word_chars: int = DEFAULT_MAX_WORD_CHARS,
) -> tuple[list[list[Any]], dict[str, int]]:
    """発話単位 alignments リストを単語単位に展開する。

    不正な形のエントリはそのまま維持する（統計 ``passthrough`` に計上）。
    Returns:
        (word_level_alignments, stats)
    """
    out: list[list[Any]] = []
    stats = {"utterances": 0, "words": 0, "passthrough": 0}
    for entry in alignments:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) < 3
            or not isinstance(entry[0], str)
            or not entry[0]
            or not isinstance(entry[1], (list, tuple))
            or len(entry[1]) < 2
        ):
            out.append(entry)
            stats["passthrough"] += 1
            continue
        text, span, label = entry[0], entry[1], entry[2]
        try:
            start, end = float(span[0]), float(span[1])
        except (TypeError, ValueError):
            out.append(entry)
            stats["passthrough"] += 1
            continue
        if end <= start:
            out.append(entry)
            stats["passthrough"] += 1
            continue
        words = segment_text(text, max_word_chars=max_word_chars)
        timed = distribute_word_times(words, start, end)
        if not timed:
            out.append(entry)
            stats["passthrough"] += 1
            continue
        stats["utterances"] += 1
        stats["words"] += len(timed)
        out.extend([word, [word_start, word_end], label] for word, word_start, word_end in timed)
    return out, stats
