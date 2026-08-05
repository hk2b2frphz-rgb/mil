#!/usr/bin/env python3
"""
人手アノテーション TSV のタイムスタンプを音声に合わせ直す。

前提: **書き起こし・話者ラベル・発話の順番は正しく、時刻だけが合っていない**。
その条件なら時刻は音声から復元できる。テキストが正、時刻はその出力になる。

  1. faster-whisper で全体を書き起こす(単語タイムスタンプ付き)。
  2. 人手 TSV と ASR の**読み**(pyopenjtalk で katakana 化)を話者ごとに
     文字列アライメントし、一意な k-gram 一致を anchor にして概算時刻を出す。
     話者ごとに分けるのは、重畳があっても同一話者内の発話順は単調だから。
  3. 概算位置の前後だけを切り出し、その発話のテキストだけで MMS_FA の CTC
     強制アライメントを掛けて start/end を実測する。
  4. silero VAD の音声区間端へ ±--snap-sec 以内で吸着させる。

応答レイテンシは「User の end -> Staff の start」で測るので、評価の精度は
そのまま境界の精度になる。3 と 4 はそのための追い込み。

**元の TSV は書き換えない。** 出力を確認してから自分で差し替える。

出力(入力 WAV ごとに <out-dir>/<wav_stem>/ 配下):
  - labels/        Audacity の File > Import > Labels で読めるラベル
                   (all / 話者ごと / flagged / original)
  - aligned.tsv    合わせ直した TSV(Speaker/Transcription/Start/End、そのまま
                   アノテーション TSV として使える形式)
  - report.tsv     1 行ごとの新旧時刻・ずれ・確信度・手法・フラグ(要確認の印)
  - summary.json   ずれの統計と診断(定数オフセットか、線形ドリフトか、乱れか)
  - asr.json       ASR の単語タイムスタンプ(再実行時のキャッシュ)
  - review_wav/    確信度の低いものを中心に、新しい境界で切った確認用 WAV

使い方:
    uv run python scripts/realign_annotation_tsv.py /path/dialogues/

    # ずれの性質だけ先に見る(ASR まで。強制アライメントも WAV 出力もしない)
    uv run python scripts/realign_annotation_tsv.py /path/dialogues/ --check-only

Audacity での確認:
    元の WAV を開き、File > Import > Labels で labels/all.txt を読む。
    original.txt も一緒に読めば、旧ラベルと新ラベルが別トラックで並ぶ。

GPU 推奨(faster-whisper large-v3 と MMS_FA)。scripts/run_realign_annotation.pbs
から投げられる。
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "eval"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

try:
    from scripts.build_test_wav_from_tsv import (  # noqa: E402
        _FNAME_RE,
        _STRIP_RE,
        Utterance,
        annotation_for,
        collect_inputs,
        read_annotation,
        require_annotations,
        write_tsv,
    )
except ImportError:  # scripts/ を直接 sys.path に持つ場合(直接実行・環境差)
    from build_test_wav_from_tsv import (  # type: ignore[no-redef]  # noqa: E402
        _FNAME_RE,
        _STRIP_RE,
        Utterance,
        annotation_for,
        collect_inputs,
        read_annotation,
        require_annotations,
        write_tsv,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("data/test_data/realign")

# 読みの正規化。表記ゆれと ASR の癖を吸収するため、長音・促音は落とし、
# 揺れやすい仮名は代表形へ寄せる。「そうですね」の ソー/ソウ のような差で
# anchor を落とさないための処置。
_KANA_RE = re.compile(r"[ァ-ヴ]")
_DROP_KANA = set("ーッ")
_FOLD_KANA = {"ヲ": "オ", "ヅ": "ズ", "ヂ": "ジ", "ヴ": "ブ", "ヰ": "イ", "ヱ": "エ"}

# Audacity のラベルはタブ区切りの 1 行 1 ラベル。テキストにタブ・改行は入らない。
_LABEL_SANITIZE = re.compile(r"[\t\r\n]+")

# 1 文字あたりのおおよその発話時間。長さの妥当性チェックにだけ使う。
_SEC_PER_KANA = 0.12


@dataclass
class Line:
    """アノテーション 1 行と、その再アライメント結果。"""

    index: int          # TSV 上の並び順(0 始まり)
    utt: Utterance
    reading: str
    est_start: float | None = None   # anchor 補間による概算
    est_end: float | None = None
    new_start: float = 0.0
    new_end: float = 0.0
    coverage: float = 0.0            # anchor が張れた文字の割合
    method: str = "none"             # anchor / fa / fa+vad / original
    flags: tuple[str, ...] = ()

    @property
    def confident(self) -> bool:
        return not self.flags and self.coverage > 0.0


# ---------------------------------------------------------------------------
# 読みの正規化
# ---------------------------------------------------------------------------

_reading_cache: dict[str, str] = {}
_g2p = None
_g2p_kind = ""


def _load_g2p() -> None:
    global _g2p, _g2p_kind
    if _g2p_kind:
        return
    try:
        import pyopenjtalk

        _g2p = pyopenjtalk
        _g2p_kind = "pyopenjtalk"
    except ImportError:
        _g2p = None
        _g2p_kind = "raw"
        logger.warning(
            "pyopenjtalk がありません。漢字の読みを取れないため anchor が減ります。"
        )
    logger.info("読み変換: %s", _g2p_kind)


def to_reading(text: str) -> str:
    """テキストを照合用のカタカナ列にする。記号・空白・長音・促音は落とす。

    読みを介すのがこの処理の要。ASR が「こんにちは」、人手 TSV が「今日は」と
    書いていても、どちらも コンニチワ になるので anchor が張れる。
    """
    if text in _reading_cache:
        return _reading_cache[text]
    _load_g2p()
    kana = text
    if _g2p is not None:
        try:
            kana = _g2p.g2p(text, kana=True)
        except Exception:  # 記号だけの行などで落ちることがある
            kana = text
    else:
        # ひらがなをカタカナへ寄せるだけの退避策。漢字は照合できない。
        kana = "".join(
            chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text
        )
    out = []
    for ch in kana:
        if ch in _DROP_KANA or not _KANA_RE.match(ch):
            continue
        out.append(_FOLD_KANA.get(ch, ch))
    result = "".join(out)
    _reading_cache[text] = result
    return result


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

class WhisperHolder:
    """モデルは 1 回だけ読む(対話ごとに読み直さない)。"""

    def __init__(self, model_name: str, device: str, compute_type: str):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def get(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "faster-whisper を読み込みます: %s (%s/%s)",
                self.model_name, self.device, self.compute_type,
            )
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model


def run_asr(
    wav_path: Path, cache_path: Path, holder: WhisperHolder, refresh: bool
) -> list[dict]:
    if cache_path.is_file() and not refresh:
        words = json.loads(cache_path.read_text(encoding="utf-8"))["words"]
        logger.info("ASR キャッシュを再利用: %s (%d 語)", cache_path, len(words))
        return words

    model = holder.get()
    # condition_on_previous_text=False: 前文脈に引きずられた作話を抑える。
    # 時刻の基準にする以上、聞こえていない語を足されるのが一番困る。
    segments, info = model.transcribe(
        str(wav_path),
        language="ja",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    words: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            words.append({
                "text": word.word,
                "start": round(float(word.start), 4),
                "end": round(float(word.end), 4),
            })
    logger.info("ASR: %d 語 (音声 %.1f 秒)", len(words), info.duration)
    cache_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8"
    )
    return words


def asr_char_stream(words: list[dict]) -> tuple[str, list[float]]:
    """ASR の単語列を「読み 1 文字 + その時刻」の列にする。

    単語内は等分。単語より細かい精度はここでは要らない(概算 anchor 用で、
    実測は後段の強制アライメントが出す)。
    """
    chars: list[str] = []
    times: list[float] = []
    for word in words:
        reading = to_reading(word["text"])
        if not reading:
            continue
        start = float(word["start"])
        duration = max(float(word["end"]) - start, 1e-3)
        for i, ch in enumerate(reading):
            chars.append(ch)
            times.append(start + duration * (i + 0.5) / len(reading))
    return "".join(chars), times


# ---------------------------------------------------------------------------
# anchor による粗い対応付け
# ---------------------------------------------------------------------------

def unique_kgrams(text: str, k: int) -> dict[str, int]:
    """その文字列の中で 1 回しか出てこない k-gram -> 位置。

    複数回出るものは対応先を決められないので捨てる。相槌のような頻出短句を
    anchor にしないための処置でもある。
    """
    seen: dict[str, int] = {}
    duplicated: set[str] = set()
    for i in range(len(text) - k + 1):
        gram = text[i:i + k]
        if gram in seen:
            duplicated.add(gram)
        else:
            seen[gram] = i
    for gram in duplicated:
        seen.pop(gram, None)
    return seen


def longest_increasing_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(gt位置, asr位置) の中から、両方が単調増加する最大の部分列を返す。

    同一話者の発話は必ず時間順に並ぶので、順序が入れ替わる対応は誤りとして
    ここで落ちる。patience sorting で O(n log n)。
    """
    if not pairs:
        return []
    pairs = sorted(pairs)
    tails: list[int] = []          # 長さ i+1 の増加列の末尾 asr 位置
    tail_index: list[int] = []     # その末尾の pairs 上の位置
    previous = [-1] * len(pairs)
    for i, (_, asr_pos) in enumerate(pairs):
        j = bisect.bisect_left(tails, asr_pos)
        if j > 0:
            previous[i] = tail_index[j - 1]
        if j == len(tails):
            tails.append(asr_pos)
            tail_index.append(i)
        else:
            tails[j] = asr_pos
            tail_index[j] = i
    result: list[tuple[int, int]] = []
    cursor = tail_index[-1]
    while cursor >= 0:
        result.append(pairs[cursor])
        cursor = previous[cursor]
    result.reverse()
    return result


class Interpolator:
    """anchor 間を線形補間して、GT 文字位置 -> 時刻 を返す。

    anchor の外側は全体の平均レート(秒/文字)で外挿する。無音の分まで文字数に
    比例して配分してしまうので、これはあくまで概算。実測は強制アライメント。
    """

    def __init__(self, anchors: list[tuple[int, float]], fallback_rate: float):
        self.xs = [a[0] for a in anchors]
        self.ys = [a[1] for a in anchors]
        self.rate = fallback_rate

    def __call__(self, x: float) -> float:
        if not self.xs:
            return x * self.rate
        if x <= self.xs[0]:
            return self.ys[0] - (self.xs[0] - x) * self.rate
        if x >= self.xs[-1]:
            return self.ys[-1] + (x - self.xs[-1]) * self.rate
        i = bisect.bisect_right(self.xs, x) - 1
        x0, x1 = self.xs[i], self.xs[i + 1]
        y0, y1 = self.ys[i], self.ys[i + 1]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def coarse_align_speaker(
    lines: list[Line], asr_text: str, asr_times: list[float], k: int
) -> int:
    """1 話者分の発話へ概算時刻と coverage を書き込む。返り値は anchor 数。"""
    gt_chars: list[str] = []
    span: list[tuple[int, int]] = []
    for line in lines:
        start = len(gt_chars)
        gt_chars.extend(line.reading)
        span.append((start, len(gt_chars)))
    gt_text = "".join(gt_chars)
    if not gt_text or not asr_text:
        return 0

    gt_positions = unique_kgrams(gt_text, k)
    asr_positions = unique_kgrams(asr_text, k)
    pairs = [
        (gt_i, asr_positions[gram])
        for gram, gt_i in gt_positions.items()
        if gram in asr_positions
    ]
    monotone = longest_increasing_pairs(pairs)

    total_asr_sec = (asr_times[-1] - asr_times[0]) if len(asr_times) > 1 else 0.0
    fallback_rate = (
        total_asr_sec / max(len(asr_text), 1) if total_asr_sec > 0 else _SEC_PER_KANA
    )
    fallback_rate = min(max(fallback_rate, 0.04), 0.40)

    anchors = [(gt_i, asr_times[asr_i]) for gt_i, asr_i in monotone]
    interp = Interpolator(anchors, fallback_rate)

    covered = np.zeros(len(gt_text), dtype=bool)
    for gt_i, _ in monotone:
        covered[gt_i:gt_i + k] = True

    for i, line in enumerate(lines):
        lo, hi = span[i]
        if hi <= lo:
            line.est_start = None
            line.est_end = None
            line.coverage = 0.0
            continue
        line.est_start = interp(lo)
        line.est_end = interp(hi)
        line.coverage = float(covered[lo:hi].mean())
    return len(monotone)


# ---------------------------------------------------------------------------
# 強制アライメントによる実測
# ---------------------------------------------------------------------------

class Refiner:
    """MMS_FA を発話 1 つ分の窓に掛けて start/end を実測する。

    リポジトリ内の ForcedAligner をそのまま使う(TTS 側と同じ実装)。窓の外側は
    使わないので、対話全体を一度に流すときの CTC 長制限にも当たらない。
    """

    def __init__(self, device: str, pad_sec: float):
        self.pad_sec = pad_sec
        self._aligner = None
        self._device = device
        self._error: Exception | None = None

    def get(self):
        if self._aligner is None and self._error is None:
            try:
                try:
                    from scripts.generate_qwen3_tts_data import ForcedAligner
                except ImportError:
                    from generate_qwen3_tts_data import ForcedAligner  # type: ignore

                self._aligner = ForcedAligner(self._device, fallback_mode="skip")
                self._aligner.load()
            except Exception as exc:  # モデル取得失敗などは概算のまま続ける
                self._error = exc
                logger.warning("MMS_FA を読めませんでした(%s)。概算のみで続けます。", exc)
        return self._aligner

    def refine(
        self, audio: np.ndarray, sr: int, text: str, start: float, end: float
    ) -> tuple[float, float] | None:
        aligner = self.get()
        if aligner is None:
            return None
        lo = max(0, int((start - self.pad_sec) * sr))
        hi = min(audio.size, int((end + self.pad_sec) * sr))
        if hi - lo < int(0.2 * sr):
            return None
        try:
            _, tight = aligner.align(audio[lo:hi], sr, [text])
        except Exception as exc:
            logger.debug("FA 失敗 (%.1fs): %s", start, exc)
            return None
        span_start, span_end = tight[0]
        if span_end <= span_start:
            return None
        base = lo / sr
        return base + span_start, base + span_end


# ---------------------------------------------------------------------------
# VAD スナップ
# ---------------------------------------------------------------------------

def speech_regions(audio: np.ndarray, sr: int) -> list[list[float]]:
    try:
        from full_duplex_audio import silero_vad

        return silero_vad(audio, sr)
    except Exception as exc:
        logger.warning("VAD を使えませんでした(%s)。スナップは行いません。", exc)
        return []


def snap_boundary(
    value: float, regions: list[list[float]], starts: list[float],
    is_start: bool, max_shift: float,
) -> float:
    """音声区間の端へ吸着させる。max_shift 以内でしか動かさない。

    1ch なので VAD の区間は相手の声も含む。大きく動かすと別の話者の端へ
    引っ張られるので、ずらせる量を小さく縛る。
    """
    if not regions:
        return value
    i = bisect.bisect_right(starts, value) - 1
    candidates: list[float] = []
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(regions):
            candidates.append(regions[j][0] if is_start else regions[j][1])
    if not candidates:
        return value
    best = min(candidates, key=lambda c: abs(c - value))
    return best if abs(best - value) <= max_shift + 1e-6 else value


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

def format_time(sec: float) -> str:
    sec = max(0.0, sec)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def label_text(line: Line, use_new: bool = True) -> str:
    """ラベルトラックに出す 1 行分の文字列。

    Audacity のラベルはタブ区切りなので、テキスト中のタブ・改行は潰す。
    """
    flags = f"[{'/'.join(line.flags)}]" if (use_new and line.flags) else ""
    text = _LABEL_SANITIZE.sub(" ", line.utt.text).strip()
    return f"{line.utt.speaker}{flags}: {text}"


def write_label_file(path: Path, rows: list[tuple[float, float, str]]) -> None:
    """Audacity の File > Import > Labels で読める形式。

    `開始秒<TAB>終了秒<TAB>ラベル` の 1 行 1 ラベル。BOM を付けると先頭の
    ラベルに化けが出るので、BOM 無しの UTF-8 で書く。
    """
    lines = [f"{start:.6f}\t{end:.6f}\t{text}" for start, end, text in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_audacity_labels(out_dir: Path, lines: list[Line]) -> list[str]:
    """確認用のラベルを話者ごと・要確認ごとに分けて書く。

    Audacity は 1 ファイル 1 ラベルトラックとして読むので、分けて出すと
    User / Staff / 旧ラベルを別トラックとして並べて見比べられる。
    """
    label_dir = out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob("*.txt"):
        stale.unlink()

    written: list[str] = []

    def dump(name: str, picked: list[Line], use_new: bool = True) -> None:
        if not picked:
            return
        rows = [
            (
                (l.new_start if use_new else l.utt.start),
                (l.new_end if use_new else l.utt.end),
                label_text(l, use_new),
            )
            for l in picked
        ]
        rows.sort(key=lambda r: r[0])
        write_label_file(label_dir / f"{name}.txt", rows)
        written.append(f"{name}.txt")

    dump("all", lines)
    for speaker in sorted({l.utt.speaker for l in lines}):
        safe = _FNAME_RE.sub("_", speaker) or "speaker"
        dump(safe, [l for l in lines if l.utt.speaker == speaker])
    dump("flagged", [l for l in lines if l.flags])
    # 旧ラベルも出しておく。新旧を別トラックで並べれば、ずれ方が一目で分かる。
    dump("original", lines, use_new=False)
    return written


def diagnose(lines: list[Line]) -> dict:
    """新旧のずれから、ずれの性質を判定する。

    定数オフセットや線形ドリフト(録音の頭切れ、サンプリングレート違い)なら
    原因が別にあるということなので、そちらを直したほうが確実に直る。
    """
    usable = [l for l in lines if l.confident]
    if len(usable) < 3:
        return {"kind": "unknown", "usable_lines": len(usable)}
    old = np.array([l.utt.start for l in usable], dtype=float)
    delta = np.array([l.new_start - l.utt.start for l in usable], dtype=float)
    slope, intercept = np.polyfit(old, delta, 1)
    residual = delta - (slope * old + intercept)
    residual_std = float(np.std(residual))
    median = float(np.median(delta))
    iqr = float(np.percentile(delta, 75) - np.percentile(delta, 25))

    if residual_std < 0.15 and abs(slope) < 1e-4:
        kind = "constant_offset"
        note = (
            f"ほぼ全体が一定量ずれています(中央値 {median:+.3f} 秒)。"
            "録音の頭切れや別ファイルからの流用が原因の可能性があります。"
        )
    elif residual_std < 0.25:
        kind = "linear_drift"
        note = (
            f"時間に比例してずれが増えています(傾き {slope:+.6f} 秒/秒、"
            f"倍率 {1.0 + slope:.6f})。サンプリングレートの取り違えが疑われます。"
        )
    else:
        kind = "irregular"
        note = (
            f"ずれが行ごとにばらついています(中央値 {median:+.3f} 秒、"
            f"IQR {iqr:.3f} 秒)。行単位で付け直すしかありません。"
        )
    return {
        "kind": kind,
        "note": note,
        "usable_lines": len(usable),
        "delta_median_sec": round(median, 4),
        "delta_iqr_sec": round(iqr, 4),
        "slope_sec_per_sec": round(float(slope), 8),
        "intercept_sec": round(float(intercept), 4),
        "residual_std_sec": round(residual_std, 4),
    }


def write_review_clips(
    out_dir: Path, audio: np.ndarray, sr: int, lines: list[Line], count: int
) -> int:
    """確信度の低いものを中心に、新しい境界で切った WAV を出す。

    前後に余白を付けて出すので、境界が語頭・語尾に合っているかを耳で確かめ
    られる。数値の確信度より、こちらのほうが早い。
    """
    if count <= 0 or not lines:
        return 0
    clip_dir = out_dir / "review_wav"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for stale in clip_dir.glob("*.wav"):
        stale.unlink()

    ranked = sorted(lines, key=lambda l: (l.coverage, -len(l.reading)))
    picked_index = {l.index for l in ranked[: max(1, count // 2)]}
    step = max(1, len(lines) // max(1, count - len(picked_index)))
    for line in lines[::step]:
        if len(picked_index) >= count:
            break
        picked_index.add(line.index)

    pad = 0.5
    written = 0
    for line in lines:
        if line.index not in picked_index:
            continue
        lo = max(0, int((line.new_start - pad) * sr))
        hi = min(audio.size, int((line.new_end + pad) * sr))
        if hi <= lo:
            continue
        text = _FNAME_RE.sub("", _STRIP_RE.sub("", line.utt.text))[:16]
        flag = "_" + "-".join(line.flags) if line.flags else ""
        name = (
            f"{line.index:04d}_{line.utt.speaker}"
            f"_{int(line.new_start // 60):02d}m{line.new_start % 60:04.1f}s"
            f"_cov{int(round(line.coverage * 100)):03d}{flag}_{text}.wav"
        )
        sf.write(clip_dir / _FNAME_RE.sub("", name), audio[lo:hi], sr)
        written += 1
    return written


# ---------------------------------------------------------------------------
# 1 対話分の処理
# ---------------------------------------------------------------------------

def process_wav(
    wav_path: Path, out_root: Path, args: argparse.Namespace,
    holder: WhisperHolder, refiner: Refiner,
) -> dict:
    annotation = args.annotation or annotation_for(wav_path)
    # read_annotation は start でソートするが、いま start は信用できない。
    # 正しいのは TSV の並び順なので、行番号で並べ直す。
    utterances = sorted(read_annotation(annotation), key=lambda u: u.lineno)
    lines = [
        Line(index=i, utt=u, reading=to_reading(u.text))
        for i, u in enumerate(utterances)
    ]

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_sec = audio.size / sr
    logger.info(
        "%s: %.1f 秒 / アノテーション %d 行", wav_path.name, total_sec, len(lines)
    )

    out_dir = out_root / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    words = run_asr(wav_path, out_dir / "asr.json", holder, args.refresh_asr)
    asr_text, asr_times = asr_char_stream(words)
    if not asr_text:
        raise SystemExit(f"{wav_path.name}: ASR から照合できる読みが取れませんでした。")

    speakers = sorted({l.utt.speaker for l in lines})
    anchor_total = 0
    for speaker in speakers:
        group = [l for l in lines if l.utt.speaker == speaker]
        found = coarse_align_speaker(group, asr_text, asr_times, args.kgram)
        anchor_total += found
        logger.info("  %s: %d 発話 / anchor %d", speaker, len(group), found)

    regions: list[list[float]] = []
    if not args.check_only and args.snap_sec > 0:
        regions = speech_regions(audio, sr)
        logger.info("  VAD 音声区間 %d 個", len(regions))
    region_starts = [r[0] for r in regions]

    for line in lines:
        if line.est_start is None or line.est_end is None:
            line.new_start = line.utt.start
            line.new_end = line.utt.end
            line.method = "original"
            line.flags = ("NOTEXT",)
            continue

        flags: list[str] = []
        start, end = line.est_start, line.est_end
        method = "anchor"
        if line.coverage < args.min_coverage:
            flags.append("LOWCOV")

        if not args.check_only and args.refine:
            refined = refiner.refine(audio, sr, line.utt.text, start, end)
            if refined is not None:
                start, end = refined
                method = "fa"
            else:
                flags.append("NOFA")

        if not args.check_only and regions and args.snap_sec > 0:
            start = snap_boundary(start, regions, region_starts, True, args.snap_sec)
            end = snap_boundary(end, regions, region_starts, False, args.snap_sec)
            method += "+vad"

        # 長さの妥当性。読みの文字数から見て極端なものは境界が外れている。
        expected = max(_SEC_PER_KANA * len(line.reading), 0.2)
        duration = end - start
        if duration < 0.35 * expected or duration > 3.0 * expected:
            flags.append("DUR")

        line.new_start = max(0.0, min(start, total_sec))
        line.new_end = max(line.new_start + 0.05, min(end, total_sec))
        line.method = method
        line.flags = tuple(flags)

    # 同一話者内で発話が重ならないよう最小限だけ均す(重なるのは異話者間だけ)。
    for speaker in speakers:
        group = [l for l in lines if l.utt.speaker == speaker]
        for previous, current in zip(group, group[1:]):
            if current.new_start < previous.new_end:
                middle = (previous.new_end + current.new_start) / 2
                previous.new_end = max(previous.new_start + 0.05, middle)
                current.new_start = max(previous.new_end + 0.001, middle)
                current.new_end = max(current.new_end, current.new_start + 0.05)
                previous.flags = tuple(dict.fromkeys(previous.flags + ("CROSS",)))
                current.flags = tuple(dict.fromkeys(current.flags + ("CROSS",)))

    # 確信度の低い行を元の時刻へ戻す運用も選べる。既定は推定値のまま(印を残す)。
    if args.fallback == "original":
        for line in lines:
            if line.flags:
                line.new_start = line.utt.start
                line.new_end = line.utt.end
                line.method = "original"

    report_rows = [
        [
            str(l.index), str(l.utt.lineno), l.utt.speaker, str(len(l.reading)),
            f"{l.utt.start:.3f}", f"{l.utt.end:.3f}",
            f"{l.new_start:.3f}", f"{l.new_end:.3f}",
            f"{l.new_start - l.utt.start:+.3f}", f"{l.new_end - l.utt.end:+.3f}",
            f"{l.utt.end - l.utt.start:.3f}", f"{l.new_end - l.new_start:.3f}",
            f"{l.coverage:.2f}", l.method, ",".join(l.flags), l.utt.text,
        ]
        for l in lines
    ]
    write_tsv(
        out_dir / "report.tsv",
        ["index", "lineno", "speaker", "chars", "old_start", "old_end",
         "new_start", "new_end", "delta_start", "delta_end",
         "old_dur", "new_dur", "coverage", "method", "flags", "text"],
        report_rows,
    )

    # ラベルは確認用なので --check-only でも出す(そのときの境界は概算)。
    labels = write_audacity_labels(out_dir, lines)

    clips = 0
    if not args.check_only:
        write_tsv(
            out_dir / "aligned.tsv",
            ["Speaker", "Transcription", "Start", "End"],
            [
                [l.utt.speaker, l.utt.text,
                 format_time(l.new_start), format_time(l.new_end)]
                for l in lines
            ],
        )
        clips = write_review_clips(out_dir, audio, sr, lines, args.review_clips)

    diagnosis = diagnose(lines)
    flagged = sum(1 for l in lines if l.flags)
    coverage_median = statistics.median([l.coverage for l in lines]) if lines else 0.0
    summary = {
        "name": wav_path.stem,
        "wav": str(wav_path),
        "annotation": str(annotation),
        "audio_sec": round(total_sec, 1),
        "lines": len(lines),
        "speakers": speakers,
        "anchors": anchor_total,
        "coverage_median": round(coverage_median, 3),
        "flagged": flagged,
        "flag_counts": {
            flag: sum(1 for l in lines if flag in l.flags)
            for flag in ("LOWCOV", "NOFA", "DUR", "CROSS", "NOTEXT")
        },
        "labels": labels,
        "review_clips": clips,
        "diagnosis": diagnosis,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "  anchor %d / coverage 中央値 %.2f / 要確認 %d 行",
        anchor_total, coverage_median, flagged,
    )
    if diagnosis.get("note"):
        logger.info("  診断: %s", diagnosis["note"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="音声フォルダ、または WAV(混在可)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"出力先(既定: {DEFAULT_OUT_DIR})")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--annotation", type=Path, default=None,
                        help="アノテーション TSV を明示指定(WAV を 1 つ渡すときのみ)")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--refresh-asr", action="store_true",
                        help="ASR キャッシュを無視して書き起こし直す")
    parser.add_argument("--kgram", type=int, default=6,
                        help="anchor に使う読みの n-gram 長(既定 6)。"
                             "短くすると anchor は増えるが誤対応も増える")
    parser.add_argument("--min-coverage", type=float, default=0.3,
                        help="この割合を下回る行に LOWCOV 印を付ける(既定 0.3)")
    parser.add_argument("--refine", dest="refine", action="store_true", default=True,
                        help="MMS_FA で境界を実測する(既定 on)")
    parser.add_argument("--no-refine", dest="refine", action="store_false")
    parser.add_argument("--refine-pad-sec", type=float, default=2.0,
                        help="強制アライメントに渡す窓の余白(既定 2.0 秒)")
    parser.add_argument("--snap-sec", type=float, default=0.4,
                        help="VAD の音声区間端へ吸着させる最大量(既定 0.4 秒、0 で無効)")
    parser.add_argument("--fallback", choices=["estimate", "original"],
                        default="estimate",
                        help="印の付いた行の扱い。estimate=推定値のまま(既定)、"
                             "original=元の時刻へ戻す")
    parser.add_argument("--review-clips", type=int, default=12,
                        help="確認用に切り出す WAV の数(既定 12、0 で無効)")
    parser.add_argument("--check-only", action="store_true",
                        help="ずれの性質だけ見る。aligned.tsv も WAV も出さない"
                             "(ラベルは概算のまま出す)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wavs = collect_inputs(args.inputs, args.recursive)
    if args.annotation is not None and len(wavs) > 1:
        raise SystemExit("--annotation は WAV を 1 つだけ渡すときに使ってください。")
    require_annotations(wavs, args.annotation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    holder = WhisperHolder(args.whisper_model, args.device, args.compute_type)
    refiner = Refiner(args.device, args.refine_pad_sec)

    summaries = [
        process_wav(wav, args.out_dir, args, holder, refiner) for wav in wavs
    ]
    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "settings": {
            "whisper_model": args.whisper_model,
            "kgram": args.kgram,
            "refine": args.refine and not args.check_only,
            "refine_pad_sec": args.refine_pad_sec,
            "snap_sec": args.snap_sec,
            "fallback": args.fallback,
            "check_only": args.check_only,
        },
        "dialogues": summaries,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== 再アライメント ===")
    print(f"{args.out_dir.resolve()}")
    for summary in summaries:
        print(
            f"  {summary['name']:<24} {summary['lines']:>4} 行"
            f"  anchor {summary['anchors']:>4}"
            f"  coverage {summary['coverage_median']:.2f}"
            f"  要確認 {summary['flagged']:>3}"
            f"  [{summary['diagnosis'].get('kind', '?')}]"
        )
        if summary["diagnosis"].get("note"):
            print(f"    {summary['diagnosis']['note']}")
    print("\nAudacity で確認する:")
    print("  元の WAV を開き、File > Import > Labels で labels/ の中を読む。")
    print("  all.txt(新) と original.txt(旧) を両方読めば別トラックで並ぶ。")
    for summary in summaries:
        print(f"  {summary['out_dir']}/labels/")
    print("\nreport.tsv の flags 列(LOWCOV/NOFA/DUR/CROSS)が付いた行は特に確認を。")
    if args.check_only:
        print("--check-only なので aligned.tsv は出していません(ラベルは概算)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
