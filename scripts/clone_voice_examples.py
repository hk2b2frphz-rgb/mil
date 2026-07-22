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


def build_aggregate_reference(
    analysis_dir: Path, speaker: str, segs: list[dict[str, Any]],
    out_wav: Path, max_sec: float, gap_sec: float,
) -> tuple[Path, str, float, int] | None:
    """話者の重畳なし区間を(時系列順に)連結した1本の参照 WAV を作る。

    x-vector は音声全体をプーリングして1ベクトルにするので、長い連結を渡せば
    実質「全クリップ集約の話者埋め込み」になる。max_sec で頭打ち(収穫逓減+
    メモリ)。相槌も含め話者の音声はすべて使う(声質推定には情報が多いほど良い)。
    戻り値: (参照wav, 連結テキスト, 総秒数, 使用クリップ数) or None。
    """
    import numpy as np
    import soundfile as sf

    clips = sorted(
        (s for s in segs
         if s.get("speaker") == speaker and float(s.get("overlap_sec", 0.0)) == 0.0),
        key=lambda s: float(s["start"]),
    )
    pieces: list[np.ndarray] = []
    texts: list[str] = []
    total = 0.0
    sr0: int | None = None
    used = 0
    for s in clips:
        wav = find_segment_wav(analysis_dir, speaker, s["_index"])
        if wav is None:
            continue
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr0 is None:
            sr0 = int(sr)
        elif sr != sr0:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr0)
        pieces.append(audio)
        if gap_sec > 0:
            pieces.append(np.zeros(int(gap_sec * sr0), dtype=np.float32))
        txt = str(s.get("text", "")).strip()
        if txt:
            texts.append(txt)
        total += len(audio) / sr0
        used += 1
        if total >= max_sec:
            break
    if not pieces or sr0 is None:
        return None
    agg = np.concatenate(pieces).astype(np.float32)
    sf.write(str(out_wav), agg, sr0)
    # x_vector_only では ref_text は基本無視されるが、API 都合で連結テキストを渡す。
    return out_wav, " ".join(texts)[:500], round(total, 1), used


def select_references(
    segs: list[dict[str, Any]], speaker: str, min_sec: float, max_sec: float, n: int
) -> list[dict[str, Any]]:
    """参照に向く区間を上位 n 件返す: 重畳なし・相槌でない・テキストありの中から、
    推奨長レンジ内を長い順に。レンジ内が足りなければ全 clean から補う。"""
    clean = [
        s for s in segs
        if s.get("speaker") == speaker
        and float(s.get("overlap_sec", 0.0)) == 0.0
        and not s.get("is_aizuchi", False)
        and str(s.get("text", "")).strip()
    ]
    in_range = sorted(
        (s for s in clean if min_sec <= _dur(s) <= max_sec), key=_dur, reverse=True
    )
    if len(in_range) >= n:
        return in_range[:n]
    # レンジ内が足りない分はレンジ外の clean(長い順)で埋める。
    rest = sorted(
        (s for s in clean if s not in in_range), key=_dur, reverse=True
    )
    return (in_range + rest)[:n]


def load_examples(args: argparse.Namespace) -> list[str]:
    if args.examples_file:
        lines = Path(args.examples_file).read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if ln.strip()]
        if not texts:
            raise SystemExit(f"例文ファイルが空です: {args.examples_file}")
        return texts
    return list(DEFAULT_EXAMPLES)


def resolve_references(
    args: argparse.Namespace,
) -> list[tuple[Path, str, dict[str, Any] | None]]:
    """(ref_wav, ref_text, timeline セグメント or None) の一覧を返す。

    --ref-wav 指定時はその1件のみ。未指定なら analysis-dir から上位
    --num-refs 件を自動選別する。"""
    if args.ref_wav:
        ref_wav = Path(args.ref_wav)
        if not ref_wav.is_file():
            raise SystemExit(f"--ref-wav が存在しません: {ref_wav}")
        if not args.ref_text:
            raise SystemExit("--ref-wav を使うときは --ref-text も必須です")
        return [(ref_wav, args.ref_text.strip(), None)]

    if not args.analysis_dir:
        raise SystemExit("--analysis-dir か、--ref-wav/--ref-text のどちらかが必要です")
    analysis_dir = Path(args.analysis_dir)
    segs = load_timeline(analysis_dir)
    chosens = select_references(
        segs, args.speaker, args.min_ref_sec, args.max_ref_sec, args.num_refs
    )
    if not chosens:
        raise SystemExit(
            f"話者 {args.speaker} に参照向きの区間が見つかりません"
            "(重畳なし・相槌でない・テキストありが必要)。--ref-wav/--ref-text で明示指定してください。"
        )
    refs: list[tuple[Path, str, dict[str, Any] | None]] = []
    for chosen in chosens:
        ref_wav = find_segment_wav(analysis_dir, args.speaker, chosen["_index"])
        if ref_wav is None:
            logger.warning(
                "区間 %04d の WAV が見つからずスキップ: segments/%s/",
                chosen["_index"], args.speaker,
            )
            continue
        logger.info(
            "参照候補: 話者%s idx=%d dur=%.1fs text=%r",
            args.speaker, chosen["_index"], _dur(chosen), chosen["text"],
        )
        refs.append((ref_wav, str(chosen["text"]).strip(), chosen))
    if not refs:
        raise SystemExit("参照候補の WAV が1件も見つかりませんでした")
    return refs


def resolve_modes(args: argparse.Namespace) -> list[str]:
    """使うクローンモードの一覧。--x-vector-only が優先。"""
    if args.x_vector_only:
        return ["x-vector"]
    alias = {
        "in-context": "in-context", "incontext": "in-context", "ic": "in-context",
        "x-vector": "x-vector", "xvector": "x-vector", "xv": "x-vector",
    }
    modes: list[str] = []
    for raw in args.modes.split(","):
        raw = raw.strip().lower()
        if not raw:
            continue
        if raw not in alias:
            raise SystemExit(f"未知のモード: {raw!r}(in-context / x-vector)")
        canonical = alias[raw]
        if canonical not in modes:
            modes.append(canonical)
    return modes or ["in-context"]


def load_qwen_model(args: argparse.Namespace, model_id: str):
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

    logger.info("Qwen3-TTS 読み込み中: %s (device=%s dtype=%s attn=%s)",
                model_id, device_map, args.dtype, args.attn_impl)
    logger.info("※初回はモデル(数GB)のダウンロードで時間がかかります")
    try:
        with Heartbeat("モデルDL/ロード"):
            return Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if load_kwargs.pop("attn_implementation", None) is not None:
            logger.warning("attn=%s の読み込みに失敗。既定の attention で再試行: %s",
                           args.attn_impl, exc)
            with Heartbeat("モデルDL/ロード(再試行)"):
                return Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        raise


def free_model(model) -> None:
    """次のモデルをロードする前に VRAM を解放する。"""
    try:
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # 解放失敗は致命的ではない
        logger.warning("モデル解放時の警告: %s", exc)


def run_customvoice(
    model, examples: list[str], speaker: str, language: str,
    instruct: str | None, subdir: Path,
) -> list[str]:
    """CustomVoice プリセット話者で例文を合成し、ファイル名一覧を返す。"""
    import numpy as np
    import soundfile as sf
    import torch

    out_files: list[str] = []
    total = len(examples)
    logger.info("CustomVoice(%s)で %d 文を合成中...", speaker, total)
    with Heartbeat(f"CustomVoice合成 {speaker}"):
        for i, text in enumerate(examples):
            kwargs: dict[str, Any] = {"text": text, "language": language, "speaker": speaker}
            if instruct:
                kwargs["instruct"] = instruct
            with torch.no_grad():
                wavs, sr = model.generate_custom_voice(**kwargs)
            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            path = subdir / f"example_{i:02d}.wav"
            sf.write(str(path), audio, int(sr))
            out_files.append(path.name)
    return out_files


def run_clone_combo(model, tag: str, ref, mode: str, examples: list[str],
                    args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf

    ref_wav, ref_text, chosen = ref
    x_only = mode == "x-vector"
    subdir = args.out_dir / tag
    subdir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s (参照 idx=%s dur=%.1fs) ===", tag,
                chosen["_index"] if chosen else "-",
                _dur(chosen) if chosen else -1.0)
    with Heartbeat(f"プロンプト作成 {tag}"):
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(ref_wav), ref_text=ref_text, x_vector_only_mode=x_only)
    with Heartbeat(f"合成 {tag}"):
        wavs, sr = model.generate_voice_clone(
            text=examples, language=[args.language] * len(examples),
            voice_clone_prompt=prompt_items)
    out_files: list[str] = []
    for i, wav in enumerate(wavs):
        path = subdir / f"example_{i:02d}.wav"
        sf.write(str(path), wav, sr)
        out_files.append(path.name)
    shutil.copy(ref_wav, subdir / "reference.wav")
    combo = {
        "tag": tag, "mode": mode, "sample_rate": int(sr),
        "reference": {
            "wav": str(ref_wav), "text": ref_text,
            "timeline_index": chosen["_index"] if chosen else None,
            "duration_sec": round(_dur(chosen), 2) if chosen else None,
        },
        "files": out_files,
    }
    (subdir / "manifest.json").write_text(
        json.dumps(combo, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("  -> %s に %d 文出力", subdir, len(out_files))
    return combo


def run_xvectorall_combo(model, tag: str, examples: list[str],
                         args: argparse.Namespace) -> dict[str, Any] | None:
    import soundfile as sf

    analysis_dir = Path(args.analysis_dir)
    segs = load_timeline(analysis_dir)
    agg_wav = args.out_dir / "_xvector_all_ref.wav"
    built = build_aggregate_reference(
        analysis_dir, args.speaker, segs, agg_wav,
        args.xvector_max_sec, args.xvector_gap_sec)
    if built is None:
        logger.warning("集約 x-vector: 連結できる区間がなくスキップ")
        return None
    agg_path, agg_text, agg_sec, n_clips = built
    subdir = args.out_dir / tag
    subdir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s (%d 区間 / %.1f 秒を集約) ===", tag, n_clips, agg_sec)
    with Heartbeat(f"プロンプト作成 {tag}"):
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(agg_path), ref_text=agg_text, x_vector_only_mode=True)
    with Heartbeat(f"合成 {tag}"):
        wavs, sr = model.generate_voice_clone(
            text=examples, language=[args.language] * len(examples),
            voice_clone_prompt=prompt_items)
    out_files: list[str] = []
    for i, wav in enumerate(wavs):
        path = subdir / f"example_{i:02d}.wav"
        sf.write(str(path), wav, sr)
        out_files.append(path.name)
    shutil.copy(agg_path, subdir / "reference.wav")
    combo = {
        "tag": tag, "mode": "x-vector-all", "sample_rate": int(sr),
        "reference": {"wav": str(agg_path), "aggregated_sec": agg_sec, "n_clips": n_clips},
        "files": out_files,
    }
    (subdir / "manifest.json").write_text(
        json.dumps(combo, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("  -> %s に %d 文出力", subdir, len(out_files))
    return combo


def run_customvoice_combo(cv_model, tag: str, examples: list[str],
                          args: argparse.Namespace) -> dict[str, Any]:
    subdir = args.out_dir / tag
    subdir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s (プリセット話者) ===", tag)
    cv_files = run_customvoice(
        cv_model, examples, args.customvoice_speaker,
        args.language, args.customvoice_instruct, subdir)
    combo = {
        "tag": tag, "mode": "customvoice",
        "reference": {
            "wav": None, "text": None,
            "preset_speaker": args.customvoice_speaker,
            "model": args.customvoice_model,
            "instruct": args.customvoice_instruct,
        },
        "files": cv_files,
    }
    (subdir / "manifest.json").write_text(
        json.dumps(combo, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("  -> %s に %d 文出力", subdir, len(cv_files))
    return combo


def build_plan(refs, modes: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    """決定的なコンボ計画。全シャード・aggregate で同じ計画を再構成できる。

    順序は Base モデルのタスク(clone → xvectorall)を先に、CustomVoice を最後に
    置く。%num_shards で分配したとき、1シャードが両モデルを持つ場合でも順に処理
    すればモデル切替が1回で済む。"""
    plan: list[dict[str, Any]] = []
    for k in range(len(refs)):
        for mode in modes:
            plan.append({"kind": "clone", "tag": f"ref{k:02d}_{mode}", "k": k, "mode": mode})
    if args.xvector_all and not args.ref_wav and args.analysis_dir:
        segs = load_timeline(Path(args.analysis_dir))
        has_clean = any(
            s.get("speaker") == args.speaker and float(s.get("overlap_sec", 0.0)) == 0.0
            for s in segs
        )
        if has_clean:
            plan.append({"kind": "xvectorall", "tag": f"xvectorall_{args.speaker}"})
        else:
            logger.warning("集約 x-vector: 話者%s にクリーン区間がなく plan から除外", args.speaker)
    if args.customvoice:
        plan.append({"kind": "customvoice", "tag": f"customvoice_{args.customvoice_speaker}"})
    return plan


def run_shard(args: argparse.Namespace, plan: list[dict[str, Any]], refs,
              examples: list[str], shard_index: int, num_shards: int) -> None:
    """plan を num_shards で分配し、このシャード担当分だけ合成する。

    モデルは遅延ロード: clone/xvectorall で Base、customvoice で CustomVoice。
    両方を担当する場合は切替時に前のモデルを解放して VRAM を空ける。"""
    my_tasks = [t for i, t in enumerate(plan) if i % num_shards == shard_index]
    if not my_tasks:
        logger.info("shard %d/%d: 担当タスクなし", shard_index, num_shards)
        return
    logger.info("shard %d/%d: %d タスク %s",
                shard_index, num_shards, len(my_tasks), [t["tag"] for t in my_tasks])

    base_model = None
    cv_model = None
    try:
        for t in my_tasks:
            if t["kind"] in ("clone", "xvectorall"):
                if base_model is None:
                    if cv_model is not None:
                        free_model(cv_model)
                        cv_model = None
                    base_model = load_qwen_model(args, args.model)
                if t["kind"] == "clone":
                    run_clone_combo(base_model, t["tag"], refs[t["k"]], t["mode"], examples, args)
                else:
                    run_xvectorall_combo(base_model, t["tag"], examples, args)
            elif t["kind"] == "customvoice":
                if cv_model is None:
                    if base_model is not None:
                        free_model(base_model)
                        base_model = None
                    cv_model = load_qwen_model(args, args.customvoice_model)
                run_customvoice_combo(cv_model, t["tag"], examples, args)
    finally:
        if base_model is not None:
            free_model(base_model)
        if cv_model is not None:
            free_model(cv_model)


def aggregate_and_verify(args: argparse.Namespace, plan: list[dict[str, Any]],
                         examples: list[str]) -> None:
    """全シャード完了後: 各コンボの完全性を検証し、index.json を書く。

    期待コンボ数 = len(plan)。各コンボは manifest.json と len(examples) 本の
    example_*.wav が揃って初めて OK。1つでも欠ければ非ゼロ終了(ジョブ側で検知)。"""
    expected = len(examples)
    combos: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    sample_rate = 0
    for t in plan:
        subdir = args.out_dir / t["tag"]
        mani = subdir / "manifest.json"
        wavs = sorted(subdir.glob("example_*.wav"))
        if mani.is_file():
            combo = json.loads(mani.read_text(encoding="utf-8"))
            combos.append(combo)
            sample_rate = combo.get("sample_rate", sample_rate) or sample_rate
        if not mani.is_file() or len(wavs) != expected:
            missing.append({"tag": t["tag"], "found": len(wavs),
                            "expected": expected, "manifest": mani.is_file()})

    index = {
        "clone_model": args.model,
        "customvoice_model": args.customvoice_model if args.customvoice else None,
        "language": args.language,
        "speaker": args.speaker,
        "sample_rate": sample_rate,
        "num_refs": args.num_refs,
        "examples": examples,
        "combos": combos,
        "verification": {
            "expected_combos": len(plan),
            "complete_combos": len(plan) - len(missing),
            "missing": missing,
        },
    }
    (args.out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 耳で確認 ===", file=sys.stderr)
    print(f"{args.out_dir.resolve()}", file=sys.stderr)
    for combo in combos:
        print(f"  {combo['tag']}/  ({combo.get('mode')})", file=sys.stderr)
    print("  index.json  全組み合わせの一覧+検証結果", file=sys.stderr)

    if missing:
        logger.error("検証失敗: %d/%d コンボが不完全", len(missing), len(plan))
        for m in missing:
            logger.error("  %s: %d/%d files, manifest=%s",
                         m["tag"], m["found"], m["expected"], m["manifest"])
        raise SystemExit(1)
    logger.info("検証OK: %d コンボすべて %d 文そろっています", len(plan), expected)


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
    parser.add_argument("--num-refs", type=int, default=3,
                        help="自動選別する参照音声の数(既定 3。それぞれで合成し聴き比べ)")
    parser.add_argument("--modes", default="in-context,x-vector",
                        help="試すクローンモード(カンマ区切り: in-context,x-vector)")
    parser.add_argument("--x-vector-only", action="store_true",
                        help="x-vector モードのみに絞る(--modes より優先)")
    parser.add_argument("--xvector-all", dest="xvector_all", action="store_true",
                        default=True,
                        help="話者の全クリーン区間を連結した1本から x-vector を取る"
                             "集約モードも試す(既定 ON)")
    parser.add_argument("--no-xvector-all", dest="xvector_all", action="store_false",
                        help="集約 x-vector モードを試さない")
    parser.add_argument("--xvector-max-sec", type=float, default=90.0,
                        help="集約 x-vector に使う参照の総秒数の上限(既定 90)")
    parser.add_argument("--xvector-gap-sec", type=float, default=0.1,
                        help="集約参照でクリップ間に挟む無音秒(既定 0.1)")
    parser.add_argument("--customvoice", dest="customvoice", action="store_true",
                        default=True,
                        help="比較用に CustomVoice プリセットも合成(既定 ON)")
    parser.add_argument("--no-customvoice", dest="customvoice", action="store_false",
                        help="CustomVoice ベースラインを合成しない")
    parser.add_argument("--customvoice-model",
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="CustomVoice モデル ID")
    parser.add_argument("--customvoice-speaker", default="Ono_Anna",
                        help="CustomVoice のプリセット話者(既定 Ono_Anna)")
    parser.add_argument("--customvoice-instruct", default=None,
                        help="CustomVoice の instruct(既定なし=素のプリセット)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="並列シャード総数(通常は使用 GPU 数)。PBS が設定")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="このプロセスのシャード番号(0 始まり)")
    parser.add_argument("--aggregate", action="store_true",
                        help="合成せず、各コンボの完全性を検証して index.json を書く")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards は 1 以上")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index は 0..num_shards-1 の範囲")

    refs = resolve_references(args)
    modes = resolve_modes(args)
    examples = load_examples(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(refs, modes, args)
    logger.info("コンボ計画: %d 通り %s", len(plan), [t["tag"] for t in plan])

    # 検証のみ(全シャード完了後に呼ばれる)。
    if args.aggregate:
        aggregate_and_verify(args, plan, examples)
        return

    # このシャード担当分を合成。
    run_shard(args, plan, refs, examples, args.shard_index, args.num_shards)

    # 単一プロセス実行(num_shards=1)ならそのまま検証+index も書く。
    # 複数シャード時は各シャードは合成のみ行い、検証は別途 --aggregate で行う。
    if args.num_shards == 1:
        aggregate_and_verify(args, plan, examples)


if __name__ == "__main__":
    main()
