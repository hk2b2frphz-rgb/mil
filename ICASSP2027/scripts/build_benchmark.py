#!/usr/bin/env python3
"""Build the clarification benchmark dataset (audio + manifest).

Pipeline per acoustic base item:
  1. Obtain the clean carrier audio: TTS-synthesize the utterance
     (MASSIVE; Qwen3-TTS, one call so prosody is natural across the slot)
     or load the real corpus recording (SLURP).
  2. Forced-align (MMS_FA) the pre/slot/post texts to locate the slot span.
  3. Render every corruption condition from the shared registry, writing
     the corrupted wav, the clean wav, and the pre-synthesized repair turn
     (always TTS).
  4. Optionally annotate every case with the weak-ASR recoverability
     oracle (--oracle).

Corpora x languages:
  --language en --corpus massive       MASSIVE en-US, TTS (primary)
  --language en --corpus slurp         SLURP real recordings
                                       (--slurp-jsonl / --slurp-audio-dir)
  --language ja --corpus massive       MASSIVE ja-JP, TTS (generality)
  --demo                               built-in items, smoke tests only

Run on a GPU node:
  uv run python ICASSP2027/scripts/build_benchmark.py \
      --out-dir ICASSP2027/runs/bench_en_massive \
      --language en --corpus massive --max-items 60 --oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "eval", REPO_ROOT / "scripts",
          REPO_ROOT / "ICASSP2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clarify import corpora, corruptions, slots  # noqa: E402
from clarify.lang import get_pack  # noqa: E402
from clarify.scenario import (  # noqa: E402
    BaseItem, BenchmarkCase, case_seed, write_manifest,
)

# Reuse the existing evaluation TTS/alignment stack.
import build_full_duplex_ja_dataset as fdb  # noqa: E402
from full_duplex_audio import write_wav_mono  # noqa: E402

DEMO_ITEMS: dict[str, list[dict]] = {
    "ja": [
        {"id": "demo0", "intent": "alarm_set",
         "annot_utt": "明日の[time : 朝7時]にアラームをかけて",
         "utt": "明日の朝7時にアラームをかけて"},
        {"id": "demo1", "intent": "calendar_set",
         "annot_utt": "[date : 金曜日]に歯医者の予定を入れて",
         "utt": "金曜日に歯医者の予定を入れて"},
        {"id": "demo2", "intent": "play_music",
         "annot_utt": "[artist_name : 米津玄師]の曲をかけて",
         "utt": "米津玄師の曲をかけて"},
        {"id": "demo3", "intent": "transport_query",
         "annot_utt": "[place_name : 横浜駅]までの終電を調べて",
         "utt": "横浜駅までの終電を調べて"},
    ],
    "en": [
        {"id": "demo0", "intent": "alarm_set",
         "annot_utt": "wake me up at [time : seven am] tomorrow",
         "utt": "wake me up at seven am tomorrow"},
        {"id": "demo1", "intent": "calendar_set",
         "annot_utt": "add a dentist appointment on [date : friday]",
         "utt": "add a dentist appointment on friday"},
        {"id": "demo2", "intent": "play_music",
         "annot_utt": "play some songs by [artist_name : taylor swift]",
         "utt": "play some songs by taylor swift"},
        {"id": "demo3", "intent": "transport_query",
         "annot_utt": "find the last train to [place_name : central station]",
         "utt": "find the last train to central station"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--language", default="en", choices=["en", "ja"])
    parser.add_argument("--corpus", default="massive",
                        choices=["massive", "slurp"])
    parser.add_argument("--split", default="test",
                        help="MASSIVE split for base items.")
    parser.add_argument("--massive-jsonl", type=Path, default=None,
                        help="Pre-exported MASSIVE rows (offline).")
    parser.add_argument("--slurp-jsonl", type=Path, default=None,
                        help="SLURP metadata jsonl (e.g. dataset/slurp/test.jsonl).")
    parser.add_argument("--slurp-audio-dir", type=Path, default=None,
                        help="Directory containing slurp_real audio files.")
    parser.add_argument("--demo", action="store_true",
                        help="Use built-in demo items (smoke test).")
    parser.add_argument("--max-items", type=int, default=60)
    parser.add_argument("--conditions", default="all",
                        help="Comma-separated condition names or 'all'.")
    parser.add_argument("--no-underspecified", action="store_true")
    parser.add_argument("--oracle", action="store_true",
                        help="Run the weak-ASR recoverability oracle.")
    parser.add_argument("--oracle-model", default="small")
    # TTS options mirror eval/build_full_duplex_ja_dataset.py.
    parser.add_argument("--tts-backend", default="auto",
                        choices=["auto", "qwen3", "pyopenjtalk"])
    parser.add_argument("--tts-model",
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--tts-speaker", default=None,
                        help="Default: language pack's speaker.")
    parser.add_argument("--tts-language", default=None,
                        help="Default: language pack's TTS language.")
    parser.add_argument("--tts-instruct", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict]:
    if args.demo:
        rows = [dict(r) for r in DEMO_ITEMS[args.language]]
        for row in rows:
            row["source"] = "demo"
            row["audio_file"] = None
        return rows
    return corpora.load_corpus(
        args.corpus, args.language, split=args.split,
        massive_jsonl=args.massive_jsonl,
        slurp_jsonl=args.slurp_jsonl,
        slurp_audio_dir=args.slurp_audio_dir,
    )


def load_real_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load a corpus recording (flac/wav) as mono float32 at sample_rate."""
    import librosa

    pcm, _sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(pcm, dtype=np.float32)


def locate_slot_span(
    aligner, pcm: np.ndarray, sample_rate: int, item: BaseItem
) -> tuple[float, float] | None:
    """Forced-align pre/slot/post; return the slot's tight (start, end)."""
    texts = [t for t in (item.pre_text, item.slot_text, item.post_text) if t]
    slot_index = 1 if item.pre_text else 0
    if aligner is None:
        return None
    try:
        _expanded, tight = aligner.align(pcm, sample_rate, texts)
    except Exception as exc:  # alignment can fail on odd tokens
        print(f"[bench] alignment failed for {item.base_id}: {exc}")
        return None
    if len(tight) != len(texts):
        return None
    start, end = tight[slot_index]
    if end - start < 0.05:
        return None
    return float(start), float(end)


def proportional_span(
    pcm: np.ndarray, sample_rate: int, item: BaseItem
) -> tuple[float, float]:
    """Character-proportional fallback span (recorded as such)."""
    total = len(pcm) / sample_rate
    n_pre = len(item.pre_text)
    n_slot = len(item.slot_text)
    n_all = max(1, n_pre + n_slot + len(item.post_text))
    start = total * n_pre / n_all
    end = total * (n_pre + n_slot) / n_all
    return start, end


def main() -> int:
    args = parse_args()
    pack = get_pack(args.language)
    tts_speaker = args.tts_speaker or pack.tts_default_speaker
    out_dir: Path = args.out_dir
    audio_dir = out_dir / "audio"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args)
    items = slots.select_base_items(rows, language=args.language,
                                    max_items=args.max_items)
    if not items:
        raise SystemExit("no acoustic base items selected; check input rows")
    n_real = sum(1 for i in items if i.audio_source == "real")
    print(f"[bench] selected {len(items)} acoustic base items "
          f"({n_real} with real audio) [{args.language}/{args.corpus}]")
    if not args.no_underspecified and args.corpus == "massive":
        under = slots.build_underspecified_items(args.language)
        print(f"[bench] plus {len(under)} underspecified items")
    else:
        under = []

    all_conditions = corruptions.build_default_conditions()
    if args.conditions != "all":
        wanted = set(args.conditions.split(","))
        unknown = wanted - {c.name for c in all_conditions}
        if unknown:
            raise SystemExit(f"unknown conditions: {sorted(unknown)}")
        all_conditions = [c for c in all_conditions if c.name in wanted]

    # Fake args namespace for the shared TTS initializer.
    args.tts_speaker = tts_speaker
    if args.tts_language is None:
        args.tts_language = pack.tts_language
    tts_backend, tts_engine = fdb.initialize_tts(args)
    if args.language == "en" and tts_backend == "pyopenjtalk":
        raise SystemExit(
            "pyopenjtalk cannot synthesize English; qwen-tts must be "
            "available for --language en (use --tts-backend qwen3)."
        )
    aligner = fdb.initialize_aligner(True, args.device)
    sr = args.sample_rate

    def synth(text: str) -> np.ndarray:
        return fdb.synthesize(
            text, sr, args.speed, tts_backend, tts_engine, tts_speaker,
            speaker_role="user",
        )

    cases: list[BenchmarkCase] = []
    span_fallbacks = 0
    for item in items + under:
        item.validate()
        if item.audio_source == "real":
            clean = load_real_audio(Path(item.meta["audio_file"]), sr)
        else:
            clean = synth(item.utterance_text)
        repair = synth(item.repair_text)
        clean_rel = f"audio/{item.base_id}__clean.wav"
        repair_rel = f"audio/{item.base_id}__repair.wav"
        write_wav_mono(out_dir / clean_rel, clean, sr)
        write_wav_mono(out_dir / repair_rel, repair, sr)

        if item.arm == "acoustic":
            span = locate_slot_span(aligner, clean, sr, item)
            span_source = "forced_alignment"
            if span is None:
                span = proportional_span(clean, sr, item)
                span_source = "proportional_fallback"
                span_fallbacks += 1
            conds = all_conditions
        else:
            span, span_source = None, "n/a"
            conds = [c for c in all_conditions if c.name == "clean"]

        for cond in conds:
            case_id = f"{item.base_id}__{cond.name}"
            seed = case_seed(item.base_id, cond.name)
            rng = np.random.default_rng(seed)
            if cond.name == "clean" or span is None:
                corrupted = clean.copy()
            else:
                corrupted = cond.apply(clean, sr, span[0], span[1], rng)
            audio_rel = f"audio/{case_id}.wav"
            write_wav_mono(out_dir / audio_rel, corrupted, sr)
            expected = cond.expected_behavior
            if item.arm == "underspecified":
                expected = "ask"
            cases.append(BenchmarkCase(
                case_id=case_id,
                base=item,
                condition=cond.name,
                expected_behavior=expected,
                audio_path=audio_rel,
                clean_audio_path=clean_rel,
                repair_audio_path=repair_rel,
                span_start_sec=None if span is None else round(span[0], 4),
                span_end_sec=None if span is None else round(span[1], 4),
                sample_rate=sr,
                seed_base=seed,
                tts={"backend": tts_backend, "speaker": tts_speaker,
                     "model": args.tts_model, "speed": args.speed,
                     "span_source": span_source,
                     "carrier": item.audio_source},
            ))

    write_manifest(out_dir, cases, profile_extra={
        "language": args.language,
        "corpus": args.corpus,
        "tts_backend": tts_backend,
        "n_acoustic_items": len(items),
        "n_underspecified_items": len(under),
        "span_alignment_fallbacks": span_fallbacks,
        "source": ("demo" if args.demo else
                   str(args.massive_jsonl or args.slurp_jsonl
                       or f"{args.corpus}:{args.split}")),
    })
    print(f"[bench] wrote {len(cases)} cases to {out_dir} "
          f"(span fallbacks: {span_fallbacks})")

    if args.oracle:
        from clarify.asr_oracle import run_oracle
        case_dicts = [c.to_json() for c in cases]
        oracle_path = out_dir / "asr_oracle.jsonl"
        results = run_oracle(case_dicts, out_dir, oracle_path,
                             model_size=args.oracle_model,
                             device=args.device)
        n_rec = sum(r.slot_recovered for r in results)
        print(f"[bench] oracle: slot recovered in {n_rec}/{len(results)} "
              f"cases -> {oracle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
