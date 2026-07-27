#!/usr/bin/env python3
"""実話者2名の声で 1 対話を合成する(聴取確認用テスト)。

clone_voice_examples.py が例文を 1 話者ぶん喋らせるのに対し、こちらは user /
moshi の 2 ロールにそれぞれ別の参照を割り当て、台本を頭から通して 1 本の対話
WAV にする。狙いは「実相談員 × 実相談者の声で対話が成立して聴こえるか」の確認。

推論は clone_voice_examples.py と同じ qwen-tts パッケージ直叩き(メイン環境)で、
vLLM-Omni は使わない。バッチ生成のクローン経路(generate_qwen3_tts_data.py の
--qwen-clone-*)は vLLM-Omni 専用なので、テスト段階ではこちらの方が早い。

参照は resolve_clone_refs.py と共通の解決を使う:
  --analysis-dir       analyze_real_dialogue.py の出力から話者ごとに refNN を選別
  --clone-out-dir-*    clone_voice_examples.py の出力の refNN をそのまま使う

台本は次の優先順で決まる:
  --script-file        1 行 1 ターンの素テキスト('user: ...' / 'moshi: ...')。
                       台詞をいじるだけならこれが一番速い。
  --dialogues-jsonl    generate_qwen3_tts_data.py と同じ形式の対話 JSONL。
  (どちらも未指定)     内蔵の短い台本。

参照の書き起こしが whisper の誤りで汚れている場合は --user-ref-text /
--moshi-ref-text で上書きできる(timeline.jsonl は書き換えない)。

使い方:
    uv run python scripts/clone_dialogue_pilot.py \
        --analysis-dir data/real_dialogue/<jobid>/<stem> \
        --user-speaker A --moshi-speaker B \
        --out-dir data/clone_dialogue_pilot/test01

出力 <out_dir>/ 配下:
  - dialogue.wav        連結した対話全体(モノラル)
  - dialogue_stereo.wav 同じタイミングで L=user / R=moshi に振り分けたもの。
                        1 本で通して聴きながらロールを聴き分けられる
                        (analyze_real_dialogue.py の stereo_diarized.wav と同趣旨)
  - turns/00_user.wav   ターンごとの合成結果
  - manifest.json       参照・台本・タイミングの記録
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.clone_voice_examples import (
        free_model,
        generate_clone_batched,
        load_qwen_model,
    )
    from scripts.resolve_clone_refs import (
        resolve_from_analysis_dir,
        resolve_from_clone_out_dir,
    )
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from clone_voice_examples import (  # type: ignore[no-redef]
        free_model,
        generate_clone_batched,
        load_qwen_model,
    )
    from resolve_clone_refs import (  # type: ignore[no-redef]
        resolve_from_analysis_dir,
        resolve_from_clone_out_dir,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROLES = ("user", "moshi")

# 内蔵台本。短く、相槌と間の取り方が見える程度の長さにしてある。
DEFAULT_TURNS: list[dict[str, str]] = [
    {"speaker": "user",  "text": "こんばんは。相談というほどでもないんですが、少し話してもいいですか。"},
    {"speaker": "moshi", "text": "もちろんです。来てくれてありがとうございます。どうぞゆっくり話してください。"},
    {"speaker": "user",  "text": "最近、夜になると少し寂しくなるんですよね。"},
    {"speaker": "moshi", "text": "そうですか。夜は特に静かになって、気持ちが大きくなることがありますよね。"},
    {"speaker": "user",  "text": "そうなんです。誰かと話すとちょっと楽になります。"},
    {"speaker": "moshi", "text": "ここで話してくれてよかったです。急がなくて大丈夫ですよ。"},
]


def resolve_role_references(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """user / moshi それぞれの参照(wav + text)を解決する。

    --*-ref-text は解決結果の書き起こしを上書きする。参照の書き起こしは whisper
    の出力なので誤りが混じることがあり、in-context クローンではそれがそのまま
    条件付けの汚れになる。timeline.jsonl を直接書き換えると diarization をやり
    直した瞬間に消え、何で回したか追えなくなるので、上書きはここで受ける
    (manifest.json に overridden として残る)。
    """
    refs: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        clone_out = getattr(args, f"clone_out_dir_{role}")
        ref_wav = getattr(args, f"{role}_ref_wav")
        ref_text = getattr(args, f"{role}_ref_text")
        if ref_wav:
            if not Path(ref_wav).is_file():
                raise SystemExit(f"--{role}-ref-wav が存在しません: {ref_wav}")
            info = {
                "source": "explicit", "wav": str(ref_wav),
                "text": (ref_text or "").strip(),
                "timeline_index": None, "duration_sec": None,
            }
        elif clone_out:
            info = resolve_from_clone_out_dir(Path(clone_out), args.rank, args.mode)
        else:
            if not args.analysis_dir:
                raise SystemExit(
                    "--analysis-dir か --clone-out-dir-user/--clone-out-dir-moshi が必要です"
                )
            speaker = getattr(args, f"{role}_speaker")
            info = resolve_from_analysis_dir(
                Path(args.analysis_dir), speaker, args.rank,
                args.min_ref_sec, args.max_ref_sec,
            )
        if ref_text is not None and not ref_wav:
            info = {**info, "text": ref_text.strip(), "text_overridden": True}
        if args.mode == "in-context" and not info["text"]:
            raise SystemExit(
                f"{role} の参照に書き起こしがありません。--mode x-vector なら不要です。"
            )
        logger.info("%s 参照: %s (%.1fs) text=%r",
                    role, info["wav"], float(info.get("duration_sec") or -1.0), info["text"])
        refs[role] = info
    if refs["user"]["wav"] == refs["moshi"]["wav"] and not args.allow_same_reference:
        raise SystemExit(
            "user と moshi が同じ参照を指しています。両ロールが同じ声になり対話に"
            "ならないので、話者の割り当てを見直してください(--allow-same-reference で強行可)。"
        )
    return refs


def load_script_file(path: Path) -> list[dict[str, str]]:
    """1 行 1 ターンの素のテキスト台本を読む。

        user: こんばんは。少し話してもいいですか。
        moshi: もちろんです。ゆっくりどうぞ。
        # 行頭 # はコメント、空行は無視

    台詞をいじるだけなら JSONL を組むより速いので、テスト用にこの形式を持つ。
    """
    turns: list[dict[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        speaker, sep, text = line.partition(":")
        speaker = speaker.strip().lower()
        if not sep or speaker not in ROLES:
            raise SystemExit(
                f"{path}:{lineno}: 行頭は 'user:' か 'moshi:' が必要です: {raw!r}"
            )
        text = text.strip()
        if not text:
            raise SystemExit(f"{path}:{lineno}: 台詞が空です")
        turns.append({"speaker": speaker, "text": text})
    if not turns:
        raise SystemExit(f"台本が空です: {path}")
    logger.info("台本: %s (%d ターン)", path, len(turns))
    return turns


def load_turns(args: argparse.Namespace) -> list[dict[str, str]]:
    """台本を {"speaker", "text"} の列で返す。"""
    if args.script_file:
        return load_script_file(Path(args.script_file))
    if not args.dialogues_jsonl:
        logger.info("内蔵台本を使用(%d ターン)", len(DEFAULT_TURNS))
        return [dict(t) for t in DEFAULT_TURNS]

    path = Path(args.dialogues_jsonl)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"対話が 1 件もありません: {path}")
    if args.dialogue_index >= len(rows):
        raise SystemExit(
            f"--dialogue-index {args.dialogue_index} は範囲外です(全 {len(rows)} 件)"
        )
    dialogue = rows[args.dialogue_index]
    turns = []
    for turn in dialogue.get("turns") or []:
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        if speaker not in ROLES:
            # silence / other / background は今回のテスト対象外なので落とす。
            logger.info("ロール %r のターンをスキップ: %.30s", speaker, text)
            continue
        turns.append({"speaker": speaker, "text": text})
    if not turns:
        raise SystemExit(
            f"user/moshi のターンがありません: {path} の {args.dialogue_index} 番目"
        )
    logger.info("台本: %s (id=%s, %d ターン)", path, dialogue.get("id"), len(turns))
    return turns


def synthesize_turns(
    model, turns: list[dict[str, str]], refs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[Any], int]:
    """ターン列を合成する。ロールごとにまとめて投げ、元の順序へ戻す。

    参照プロンプトの作成はロールにつき 1 回でよく、同じロールのターンは 1 つの
    プロンプトを共有できる。ターン順に 1 本ずつ投げるとプロンプト作成と
    バッチ立ち上げを毎回やり直すことになるので、ロール単位でまとめる。
    """
    x_only = args.mode == "x-vector"
    wavs: list[Any] = [None] * len(turns)
    sample_rate = 0

    for role in ROLES:
        indices = [i for i, t in enumerate(turns) if t["speaker"] == role]
        if not indices:
            continue
        ref = refs[role]
        logger.info("=== %s: %d ターンを合成 ===", role, len(indices))
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(ref["wav"]),
            ref_text=ref["text"],
            x_vector_only_mode=x_only,
        )
        texts = [turns[i]["text"] for i in indices]
        role_wavs, sr = generate_clone_batched(model, texts, prompt_items, args)
        if len(role_wavs) != len(indices):
            raise SystemExit(
                f"{role}: {len(indices)} ターン投げて {len(role_wavs)} 本しか返りません"
            )
        sample_rate = int(sr) or sample_rate
        for slot, wav in zip(indices, role_wavs):
            wavs[slot] = wav

    if sample_rate <= 0:
        raise SystemExit("サンプリングレートが取得できませんでした")
    return wavs, sample_rate


def assemble_dialogue(
    wavs: list[Any], turns: list[dict[str, str]], sample_rate: int,
    lead_in_sec: float, gap_sec: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """ターン波形を無音で繋いで 1 本にし、各ターンの位置を返す。

    タイムラインには秒だけでなくサンプル位置も入れる。ステレオ版はこの位置を
    使ってモノラル版と完全に同じタイミングへ置くので、秒に丸めた値から再計算
    すると 1 サンプルずれる。
    """
    import numpy as np

    gap = np.zeros(int(sample_rate * max(0.0, gap_sec)), dtype=np.float32)
    pieces: list[Any] = []
    timeline: list[dict[str, Any]] = []
    cursor = 0  # サンプル数

    if lead_in_sec > 0:
        lead = np.zeros(int(sample_rate * lead_in_sec), dtype=np.float32)
        pieces.append(lead)
        cursor += len(lead)

    for i, (wav, turn) in enumerate(zip(wavs, turns)):
        audio = np.asarray(wav, dtype=np.float32).squeeze()
        start = cursor
        pieces.append(audio)
        cursor += len(audio)
        timeline.append({
            "index": i,
            "speaker": turn["speaker"],
            "text": turn["text"],
            "start": round(start / sample_rate, 3),
            "end": round(cursor / sample_rate, 3),
            "start_sample": start,
            "num_samples": len(audio),
        })
        if i < len(wavs) - 1 and len(gap):
            pieces.append(gap)
            cursor += len(gap)

    return np.concatenate(pieces), timeline


def assemble_stereo(
    wavs: list[Any], timeline: list[dict[str, Any]], total_samples: int,
    left_role: str,
) -> Any:
    """ロールごとに L/R へ振り分けたステレオ版を作る。

    analyze_real_dialogue.py の stereo_diarized.wav と同じ発想で、どちらの声が
    どちらのロールかを聴き分けやすくするためのもの。タイミングはモノラル版と
    同一(ターンは元々重ならないので、L と R が交互に鳴る)。
    """
    import numpy as np

    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    for wav, entry in zip(wavs, timeline):
        audio = np.asarray(wav, dtype=np.float32).squeeze()
        channel = 0 if entry["speaker"] == left_role else 1
        start = entry["start_sample"]
        # 重畳は現状ないが、将来重ねる場合に切り捨てず混ざるよう加算する。
        stereo[start:start + len(audio), channel] += audio
    return stereo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, type=Path)

    src = parser.add_argument_group("参照の解決元")
    src.add_argument("--analysis-dir", default=None,
                     help="analyze_real_dialogue.py の出力 <wav_stem>/")
    src.add_argument("--clone-out-dir-user", default=None,
                     help="user ロールに使う clone_voice_examples.py の出力ディレクトリ")
    src.add_argument("--clone-out-dir-moshi", default=None,
                     help="moshi ロールに使う clone_voice_examples.py の出力ディレクトリ")
    src.add_argument("--user-speaker", default="A",
                     help="--analysis-dir 使用時、user ロールに割り当てる話者(既定 A)")
    src.add_argument("--moshi-speaker", default="B",
                     help="--analysis-dir 使用時、moshi ロールに割り当てる話者(既定 B)")
    src.add_argument("--rank", type=int, default=0, help="参照の順位。0 = ref00")
    src.add_argument("--mode", default="in-context", choices=["in-context", "x-vector"])
    src.add_argument("--min-ref-sec", type=float, default=3.0)
    src.add_argument("--max-ref-sec", type=float, default=12.0)
    src.add_argument("--allow-same-reference", action="store_true",
                     help="両ロールが同じ参照でも続行する(声が同一になる)")
    for role in ROLES:
        src.add_argument(f"--{role}-ref-wav", default=None,
                         help=f"{role} の参照 WAV を明示指定(自動選別を使わない)")
        src.add_argument(f"--{role}-ref-text", default=None,
                         help=f"{role} の参照の書き起こしを上書き(whisper の誤りを直す用)")

    script = parser.add_argument_group("台本")
    script.add_argument("--script-file", default=None,
                        help="1 行 1 ターンの素のテキスト台本('user: ...' / 'moshi: ...')")
    script.add_argument("--dialogues-jsonl", default=None,
                        help="対話 JSONL。--script-file も未指定なら内蔵台本。")
    script.add_argument("--dialogue-index", type=int, default=0,
                        help="JSONL の何番目の対話を使うか(既定 0)")

    gen = parser.add_argument_group("生成")
    gen.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    gen.add_argument("--language", default="Japanese")
    gen.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    gen.add_argument("--dtype", default="float16",
                     choices=["float16", "bfloat16", "float32"])
    gen.add_argument("--attn-impl", default="default")
    gen.add_argument("--gen-batch-size", type=int, default=4)
    gen.add_argument("--max-new-tokens", type=int, default=4096)
    gen.add_argument("--gap-sec", type=float, default=0.4, help="ターン間の無音(秒)")
    gen.add_argument("--lead-in-sec", type=float, default=0.3)
    gen.add_argument("--stereo-left", default="user", choices=list(ROLES),
                     help="dialogue_stereo.wav の L チャンネルに置くロール(既定 user)")

    args = parser.parse_args()
    if args.rank < 0:
        parser.error("--rank must be >= 0")
    if "Base" not in args.model:
        parser.error(f"ボイスクローンには Base モデルが必要です: {args.model}")

    import soundfile as sf

    refs = resolve_role_references(args)
    turns = load_turns(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_qwen_model(args, args.model)
    try:
        wavs, sample_rate = synthesize_turns(model, turns, refs, args)
    finally:
        free_model(model)

    turn_dir = args.out_dir / "turns"
    turn_dir.mkdir(exist_ok=True)
    turn_files: list[str] = []
    for i, (wav, turn) in enumerate(zip(wavs, turns)):
        name = f"{i:02d}_{turn['speaker']}.wav"
        sf.write(str(turn_dir / name), wav, sample_rate)
        turn_files.append(name)

    dialogue, timeline = assemble_dialogue(
        wavs, turns, sample_rate, args.lead_in_sec, args.gap_sec)
    dialogue_path = args.out_dir / "dialogue.wav"
    sf.write(str(dialogue_path), dialogue, sample_rate)

    right_role = ROLES[1] if args.stereo_left == ROLES[0] else ROLES[0]
    stereo = assemble_stereo(wavs, timeline, len(dialogue), args.stereo_left)
    stereo_path = args.out_dir / "dialogue_stereo.wav"
    sf.write(str(stereo_path), stereo, sample_rate)

    manifest = {
        "model": args.model,
        "mode": args.mode,
        "rank": args.rank,
        "language": args.language,
        "sample_rate": sample_rate,
        "gap_sec": args.gap_sec,
        "lead_in_sec": args.lead_in_sec,
        "duration_sec": round(len(dialogue) / sample_rate, 2),
        "stereo": {"left": args.stereo_left, "right": right_role},
        "references": refs,
        "turns": timeline,
        "turn_files": turn_files,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("完了: %s (%.1f 秒, %d ターン)",
                dialogue_path, manifest["duration_sec"], len(turns))
    print(f"dialogue: {dialogue_path}")
    print(f"stereo:   {stereo_path} (L={args.stereo_left} / R={right_role})")
    print(f"manifest: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
