#!/usr/bin/env python3
"""ボイスクローンが本当に参照音声を使っているかを実測で確かめる。

「クローンできているつもりで、実は参照が効いていない」は音を聴くだけでは
判別しにくい。参照が無視されていてもモデルは日本語を自然に喋るので、それらしく
聞こえてしまうため。ここでは同じ文を参照 A と参照 B の 2 通りで合成し、
出力が参照ごとに変わっているかを機械的に見る。

判定は 2 段階:

  1. 出力 A と出力 B が別物か
     参照が無視されていれば、同じ文・同じモデルなので出力はほぼ同一になる。
     ここが「ほぼ同一」なら、クローンは効いていない。これが決定的な判定。

  2. 各出力が自分の参照側に寄っているか
     MFCC 平均のコサイン類似度という粗い指標での確認。話者照合モデルではない
     ので絶対値に意味はなく、A/B どちらに寄ったかの相対比較のみを見る。

使い方:
    uv run python scripts/verify_clone_reference.py \
        --ref-a <話者Aの参照>.wav --ref-text-a "その書き起こし" \
        --ref-b <話者Bの参照>.wav --ref-text-b "その書き起こし" \
        --out-dir data/clone_verify/test01

参照は resolve_clone_refs.py で取れる:
    uv run python scripts/resolve_clone_refs.py --analysis-dir <dir> --speaker A
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
    from scripts.clone_voice_examples import free_model, load_qwen_model
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from clone_voice_examples import free_model, load_qwen_model  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_SR = 16000
# 出力が「ほぼ同一」とみなす閾値。参照が無視されていれば同じ文・同じモデルの
# 生成なので、ここを超える一致になる。サンプリングのゆらぎで完全一致には
# ならないため 1.0 ではなくこの値で切る。
IDENTICAL_THRESHOLD = 0.995


def load_mono(path: Path) -> Any:
    """WAV を 16kHz モノラルの float32 で読む。"""
    import numpy as np
    import soundfile as sf
    import torchaudio

    audio, sr = sf.read(str(path), dtype="float32")
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != ANALYSIS_SR:
        import torch
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, ANALYSIS_SR
        ).numpy()
    return audio


def voice_fingerprint(audio: Any) -> Any:
    """MFCC の時間平均。話者照合モデルではないので相対比較にのみ使う。"""
    import torch
    import torchaudio

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=ANALYSIS_SR,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
    )
    with torch.no_grad():
        coeffs = mfcc(torch.from_numpy(audio).unsqueeze(0))
    return coeffs.squeeze(0).mean(dim=1)


def cosine(a: Any, b: Any) -> float:
    import torch
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def waveform_similarity(a: Any, b: Any) -> float:
    """2 本の生成波形がどれだけ同一かを見る（参照が効いていないと 1 に近づく）。"""
    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    x, y = a[:n], b[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return abs(float(np.dot(x, y)) / denom)


def synthesize(model, text: str, ref_wav: str, ref_text: str | None,
               args: argparse.Namespace) -> tuple[Any, int]:
    import numpy as np
    import torch

    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_wav),
        ref_text=ref_text,
        x_vector_only_mode=(args.mode == "x-vector"),
    )
    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=[text],
            language=[args.language],
            voice_clone_prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
    wav = wavs[0]
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    return np.asarray(wav, dtype="float32").squeeze(), int(sr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-text-a", default=None)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--ref-text-b", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--text", default="もしもし、こちら孤独孤立相談窓口になります。",
                        help="両参照で合成する共通の文")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--mode", default="in-context", choices=["in-context", "x-vector"])
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-impl", default="default")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.mode == "in-context" and not (args.ref_text_a and args.ref_text_b):
        parser.error("in-context モードでは --ref-text-a / --ref-text-b が必要です")
    for path in (args.ref_a, args.ref_b):
        if not Path(path).is_file():
            parser.error(f"参照 WAV が見つかりません: {path}")

    import soundfile as sf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_qwen_model(args, args.model)
    try:
        logger.info("参照 A で合成: %s", args.ref_a)
        out_a, sr = synthesize(model, args.text, args.ref_a, args.ref_text_a, args)
        logger.info("参照 B で合成: %s", args.ref_b)
        out_b, _ = synthesize(model, args.text, args.ref_b, args.ref_text_b, args)
    finally:
        free_model(model)

    sf.write(str(args.out_dir / "out_ref_a.wav"), out_a, sr)
    sf.write(str(args.out_dir / "out_ref_b.wav"), out_b, sr)

    ref_a, ref_b = load_mono(Path(args.ref_a)), load_mono(Path(args.ref_b))
    gen_a, gen_b = load_mono(args.out_dir / "out_ref_a.wav"), load_mono(args.out_dir / "out_ref_b.wav")
    fp = {k: voice_fingerprint(v) for k, v in
          {"ref_a": ref_a, "ref_b": ref_b, "gen_a": gen_a, "gen_b": gen_b}.items()}

    same_output = waveform_similarity(gen_a, gen_b)
    result = {
        "text": args.text,
        "mode": args.mode,
        "ref_a": args.ref_a,
        "ref_b": args.ref_b,
        "output_waveform_similarity": round(same_output, 4),
        "gen_a_vs_ref_a": round(cosine(fp["gen_a"], fp["ref_a"]), 4),
        "gen_a_vs_ref_b": round(cosine(fp["gen_a"], fp["ref_b"]), 4),
        "gen_b_vs_ref_b": round(cosine(fp["gen_b"], fp["ref_b"]), 4),
        "gen_b_vs_ref_a": round(cosine(fp["gen_b"], fp["ref_a"]), 4),
    }

    # 判定1: 参照を変えて出力が変わったか。ここが本質。
    reference_has_effect = same_output < IDENTICAL_THRESHOLD
    # 判定2: それぞれ自分の参照側に寄ったか（粗い指標なので参考値）。
    a_leans_right = result["gen_a_vs_ref_a"] > result["gen_a_vs_ref_b"]
    b_leans_right = result["gen_b_vs_ref_b"] > result["gen_b_vs_ref_a"]
    result["reference_has_effect"] = reference_has_effect
    result["leans_to_own_reference"] = {"a": a_leans_right, "b": b_leans_right}

    (args.out_dir / "verify.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    if not reference_has_effect:
        print("判定: 参照が効いていません。2 つの参照で出力がほぼ同一です "
              f"(波形類似度 {same_output:.4f} >= {IDENTICAL_THRESHOLD})。")
        raise SystemExit(1)
    print(f"判定: 参照は効いています (波形類似度 {same_output:.4f} < {IDENTICAL_THRESHOLD})。")
    if a_leans_right and b_leans_right:
        print("      各出力とも自分の参照側に寄っています。")
    else:
        print("      ただし MFCC 上はどちらかが自分の参照側に寄っていません。"
              "粗い指標なので断定はできませんが、out_ref_a.wav と out_ref_b.wav を"
              "聴いて確認してください。")


if __name__ == "__main__":
    main()
