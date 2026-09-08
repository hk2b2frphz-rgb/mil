#!/usr/bin/env python3
"""参照が発話を変えるかを確かめるための、英語の質問音声と sidecar を作る。

これが確かめたい一点:
    注入した参照テキストを moshi が実際に読んでいるのか、それとも重みが
    覚えている値を喋っているだけなのか。

やり方:
    同じ質問音声に対して、参照だけを差し替えた入力ディレクトリを3つ作る。
    moshi-rag の run_inference は WAV の隣の <stem>.json を sidecar として
    読み、--use-gt-reference が付いていれば gt_reference_text をそのまま
    注入する。検索バックエンドを一切通さずに試せる。

      true     … 知識表どおりの正しい参照
      altered  … 値だけを別の数字に差し替えた参照（表には無い値）
      empty    … 参照なし

    true と altered で答えが変わるなら、モデルは参照を読んでいる。
    どちらも同じ値を言うなら、重みの中の知識で答えており、表を差し替えても
    現場の規定値には追従しないことになる。ねじPoCの設計判断はここで決まる。
    empty は、参照が無いときに黙るのか作話するのかを見る。

使い方:
    python screw_poc/scripts/moshi_rag/build_probe_inputs.py \
        --out-dir screw_poc/artifacts/moshi_rag_probe --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import from the concrete directory: an installed package named `scripts`
# can shadow this repository's scripts/ namespace package.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_qwen3_tts_data import Qwen3TTS  # noqa: E402

KNOWLEDGE = REPO_ROOT / "screw_poc" / "knowledge" / "screw_knowledge_en.csv"

# 質問と、値を差し替えた参照。差し替え後の値は表のどこにも無い数字にする。
# 「たまたま別の行と一致した」という説明を残さないため。
PROBES: tuple[dict[str, str], ...] = (
    {
        "id": "probe_pitch_m8",
        "knowledge_id": "K05",
        "question": "What is the coarse pitch of an M8 screw?",
        "altered": "The coarse pitch of M8 is 2.3 millimeters.",
    },
    {
        "id": "probe_pitch_m10",
        "knowledge_id": "K06",
        "question": "What is the coarse pitch of an M10 screw?",
        "altered": "The coarse pitch of M10 is 3.4 millimeters.",
    },
    {
        "id": "probe_torque_m10",
        "knowledge_id": "K18",
        "question": "What is the maximum tightening torque for the SSH-M10-SD-EL?",
        "altered": "The maximum tightening torque of the NBK SSH-M10-SD-EL is 61 newton meters.",
    },
    {
        "id": "probe_torque_m8",
        "knowledge_id": "K17",
        "question": "What is the maximum tightening torque for the SSH-M8-SD-EL?",
        "altered": "The maximum tightening torque of the NBK SSH-M8-SD-EL is 39 newton meters.",
    },
    # 表に無い呼び径。参照が空のときに、モデルが作話するかどうかを見る。
    {
        "id": "probe_absent_m14",
        "knowledge_id": "",
        "question": "What is the coarse pitch of an M14 screw?",
        "altered": "",
    },
)

VARIANTS = ("true", "altered", "empty")


def load_true_references() -> dict[str, str]:
    rows = {}
    with KNOWLEDGE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = " ".join(
                part for part in (row["answer_text"].strip(), row["caution_text"].strip()) if part
            )
            rows[row["knowledge_id"].strip()] = text
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--speaker", default="Serena")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    true_refs = load_true_references()
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTS(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device=args.device,
        dtype_str=args.dtype,
        attn_impl="default",
        speaker_user=args.speaker,
        speaker_moshi="Ono_Anna",
        language="English",
        instruct_user=None,
        instruct_moshi=None,
    )
    import sphn  # type: ignore[import]

    # 音声は一度だけ合成する。3つの条件で同じ音を使わないと、答えが変わった
    # 理由が参照なのか音声の揺れなのか分からなくなる。
    for probe in PROBES:
        pcm = np.asarray(
            tts.synthesize(probe["question"], "user", speaker_override=args.speaker),
            dtype=np.float32,
        ).reshape(-1)
        sphn.write_wav(str(audio_dir / f"{probe['id']}.wav"), pcm, tts.sample_rate)

    manifest: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_dir = args.out_dir / f"inputs_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for probe in PROBES:
            probe_id = probe["id"]
            shutil.copyfile(audio_dir / f"{probe_id}.wav", variant_dir / f"{probe_id}.wav")
            if variant == "true":
                reference = true_refs.get(probe["knowledge_id"], "")
            elif variant == "altered":
                reference = probe["altered"]
            else:
                reference = ""
            sidecar = {
                "topic": probe["question"],
                "gt_reference_text": reference,
                "answer": true_refs.get(probe["knowledge_id"], ""),
            }
            (variant_dir / f"{probe_id}.json").write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            manifest.append(
                {
                    "id": probe_id,
                    "variant": variant,
                    "knowledge_id": probe["knowledge_id"],
                    "question": probe["question"],
                    "injected_reference": reference,
                }
            )

    (args.out_dir / "probe_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"probes": len(PROBES), "variants": list(VARIANTS), "out_dir": str(args.out_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
