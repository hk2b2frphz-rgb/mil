#!/usr/bin/env python3
"""Generate and cache the four Qwen3-TTS reference voices used by MOSS-TTSD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_qwen3_tts_data import MossTTSD, Qwen3TTS, write_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn-impl",
        default="default",
        choices=["default", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--language", default="Japanese")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        role: args.out_dir / f"{role}_{speaker}.wav"
        for role, speaker in MossTTSD.DEFAULT_REF_SPEAKERS.items()
    }
    missing_roles = [role for role, path in paths.items() if not path.is_file()]

    if missing_roles:
        tts = Qwen3TTS(
            model_id=args.model,
            device=args.device,
            dtype_str=args.dtype,
            attn_impl=args.attn_impl,
            speaker_user="Ono_Anna",
            speaker_moshi="Serena",
            language=args.language,
            instruct_user=None,
            instruct_moshi=None,
        )
        for role in missing_roles:
            speaker = MossTTSD.DEFAULT_REF_SPEAKERS[role]
            pcm = tts.synthesize(
                MossTTSD.DEFAULT_REF_TEXT,
                speaker_role="moshi" if role == "moshi" else "user",
                speaker_override=speaker,
            )
            write_wav(paths[role], pcm[np.newaxis, :], tts.sample_rate)
            print(f"generated {role}: {paths[role]}")
    else:
        print("all MOSS reference WAVs already exist; skipping synthesis")

    refs = {
        "roles": {
            role: {
                "path": path.name,
                "transcript": MossTTSD.DEFAULT_REF_TEXT,
                "speaker": MossTTSD.DEFAULT_REF_SPEAKERS[role],
            }
            for role, path in paths.items()
        }
    }
    refs_path = args.out_dir / "refs.json"
    with refs_path.open("w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {refs_path}")


if __name__ == "__main__":
    main()
