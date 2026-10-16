#!/usr/bin/env python3
"""KABURI 環境で torchaudio の読み書きが通るかだけを確かめる。

torchaudio 2.9 以降は load/save の実体が torchcodec に移っている。torchcodec の
wheel は CUDA のメジャー版ごとに別物で、PyPI の既定が torch と噛み合わないと
こうなる:

    OSError: libnvrtc.so.13: cannot open shared object file

これは import の時点では出ず、最初に wav を読もうとしたところで出る。参照パック
作りが最初の犠牲者になるので、モデルを 5GB 落とす前にここで弾く。

戻り値: 0 = 読み書きできた / 1 = できなかった（理由を stderr に出す）。
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path


def main() -> int:
    import torch
    import torchaudio

    cuda_version = getattr(torch.version, "cuda", None)
    print(f"torch={torch.__version__} cuda={cuda_version} torchaudio={torchaudio.__version__}")
    try:
        import torchcodec

        print(f"torchcodec={getattr(torchcodec, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"torchcodec: import failed ({exc})", file=sys.stderr)

    wave = torch.zeros(2, 1600)
    wave[0, ::100] = 0.5
    with tempfile.TemporaryDirectory(prefix="kaburi_io_") as work:
        path = Path(work) / "probe.wav"
        try:
            torchaudio.save(str(path), wave, 16000, channels_first=True)
            loaded, rate = torchaudio.load(str(path))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("", file=sys.stderr)
            print("torchaudio cannot read or write audio in this environment.", file=sys.stderr)
            print(
                "If the traceback above mentions libnvrtc.so.<N>, torchcodec was "
                "built for a different CUDA major version than torch "
                f"(torch reports cuda={cuda_version}).",
                file=sys.stderr,
            )
            return 1
        if loaded.shape[0] != 2 or int(rate) != 16000:
            print(
                f"unexpected round trip: shape={tuple(loaded.shape)} rate={rate}",
                file=sys.stderr,
            )
            return 1

    print("torchaudio read/write ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
