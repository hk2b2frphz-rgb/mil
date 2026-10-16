"""Whole-file reference-conditioned voice conversion, without TTS or ASR."""
import argparse
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-reference", type=Path,
                        help="Optional clean sample of the source speaker")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.check and any(v is None for v in (args.source, args.reference, args.output)):
        parser.error("--source, --reference and --output are required")

    import torch
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check driver and CUDA-enabled PyTorch")
        capability = torch.cuda.get_device_capability(args.device)
        arch = "sm_{}{}".format(*capability)
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError("PyTorch wheel lacks " + arch + "; use setup.sh")
        print("GPU:", torch.cuda.get_device_name(args.device), arch)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "dtype: float32")
    from openvoice.api import ToneColorConverter
    model_dir = Path(__file__).resolve().parent / "models" / "OpenVoiceV2" / "converter"
    for name in ("config.json", "checkpoint.pth"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError("Run download_model.py first: " + str(model_dir / name))
    # No auxiliary watermark model is loaded. This is not a license requirement.
    converter = ToneColorConverter(str(model_dir / "config.json"),
                                   device=args.device, enable_watermark=False)
    converter.load_ckpt(str(model_dir / "checkpoint.pth"))
    converter.model.float().eval()
    if args.check:
        print("Converter loaded. Audio conversion has not been tested by --check.")
        return

    import librosa
    import numpy as np
    import soundfile as sf
    torch.manual_seed(args.seed)
    sr = converter.hps.data.sampling_rate

    def prepare(path, destination):
        if not path.is_file():
            raise FileNotFoundError(path)
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
        if len(audio) < sr or not np.isfinite(audio).all():
            raise ValueError("Need at least one second of finite audio: " + str(path))
        if np.max(np.abs(audio)) < 1e-5:
            raise ValueError("Silent input: " + str(path))
        sf.write(str(destination), audio, sr, subtype="FLOAT")
        return str(destination)

    if args.output.resolve() in {p.resolve() for p in
            (args.source, args.reference, args.source_reference) if p is not None}:
        raise ValueError("Output must not overwrite an input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openvoice-vc-") as temporary:
        work = Path(temporary)
        source = prepare(args.source, work / "source.wav")
        target = prepare(args.reference, work / "reference.wav")
        speaker_source = (prepare(args.source_reference, work / "source_ref.wav")
                          if args.source_reference else source)
        with torch.inference_mode():
            src_se = converter.extract_se(speaker_source)
            tgt_se = converter.extract_se(target)
            converted = converter.convert(source, src_se, tgt_se, output_path=None)
        if not np.isfinite(converted).all():
            raise RuntimeError("Non-finite model output")
        sf.write(str(args.output), converted, sr, subtype="PCM_16")
    print("Saved:", args.output, "sample rate:", sr)


if __name__ == "__main__":
    main()
