"""Download only the official V2 converter; retain model-card/license metadata."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    root = Path(__file__).resolve().parent
    repo = "myshell-ai/OpenVoiceV2"
    revision = HfApi().model_info(repo).sha
    destination = root / "models" / "OpenVoiceV2"
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(destination),
        allow_patterns=["converter/*", "README.md", "LICENSE*"],
    )
    for name in ("config.json", "checkpoint.pth"):
        if not (destination / "converter" / name).is_file():
            raise RuntimeError("Missing converter file: " + name)
    (root / "model-revision.json").write_text(
        json.dumps({"repo": repo, "revision": revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Model ready:", destination)


if __name__ == "__main__":
    main()
