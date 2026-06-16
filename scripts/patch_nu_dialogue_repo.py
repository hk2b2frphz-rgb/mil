#!/usr/bin/env python3
"""Apply small runtime compatibility patches to nu-dialogue/moshi-finetune."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu-repo", required=True, type=Path)
    return parser.parse_args()


def patch_tokenize_text(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_ROBUST_TEXT_ALIGNMENT" in src:
        return False

    pattern = re.compile(
        r"    # make token-level transcript by aligning the timestamps\n"
        r"    token_transcript = \[\]\n"
        r"    for i, token in enumerate\(tokens\):\n"
        r".*?"
        r"    assert not char_transcript, f\"Remaining characters: \{char_transcript\}\"\n",
        flags=re.S,
    )
    replacement = '''    # make token-level transcript by aligning the timestamps
    # AUTO_PATCH_ROBUST_TEXT_ALIGNMENT:
    # Synthetic data in miltoka has utterance-level Japanese timestamps. Some
    # SentencePiece outputs include word-boundary markers or decoded byte pieces
    # whose string length can exceed the remaining character transcript. Clamp
    # alignment instead of crashing at chars[0].
    token_transcript = []
    for token in tokens:
        token_text = token[1:] if token.startswith("▁") else token
        if not token_text:
            continue
        if not char_transcript:
            warnings.warn(f"Dropping token {token!r}: no characters remain for {text!r}")
            continue
        num_token_chars = max(1, len(token_text))
        if num_token_chars > len(char_transcript):
            warnings.warn(
                f"Clamping token/character alignment for token {token!r}; "
                f"remaining_chars={len(char_transcript)} text={text!r}"
            )
            chars = char_transcript
        else:
            chars = char_transcript[:num_token_chars]
        if not chars:
            continue
        token_transcript.append(
            {
                "speaker": chars[0]["speaker"],
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "token": token,
            }
        )
        char_transcript = char_transcript[len(chars) :]
    if char_transcript:
        warnings.warn(f"Remaining characters after token alignment are ignored: {char_transcript}")
'''
    updated, count = pattern.subn(replacement, src, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate token alignment block in {path}")
    path.write_text(updated, encoding="utf-8")
    return True


def patch_prepare_dataset(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    updated = src.replace("np.concat(", "np.concatenate(")
    if updated == src:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def patch_finetune_tracking(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_WITH_TRACKING_DEFAULT" in src:
        return False

    old = """    if args.report_to is not None:
        args.with_tracking = True
"""
    new = """    # AUTO_PATCH_WITH_TRACKING_DEFAULT:
    # nu-dialogue/moshi-finetune references args.with_tracking even when
    # --report_to is omitted. Default it explicitly for non-W&B runs.
    args.with_tracking = False
    if args.report_to is not None:
        args.with_tracking = True
"""
    if old not in src:
        raise RuntimeError(f"Could not locate report_to tracking block in {path}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    return True


def patch_finetune_safe_means(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_SAFE_EMPTY_MEAN" in src:
        return False

    marker = 'logger = get_logger(__name__)\n'
    helper = '''logger = get_logger(__name__)


# AUTO_PATCH_SAFE_EMPTY_MEAN:
# Eval batches can contain streams with no valid text/user labels after
# utterance-level timestamp conversion and length filtering. torch.mean([]) is
# NaN, so use zero-valued tensors for logging-only empty means.
def _safe_mean(values):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()
'''
    if marker not in src:
        raise RuntimeError(f"Could not locate logger declaration in {path}")
    src = src.replace(marker, helper, 1)

    replacements = {
        'text_accuracy[non_pad_indices].mean()': '_safe_mean(text_accuracy[non_pad_indices])',
        'text_accuracy[pad_indices].mean()': '_safe_mean(text_accuracy[pad_indices])',
        'audio_accuracy[..., 0][\n            audio_labels[..., 0] != moshi_lm.zero_token_id\n        ].mean()': '_safe_mean(audio_accuracy[..., 0][\n            audio_labels[..., 0] != moshi_lm.zero_token_id\n        ])',
        'audio_accuracy[..., 1:8][\n            audio_labels[..., 1:8] != moshi_lm.zero_token_id\n        ].mean()': '_safe_mean(audio_accuracy[..., 1:8][\n            audio_labels[..., 1:8] != moshi_lm.zero_token_id\n        ])',
        'audio_accuracy[..., 8][\n                    audio_labels[..., 8] != moshi_lm.zero_token_id\n                ].mean()': '_safe_mean(audio_accuracy[..., 8][\n                    audio_labels[..., 8] != moshi_lm.zero_token_id\n                ])',
        'audio_accuracy[..., 9:][\n                    audio_labels[..., 9:] != moshi_lm.zero_token_id\n                ].mean()': '_safe_mean(audio_accuracy[..., 9:][\n                    audio_labels[..., 9:] != moshi_lm.zero_token_id\n                ])',
        '    text_loss = 0.0': '    text_loss = tempformer_out.new_tensor(0.0)',
        'temp_result["non_pad_losses"].mean().detach()': '_safe_mean(temp_result["non_pad_losses"]).detach()',
        'temp_result["pad_losses"].mean().detach()': '_safe_mean(temp_result["pad_losses"]).detach()',
        'dep_result["semantic_losses"].mean().detach()': '_safe_mean(dep_result["semantic_losses"]).detach()',
        'dep_result["acoustic_losses"].mean().detach()': '_safe_mean(dep_result["acoustic_losses"]).detach()',
        'dep_result["semantic_losses_user"].mean().detach()': '_safe_mean(dep_result["semantic_losses_user"]).detach()',
        'dep_result["acoustic_losses_user"].mean().detach()': '_safe_mean(dep_result["acoustic_losses_user"]).detach()',
    }
    missing = []
    for old, new in replacements.items():
        if old not in src:
            missing.append(old)
        else:
            src = src.replace(old, new)
    if missing:
        raise RuntimeError(f"Could not locate safe-mean targets in {path}: {missing[:2]}")

    path.write_text(src, encoding="utf-8")
    return True


def patch_utils_data(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    updated = src.replace("np.concat(", "np.concatenate(")
    if updated == src:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    nu_repo = args.nu_repo.resolve()
    if not nu_repo.exists():
        raise SystemExit(f"nu repo not found: {nu_repo}")

    changes = []
    if patch_tokenize_text(nu_repo / "tools" / "tokenize_text.py"):
        changes.append("tools/tokenize_text.py")
    if patch_prepare_dataset(nu_repo / "tools" / "prepare_dataset.py"):
        changes.append("tools/prepare_dataset.py")
    if patch_finetune_tracking(nu_repo / "finetune.py"):
        changes.append("finetune.py")
    if patch_finetune_safe_means(nu_repo / "finetune.py"):
        changes.append("finetune.py:safe-means")
    if patch_utils_data(nu_repo / "utils" / "data.py"):
        changes.append("utils/data.py")

    if changes:
        print(f"[nu-patch] patched {', '.join(changes)}")
    else:
        print("[nu-patch] already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
