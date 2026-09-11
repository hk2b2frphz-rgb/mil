#!/usr/bin/env python3
"""KABURI-TTS で対話 JSONL を Moshi fine-tune 形式のステレオ学習データにする。

generate_qwen3_tts_data.py の KABURI 版。出力ディレクトリ構造と JSON の形は
そちらと同じなので、merge_training_shards.py 以降の下流はそのまま使える。

  out_dir/
    synthetic_moshi_train.jsonl   ← Moshi fine-tune manifest
    dialogues.jsonl               ← 対話スクリプト
    data_stereo/
      <stem>.wav                  ← ステレオWAV (左=moshi/相談員, 右=user/相談者)
      <stem>.json                 ← アライメント / メタデータ

Qwen3-TTS / Kokoro 版との決定的な違い:

* 発話ごとに合成して並べるのではなく、2 話者を左右チャンネルに**同時に**
  レンダリングする。かぶり（相槌の重なり）と間は KABURI の gap model が
  決めるので、--lead-in-sec / --gap-sec / --auto-overlap-aizuchi に相当する
  配置ロジックは無い。
* 発話境界は合成後の forced alignment ではなく、入力した音素ラスタから
  そのまま取れる。MMS_FA を通さないので CTC 失敗による対話の取りこぼしが無い。
* KABURI のキャンバスは 30 秒固定。傾聴対話は 2-3 分あるので、対話を 30 秒
  以内のチャンクに切って合成し、時間方向に連結する（チャンク境界では
  韻律が途切れる）。詰め込みすぎるとラスタが早口方向に圧縮されるため、
  fit_scale が --min-fit-scale を下回るチャンクは分割し直す。

使い方:
  uv run --project ../kaburi-tts python scripts/generate_kaburi_tts_data.py \
      --dialogues-jsonl data/runs/<batch>/shard_000/dialogues.jsonl \
      --out-dir data/runs/<batch>/shard_000/training_set \
      --ref-pack ../kaburi-tts/assets/test_refpack \
      --device cuda --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alignment_words import (  # noqa: E402
    get_segmenter_name,
    split_utterance_alignments,
)

logger = logging.getLogger("kaburi_tts_data")

# KABURI のキャンバス: 25 fps x 750 frames = 30 秒。チャンク分割の見積もりと
# 下流のメタデータで使うので、ここにも定数として持つ（実際の値は
# kaburi_tts.raster.pipeline から読み直して照合する）。
FPS = 25
CANVAS_FRAMES = 750

# チャンク分割の初期見積もり。日本語の発話速度 ~7.5 文字/秒に、発話あたりの
# 間 0.4 秒を足した粗い上限。ここで切ったあと fit_scale で検算する。
CHARS_PER_SEC = 7.5
GAP_ESTIMATE_SEC = 0.4

# 左=moshi(傾聴役)=チャンネル A、右=user=チャンネル B。
CHANNEL_OF_SPEAKER = {"moshi": "A", "user": "B"}
LABEL_OF_SPEAKER = {"moshi": "SPEAKER_MAIN", "user": "SPEAKER_USER"}


# ---------------------------------------------------------------------------
# 入力


def load_dialogues(path: Path) -> list[dict[str, Any]]:
    """dialogues.jsonl を読み、合成対象のターンだけ残した dict を返す。

    speaker=="silence" のターンは落とす。KABURI では間も gap model が作るので、
    明示的な無音ターンを差し込む先が無い（normal v6 は無音ゼロ設定なので実害は
    無いが、無音入りのコーパスを渡されたときに黙って壊れないようにする）。
    """
    out: list[dict[str, Any]] = []
    dropped_silence = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns: list[dict[str, str]] = []
            for turn in row.get("turns") or []:
                speaker = str(turn.get("speaker", "")).strip().lower()
                if speaker == "silence":
                    dropped_silence += 1
                    continue
                if speaker not in CHANNEL_OF_SPEAKER:
                    continue
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                # tts_text は「学習用原文とは別の読み」。KABURI にも合成側だけ
                # 差し替えて渡す。
                tts_text = str(turn.get("tts_text") or "").strip() or text
                turns.append({"speaker": speaker, "text": text, "tts_text": tts_text})
            if not turns:
                continue
            out.append({**row, "turns": turns})
    if dropped_silence:
        logger.warning(
            "speaker=silence のターンを %d 件落としました（KABURI は間を gap model で作る）",
            dropped_silence,
        )
    return out


def safe_stem(text: str, fallback: str) -> str:
    """generate_qwen3_tts_data.safe_stem と同じ規則でファイル名を作る。"""
    import re

    stem = re.sub(r"\s+", "_", text.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)
    stem = re.sub(r"[^0-9A-Za-z_.\-぀-ヿ㐀-鿿]+", "_", stem)
    return stem[:60].strip("._-") or fallback


# ---------------------------------------------------------------------------
# チャンク分割


def estimate_seconds(turns: list[dict[str, str]]) -> float:
    """チャンクの所要秒数の粗い見積もり（文字数 / 速度 + 発話ごとの間）。"""
    chars = sum(len(turn["tts_text"]) for turn in turns)
    return chars / CHARS_PER_SEC + GAP_ESTIMATE_SEC * len(turns)


def split_turns_by_estimate(
    turns: list[dict[str, str]],
    chunk_sec: float,
    max_utts: int,
) -> list[list[dict[str, str]]]:
    """見積もり秒数と発話数上限だけでチャンクに区切る（fit 検算はまだしない）。

    1 発話だけで chunk_sec を超える場合でも、その発話は単独チャンクとして残す。
    そこから先はラスタ側の圧縮に委ねるしかない（分割できる単位が無い）。
    """
    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for turn in turns:
        candidate = current + [turn]
        if current and (
            len(candidate) > max_utts or estimate_seconds(candidate) > chunk_sec
        ):
            chunks.append(current)
            current = [turn]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_in_half(turns: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    mid = max(1, len(turns) // 2)
    return turns[:mid], turns[mid:]


# ---------------------------------------------------------------------------
# KABURI 本体


class KaburiRenderer:
    """acoustic model / codec / ラスタ生成器を 1 回だけロードして使い回す。"""

    def __init__(self, args: argparse.Namespace) -> None:
        kaburi_repo = args.kaburi_repo.resolve()
        if not (kaburi_repo / "kaburi_tts").is_dir():
            raise SystemExit(f"KABURI-TTS のリポジトリが見つかりません: {kaburi_repo}")
        if str(kaburi_repo) not in sys.path:
            sys.path.insert(0, str(kaburi_repo))

        import torch
        import yaml
        from irodori_tts.codec import DACVAECodec
        from irodori_tts.tokenizer import PretrainedTextTokenizer
        from kaburi_tts.acoustic.infer import build_model, load_acoustic_checkpoint
        from kaburi_tts.raster import RasterGenerator
        from kaburi_tts.raster.pipeline import FIT_MARGIN, T as RASTER_T
        from kaburi_tts.raster.models import MAX_UTTS
        from kaburi_tts.two_stream.dataset import Normal2StreamDataset
        from kaburi_tts.two_stream.loss import build_phone_class_table_8

        if int(RASTER_T) != CANVAS_FRAMES:
            raise SystemExit(
                "KABURI のキャンバス長が想定と違います "
                f"(kaburi_tts.raster.pipeline.T={RASTER_T}, 想定={CANVAS_FRAMES})。"
                "--chunk-sec の意味が変わるので、スクリプト側を更新してください。"
            )

        self.torch = torch
        self.max_utts = int(MAX_UTTS)
        self.fit_margin = int(FIT_MARGIN)
        self.device = torch.device(args.device)

        config_path = kaburi_repo / "configs/acoustic.yaml"
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data_config = self.config["data"]
        self.boundary_radius = int(
            self.config.get("phone_condition", {}).get("boundary_radius", 5)
        )

        manifest = Path(args.ref_pack) / "manifest.jsonl"
        if not manifest.is_file():
            raise SystemExit(
                f"ref pack に manifest.jsonl がありません: {args.ref_pack}"
                "（kaburi-tts の scripts/make_ref_pack.py で作ります）"
            )
        ref_row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
        self.speaker_a = str(ref_row["speaker_A"])
        self.speaker_b = str(ref_row["speaker_B"])
        self.template_chunk = str(ref_row["chunk_id"])

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            "llm-jp/llm-jp-3-150m", local_files_only=False
        )
        self.dataset = Normal2StreamDataset(
            str(manifest),
            text_tokenizer=tokenizer,
            max_text_len=int(data_config.get("max_text_len", 64)),
            latent_T=CANVAS_FRAMES,
            speaker_balanced=False,
            soft_phone=(
                str(self.config.get("phone_condition", {}).get("mode", "hard")) == "soft"
            ),
            soft_boundary_radius=self.boundary_radius,
        )
        self.acoustic = build_model(self.config, self.device)
        load_acoustic_checkpoint(self.acoustic, args.acoustic_ckpt, self.device)
        self.pct8 = build_phone_class_table_8(data_config["phone_vocab_path"]).to(self.device)
        self.codec = DACVAECodec.load(device=str(self.device), dtype=torch.bfloat16)
        self.sample_rate = int(self.codec.sample_rate)

        # タイミングモデルは CPU で足りる（合成の主コストは acoustic 側）。
        self.timing = RasterGenerator(
            device="cpu",
            realizer_ckpt=args.realizer_ckpt,
            gap_ckpt=args.gap_ckpt,
            release=args.raster_release,
        )
        gate_cfg = dict(self.timing.decode_cfg.get("activity_gate", {}))
        self.gate_on = bool(gate_cfg.get("enabled", True)) and not args.no_activity_gate
        self.gate_cfg = gate_cfg
        self.cfg_scale = float(args.cfg_scale)
        self.num_steps = int(args.num_steps)
        self.seed = int(args.seed)

        logger.info(
            "KABURI ready: release=%s realizer=%s activity_gate=%s speakers=%s/%s sr=%d",
            self.timing.release,
            type(self.timing.realizer).__name__,
            "on" if self.gate_on else "off",
            self.speaker_a,
            self.speaker_b,
            self.sample_rate,
        )

    # -- ラスタ ------------------------------------------------------------

    def timeline(self, turns: list[dict[str, str]], chunk_id: str):
        """RasterGenerator.timeline と同じ配置を、入力ターンとの対応付き で返す。

        kaburi_tts.raster.pipeline.RasterGenerator.timeline をそのまま使うと
        realizer が落とした発話の情報が消えて、placed[i] がどのターンなのか
        分からなくなる（alignments が書けない）。配置そのものは上流と同じ式:
        start_i = max(0, prev_any_end + gap_i)、その後 30 秒キャンバスへの
        正比例圧縮。上流が変わったらここも合わせること。
        """
        import numpy as np

        pairs = [(CHANNEL_OF_SPEAKER[t["speaker"]], t["tts_text"]) for t in turns]
        realized = self.timing.realizer.realize_chunk(pairs, chunk_id=chunk_id)
        kept = [(i, r) for i, r in enumerate(realized) if r is not None]
        if not kept:
            raise ValueError("有効な発話がありません (G2P 失敗 or 音素長超過)")
        if len(kept) > self.max_utts:
            raise ValueError(f"発話数が上限 {self.max_utts} を超えています: {len(kept)}")

        indices = [i for i, _ in kept]
        realized_utts = [r for _, r in kept]
        speakers = [pairs[i][0] for i in indices]
        gaps = self.timing._predict_gaps(realized_utts, speakers)

        placed = []
        prev_any_end = None
        for idx, (phones, durs), spk, gap in zip(indices, realized_utts, speakers, gaps):
            start = max(0.0, gap) if prev_any_end is None else max(0.0, prev_any_end + gap)
            end = start + sum(durs)
            prev_any_end = end if prev_any_end is None else max(prev_any_end, end)
            placed.append(
                dict(turn_index=idx, speaker=spk, start=start, phones=phones, durs=durs)
            )

        target = CANVAS_FRAMES - self.fit_margin
        max_end = max(u["start"] + sum(u["durs"]) for u in placed)
        scale = 1.0
        if max_end > target:
            for candidate in np.linspace(1.0, 0.2, 81):
                if all(u["start"] * candidate + sum(u["durs"]) <= target for u in placed):
                    scale = float(candidate)
                    break
            for u in placed:
                u["start"] *= scale
        return placed, {"fit_scale": scale, "n_utts": len(placed)}

    # -- 合成 --------------------------------------------------------------

    def render_chunk(self, placed: list[dict[str, Any]], seed: int):
        """配置済み発話 -> (2ch 波形, 有効フレーム数, overlap 率)。"""
        import torch
        import kaburi_tts.placement.baselines as SPB
        from kaburi_tts.acoustic.infer import synth
        from kaburi_tts.raster import RasterGenerator
        from kaburi_tts.two_stream.dataset import collate_eventmix

        phone_a, phone_b, act_a, act_b = RasterGenerator.rasterize(placed)

        aidx = SPB._find_index(self.dataset, self.template_chunk)
        if aidx is None:
            raise RuntimeError(f"template chunk が manifest にありません: {self.template_chunk}")
        batch = collate_eventmix([self.dataset[aidx]])
        batch["latent_mask"] = torch.ones_like(batch["latent_mask"], dtype=torch.bool)
        row = self.dataset.entries[aidx]
        template_a = str(row.get("speaker_A") or row.get("spk_A") or "")
        template_b = str(row.get("speaker_B") or row.get("spk_B") or "")
        if self.speaker_a == template_b and self.speaker_b == template_a:
            batch = SPB._swap_channel_refs(batch)
        batch = SPB._replace_batch_raster(
            dict(batch), phone_a, phone_b, act_a, act_b, self.boundary_radius
        )
        batch = SPB._move_batch(batch, self.device)

        wav = synth(
            self.acoustic,
            self.codec,
            batch,
            device=self.device,
            num_steps=self.num_steps,
            seed=seed,
            cfg_scale=self.cfg_scale,
            pct8=self.pct8,
        ).cpu()

        # 末尾の無発話を落とす（+8 フレームは減衰の余韻ぶん、上流と同じ）。
        act = torch.maximum(act_a, act_b)
        nonzero = torch.nonzero(act > 0)
        active_frames = int(nonzero[-1]) + 1 if len(nonzero) else 0
        if active_frames:
            cut = int((active_frames + 8) * self.sample_rate / FPS)
            if 0 < cut < wav.shape[-1]:
                wav = wav[:, :cut]
        if self.gate_on:
            from kaburi_tts.raster.activity_gate import gate_channels_first

            wav, _env = gate_channels_first(
                wav,
                torch.stack([act_a, act_b]),
                self.sample_rate,
                fps=int(self.gate_cfg.get("fps", FPS)),
                pad_frames=int(self.gate_cfg.get("pad_frames", 2)),
                fade_ms=float(self.gate_cfg.get("fade_ms", 40.0)),
            )
        overlap = float(((act_a > 0.5) & (act_b > 0.5)).float().mean())
        return wav, active_frames, overlap

    # -- 対話 1 本 ---------------------------------------------------------

    def plan_chunks(
        self,
        turns: list[dict[str, str]],
        dialogue_id: str,
        chunk_sec: float,
        min_fit_scale: float,
    ) -> list[tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]]:
        """チャンクに切り、fit_scale が閾値を下回るものは分割し直す。

        fit_scale < 1 は「30 秒に収めるために発話開始を前へ詰めた」ことを意味
        する。極端に詰めると相槌が本来と違う位置に刺さるので、分割して
        やり直す。1 発話まで割っても収まらない場合はその圧縮を受け入れる。
        """
        pending = list(
            reversed(split_turns_by_estimate(turns, chunk_sec, self.max_utts))
        )
        planned: list[tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]] = []
        while pending:
            chunk = pending.pop()
            chunk_id = f"{dialogue_id}_c{len(planned):03d}"
            placed, meta = self.timeline(chunk, chunk_id)
            if meta["fit_scale"] < min_fit_scale and len(chunk) > 1:
                head, tail = split_in_half(chunk)
                pending.append(tail)
                pending.append(head)
                continue
            planned.append((chunk, placed, meta))
        return planned

    def render_dialogue(
        self,
        dialogue: dict[str, Any],
        args: argparse.Namespace,
        seed_offset: int,
    ) -> dict[str, Any]:
        import torch

        turns = dialogue["turns"]
        dialogue_id = str(dialogue.get("id") or "dialogue")
        planned = self.plan_chunks(turns, dialogue_id, args.chunk_sec, args.min_fit_scale)

        gap_samples = int(round(args.chunk_gap_sec * self.sample_rate))
        pieces: list[Any] = []
        utterance_alignments: list[list[Any]] = []
        chunk_meta: list[dict[str, Any]] = []
        offset_samples = 0
        for chunk_idx, (chunk, placed, meta) in enumerate(planned):
            wav, active_frames, overlap = self.render_chunk(
                placed, seed=self.seed + seed_offset * 997 + chunk_idx
            )
            if pieces and gap_samples > 0:
                pieces.append(torch.zeros(2, gap_samples))
                offset_samples += gap_samples
            offset_sec = offset_samples / self.sample_rate
            for utt in placed:
                start = offset_sec + utt["start"] / FPS
                end = start + sum(utt["durs"]) / FPS
                turn = chunk[utt["turn_index"]]
                utterance_alignments.append(
                    [
                        turn["text"],
                        [round(start, 4), round(end, 4)],
                        LABEL_OF_SPEAKER[turn["speaker"]],
                    ]
                )
            pieces.append(wav)
            offset_samples += wav.shape[-1]
            chunk_meta.append(
                {
                    "chunk_index": chunk_idx,
                    "n_utts": int(meta["n_utts"]),
                    "n_turns_in": len(chunk),
                    "fit_scale": round(float(meta["fit_scale"]), 4),
                    "overlap": round(overlap, 4),
                    "active_frames": active_frames,
                    "offset_sec": round(offset_sec, 4),
                }
            )

        if not pieces:
            raise ValueError("合成できたチャンクがありません")
        stereo = torch.cat(pieces, dim=-1)
        utterance_alignments.sort(key=lambda row: row[1][0])
        return {
            "stereo": stereo,
            "alignments_utterance": utterance_alignments,
            "chunks": chunk_meta,
        }


# ---------------------------------------------------------------------------
# 出力


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rebuild_from_completed(
    data_dir: Path,
    manifest_path: Path,
    dialogues_path: Path,
    done_stems: set[str],
) -> int:
    """既に書けている WAV+JSON から manifest / dialogues.jsonl を作り直す。

    generate_qwen3_tts_data.rebuild_from_completed と同じ考え方。walltime で
    途中終了しても、残っている sidecar と完全に一致する JSONL に戻してから
    続きを合成する。
    """
    out_dir = manifest_path.parent
    manifest_rows: list[tuple[str, str]] = []
    dialogue_rows: list[tuple[str, str]] = []
    for json_path in sorted(data_dir.glob("*.json")):
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            meta = payload["metadata"]
            duration = float(meta["duration_sec"])
            dialogue = meta["dialogue"]
        except (OSError, ValueError, KeyError):
            continue
        stem = json_path.stem
        done_stems.add(stem)
        rel = str(wav_path.relative_to(out_dir)).replace("\\", "/")
        manifest_rows.append(
            (stem, json.dumps({"path": rel, "duration": duration}, ensure_ascii=False))
        )
        dialogue_rows.append((stem, json.dumps(dialogue, ensure_ascii=False)))

    manifest_rows.sort()
    dialogue_rows.sort()
    manifest_path.write_text("".join(row + "\n" for _, row in manifest_rows), encoding="utf-8")
    dialogues_path.write_text("".join(row + "\n" for _, row in dialogue_rows), encoding="utf-8")
    return len(done_stems)


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dialogues with KABURI-TTS.")
    parser.add_argument("--dialogues-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--kaburi-repo",
        type=Path,
        default=Path(os.environ.get("KABURI_REPO", REPO_ROOT.parent / "kaburi-tts")),
        help="KABURI-TTS のクローン先（既定: ../kaburi-tts または $KABURI_REPO）",
    )
    parser.add_argument(
        "--ref-pack",
        type=Path,
        required=True,
        help="話者参照パック（make_ref_pack.py の出力、または同梱の assets/test_refpack）",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-dialogues", type=int, default=None)
    parser.add_argument(
        "--success-target",
        type=int,
        default=None,
        help="この本数だけ成功したら打ち切る（shard の spare 分を使い切らないため）",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-name", default="synthetic_moshi_train.jsonl")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=26.0,
        help="1 チャンクに詰める秒数の見積もり上限（キャンバスは 30 秒固定）",
    )
    parser.add_argument(
        "--chunk-gap-sec",
        type=float,
        default=0.3,
        help="連結するチャンクの間に入れる無音",
    )
    parser.add_argument(
        "--min-fit-scale",
        type=float,
        default=0.9,
        help="これを下回る圧縮が起きたチャンクは分割してやり直す",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="書き出しサンプリングレート（0 で KABURI の 48kHz のまま）",
    )
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--acoustic-ckpt", default=None)
    parser.add_argument("--realizer-ckpt", default=None)
    parser.add_argument("--gap-ckpt", default=None)
    parser.add_argument("--raster-release", default=None)
    parser.add_argument("--no-activity-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dialogues = load_dialogues(args.dialogues_jsonl)
    if args.num_dialogues is not None:
        dialogues = dialogues[: args.num_dialogues]
    if not dialogues:
        raise SystemExit(f"合成対象の対話がありません: {args.dialogues_jsonl}")
    logger.info("対話 %d 件を %s から読み込みました", len(dialogues), args.dialogues_jsonl)

    out_dir = args.out_dir
    data_dir = out_dir / "data_stereo"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest_name
    dialogues_path = out_dir / "dialogues.jsonl"

    done_stems: set[str] = set()
    if args.resume:
        done = rebuild_from_completed(data_dir, manifest_path, dialogues_path, done_stems)
        logger.info("resume: 既に %d 件が書けています", done)
    else:
        manifest_path.write_text("", encoding="utf-8")
        dialogues_path.write_text("", encoding="utf-8")

    renderer = KaburiRenderer(args)

    import torch
    import torchaudio

    success = len(done_stems)
    failed = 0
    for index, dialogue in enumerate(dialogues, start=1):
        if args.success_target is not None and success >= args.success_target:
            logger.info("success-target %d に到達したので打ち切ります", args.success_target)
            break
        dialogue_id = str(dialogue.get("id") or f"dialogue_{index:05d}")
        stem = f"sample_{index:05d}_{safe_stem(dialogue.get('title') or dialogue_id, dialogue_id)}"
        if stem in done_stems:
            continue
        log_progress = args.verbose or index % max(1, args.log_every) == 0 or index == 1
        started = time.time()
        try:
            rendered = renderer.render_dialogue(dialogue, args, seed_offset=index)
        except Exception as exc:  # noqa: BLE001 - 1 本失敗しても shard は続ける
            failed += 1
            logger.warning(
                "[%d/%d] 対話 %s の合成に失敗したのでスキップします: %s",
                index, len(dialogues), dialogue_id, exc,
                exc_info=args.verbose,
            )
            continue

        stereo = rendered["stereo"]
        sample_rate = renderer.sample_rate
        if args.sample_rate and args.sample_rate != sample_rate:
            stereo = torchaudio.functional.resample(stereo, sample_rate, args.sample_rate)
            sample_rate = args.sample_rate

        duration = stereo.shape[-1] / sample_rate
        wav_path = data_dir / f"{stem}.wav"
        json_path = wav_path.with_suffix(".json")
        torchaudio.save(str(wav_path), stereo.to(torch.float32), sample_rate, channels_first=True)

        # moshi-finetune の Interleaver は 1 エントリ = 1 単語を前提にしている
        # ので、発話単位のアライメントを単語単位に割り直して alignments に書く
        # （発話単位の原本は alignments_utterance に残す）。
        alignments, _stats = split_utterance_alignments(rendered["alignments_utterance"])
        elapsed = time.time() - started
        payload = {
            "alignments": alignments,
            "alignments_utterance": rendered["alignments_utterance"],
            "metadata": {
                "alignments_granularity": "word",
                "alignments_word_split": {
                    "version": 1,
                    "method": "char-proportional",
                    "segmenter": get_segmenter_name(),
                },
                "mode": "kaburi-raster",
                "sample_rate": sample_rate,
                "duration_sec": round(duration, 4),
                "tts_backend": "kaburi",
                "kaburi": {
                    "raster_release": renderer.timing.release,
                    "ref_pack": str(args.ref_pack),
                    "speaker_A": renderer.speaker_a,
                    "speaker_B": renderer.speaker_b,
                    "template_chunk": renderer.template_chunk,
                    "cfg_scale": args.cfg_scale,
                    "num_steps": args.num_steps,
                    "chunk_sec": args.chunk_sec,
                    "chunk_gap_sec": args.chunk_gap_sec,
                    "min_fit_scale": args.min_fit_scale,
                    "native_sample_rate": renderer.sample_rate,
                    "activity_gate": (
                        renderer.gate_cfg if renderer.gate_on else {"enabled": False}
                    ),
                    "chunks": rendered["chunks"],
                },
                "left_channel": "moshi",
                "right_channel": "user",
                "wall_time_sec": round(elapsed, 3),
                "dialogue": {
                    "id": dialogue.get("id"),
                    "category": dialogue.get("category"),
                    "risk_level": dialogue.get("risk_level"),
                    "title": dialogue.get("title"),
                    "duplex_task": dialogue.get("duplex_task"),
                    "turns": dialogue["turns"],
                },
            },
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        append_jsonl(
            dialogues_path,
            {
                "id": dialogue.get("id"),
                "category": dialogue.get("category"),
                "risk_level": dialogue.get("risk_level"),
                "title": dialogue.get("title"),
                "duplex_task": dialogue.get("duplex_task"),
                "turns": dialogue["turns"],
            },
        )
        append_jsonl(
            manifest_path,
            {
                "path": str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                "duration": duration,
            },
        )
        done_stems.add(stem)
        success += 1
        if log_progress:
            mean_overlap = sum(c["overlap"] for c in rendered["chunks"]) / len(rendered["chunks"])
            logger.info(
                "[%d/%d] %s: %.1f 秒 / chunks=%d / overlap=%.3f (wall %.1f 秒)",
                index, len(dialogues), wav_path.name, duration,
                len(rendered["chunks"]), mean_overlap, elapsed,
            )

    logger.info("完了: 成功 %d 件 / 失敗 %d 件 -> %s", success, failed, manifest_path)
    if success == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
