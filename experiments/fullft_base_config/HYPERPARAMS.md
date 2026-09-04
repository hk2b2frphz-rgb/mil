# fullft_base_config — フル fine-tuning ベース設定

`scripts/run_fullft_sweep_pair.sh` / `scripts/fullft_sweep.pbs` が
`BASE_EXP` としてコピーして使うテンプレート（各 sweep パターンはこの
config.yaml を差分パッチして使う。実際に採用された学習率などは
[knowledge/decisions/0001_fullft_lr_1e-5.md](../../knowledge/decisions/0001_fullft_lr_1e-5.md)
を参照）。

## 目的

`lora_base_config` (LoRA rank=32) と同じデータ・同じ曝露量で**全パラメータ更新**
した場合の eval loss と対話品質を比較する。LoRA の表現力上限を超えられるか
確認する。

## lora_base_config との差分

| key | lora_base_config (LoRA) | fullft_base_config (Full FT) | 理由 |
|---|---|---|---|
| `full_finetuning` | false | **true** | 全パラメータ更新 |
| `lora.enable` | true (rank=32) | **false** | LoRA 不使用 |
| `save_adapters` | true | **false** | モデル全体を保存 |
| `batch_size` | 8 | **1** | full FT の activation/optimizer state で OOM しやすいため per-forward を最小化 |
| `num_microbatches` | 1 | **8** | 実効 batch = 1×8 = 8 で lora_base_config と同等 |
| `optim.lr` | 2e-6 | **5e-7** | 全パラメータが動くので 1/4 に下げる |

## ハードウェア前提

- **NVIDIA A100 80GB × 1**
- bf16 で運用
- `gradient_checkpointing: true` 必須（フル FT では activations + optimizer state で
  80GB ギリギリ）

## VRAM 見積もり

- モデルパラメータ (bf16): ~14 GB
- Adam optimizer state (fp32 momentum + variance): ~28 GB
- Activations (gradient checkpointing ON): ~10-20 GB (batch_size 依存)
- 合計: ~52-62 GB → A100 80GB に収まる見込み

batch_size=1 + microbatch=8 で grad accum し、実効 batch を lora_base_config と揃える。
これでも OOM が出る場合は `duration_sec=80` などで系列長を下げる。

## 学習率

- フル FT で 7B モデル全体を動かすので、LoRA の lr=2e-6 より保守的に **5e-7** を採用。
- 全パラメータへの更新なので、大きすぎると catastrophic forgetting が起きる。
- 不足なら 1e-6 に上げる、過学習が早いなら 2e-7 に下げる。

## 期待される結果と判断基準

- **eval loss が lora_base_config を下回る**: フル FT の容量が活きている → データを増やせばさらに改善の余地。
- **eval loss が lora_base_config と同等**: LoRA rank=32 で十分な表現力がある → フル FT のコスト不要。
- **eval loss が lora_base_config より悪い**: lr が不適切 or catastrophic forgetting → lr 調整で再実験。
- **OOM**: batch_size=1 / microbatch=8 に変更して再実行。

## チェックポイントサイズ

フル FT では LoRA adapter (~数十 MB) ではなくモデル全体 (~14 GB/ckpt) を保存する。
`num_ckpt_keep=5` で最大 ~70 GB のディスクを消費する点に注意。

## 参考文献

- lora_base_config と同（`lora_base_config/HYPERPARAMS.md` 参照）
