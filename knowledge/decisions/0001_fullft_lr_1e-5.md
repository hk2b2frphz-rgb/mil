# 0001: full-FT のベースLRを 1e-5 にする（nu-dialogue既定 3e-5 ではなく）

状態: 下書き / 更新日: 2026-07-01

## 文脈
~3h（~250対話）規模の合成データで Moshi を full fine-tune する際の学習率。
nu-dialogue/moshi-finetune の既定は `3e-5`。

## 決定
full-FTベースライン(`f01`)のLRを **1e-5** に設定。`3e-5` は参照用 `lr_3e-5` として温存。

## 理由
このデータ量では `3e-5` だとほぼ即座に過学習する（eval lossが早期に反転）。
1e-5 で過学習の立ち上がりを観測しやすくする。schedule はエポック基準で
`max_epochs=12 / warmup=1ep / eval=0.5ep / ckpt=1ep`。

## 代替案
- `3e-5`（既定）: 過学習が速すぎ、tuningの解像度が取れない → 却下（参照用に残す）。
- LR sweep `lr_2e-5 / lr_1e-5 / lr_5e-6`: 比較のため別途用意。

## 影響・振り返り
- keep-best-only で eval loss 最小の checkpoint を自動選択・export する運用と対。
- <TODO: 実際のsweep結果でベストLR/エポックを追記>

## 関連
- コード: [scripts/run_fullft_sweep_pair.sh](../../scripts/run_fullft_sweep_pair.sh), [experiments/fullft_3h_sweep_10_patterns.md](../../experiments/fullft_3h_sweep_10_patterns.md)
- 概念/論文: [[LoRA]] / [[ZeRO]]（paper_map）
