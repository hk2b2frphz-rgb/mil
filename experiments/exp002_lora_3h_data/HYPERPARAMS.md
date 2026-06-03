# exp002_lora_3h_data — データ量を ~3h に増やした比較実験

## 目的

exp001 (~1h, 100 対話) に対し、合成データを **~3h (250 対話)** に増やしたときの
eval loss と最終的な対話品質の改善幅を測る。「データを増やせば良くなるのか、
それともこの LoRA + tokenizer 構成ではここで頭打ちか」を切り分けるための実験。

## exp001 との差分

| key | exp001 | exp002 | 理由 |
|---|---|---|---|
| 元データの本数 | 100 対話 (~1h) | **250 対話 (~3h)** | NUM_CASES=250 で `run_pipeline.sh` 再実行 |
| `max_steps` | 500 | **1200** | データ量 2.5 倍 → 同 epoch 数（~44）を維持するため step も 2.5 倍 |
| `ckpt_freq` | 50 | **120** | 保存ポイント数を ~10 個に揃える（10 通りの best 候補） |
| `eval_freq` | 25 | **60** | ckpt_freq の半分 |
| その他全て | — | **完全同値** | データ量の純粋な効果だけを観察するため固定 |

## 揃えた項目（exp001 と同値）

- `lora.rank=32`, `lora.scaling=2.0`, `ft_embed=false`
- `optim.lr=2e-6`, `weight_decay=0.1`, `pct_start=0.05`
- `batch_size=8`, `num_microbatches=1`, `duration_sec=100`
- `first_codebook_weight_multiplier=100`, `text_padding_weight=0.5`
- `gradient_checkpointing=true`, `param_dtype="bfloat16"`
- `seed=0`

## 期待される結果と判断基準

- **eval loss の到達点**: exp001 の最終 eval loss を下回れば「データ増で改善する
  フェーズにある」。差が誤差（< 数%）なら「100 対話で既に LoRA の表現力上限」。
- **過学習開始 step**: 250 対話 / batch 8 → 28 step/epoch なので、過学習が出るとしたら
  exp001 同等の epoch 数 (~44) 付近、つまり step ~1230 前後。
  500 step 以前に飽和する場合は LoRA rank 不足の可能性が濃い → 次の実験では rank を
  64 に上げて exp003 を作る。
- **学習時間**: exp001 の 500 step に対し 2.4 倍 → 概ね 2 倍強。
  A100 80GB / batch 8 で eval 含めて 1〜2 時間想定。

## 参考文献

- exp001 と同（`exp001_lora_baseline/HYPERPARAMS.md` 参照）
- データスケーリングについては Kaplan et al. "Scaling Laws for Neural Language
  Models" (2020) https://arxiv.org/abs/2001.08361 が一般論。本実験はデータ点 2 つ
  なのでスケーリング則の検証ではなく、純粋な「もっと効くか確認」。
