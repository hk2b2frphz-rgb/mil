# exp001_lora_baseline — ハイパラ設計意図

llm-jp/llm-jp-moshi-v1 に日本語孤独感カウンセリング会話を LoRA で追加適応する
最初のベースライン。データは Qwen3-TTS で合成した約 100 対話（~1h）。

## 出発点

moshi-finetune 公式 example (`../moshi-finetune/example/moshi_7B.yaml`) を出発点に、
**(a) 1 GPU 24GB に収まる構成**、**(b) 100 サンプル規模の小データに合わせた正則化**、
**(c) 過学習を見える化する eval split** の 3 点だけ調整する方針。元 example の
意図しない逸脱を避ける。

## 各ハイパラの根拠

### LoRA: `rank=32`, `scaling=2.0`, `ft_embed=false`

- **公式 example は `rank=128`**。これは大規模 finetune 用（例の `batch_size=16` /
  `max_steps=2000` で十分なデータが流れる前提）。今回は 100 サンプル ≈
  公式の 1/32 のデータ量なので、過学習防止のため rank を 1/4 の 32 に下げる。
- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)**
  によれば rank 4–32 で full FT に近い性能。Multimodal/audio 系の Moshi では
  もう少し容量が要るとされるので 32 を採用（8 は不足、128 は過剰、の中庸）。
- **`scaling = α/r`**。α=64 で公式と同じ感覚（公式は r=128, α=256 で同じ α/r=2）。
  LoRA paper 慣例の「α は r の 2 倍」を踏襲。
- **`ft_embed=false`**: 公式デフォルト。embedding まで学習すると tokenizer 全体に
  影響しデータ規模に対し過剰。

### Optimizer: `lr=2e-6`, `weight_decay=0.1`, `pct_start=0.05`

- **`lr=2e-6` は公式 example と同値**。これは OneCycleLR の **peak lr** である点に
  注意。低く見えるが warmup → cooldown スケジュール下では妥当。LoRA 全般での
  慣行値 1e-4 とは前提が違う（あちらは AdamW の定常 lr）。
- **`pct_start=0.05`** は OneCycleLR の warmup 比率（Smith, "Super-Convergence" 2017）。
  公式準拠。max_steps=800 のうち最初の 40 step が warmup。
- **`weight_decay=0.1`** も公式と同値。LoRA のみ更新するので model 本体は
  動かず、LoRA パラメータへの L2 ペナルティとして機能。

### バッチサイズ周り: `batch_size=1`, `num_microbatches=8`, `duration_sec=100`

- **`batch_size=1`** は 24GB GPU の現実的上限。`duration_sec=100` で 100 秒の
  audio + Mimi codes + LoRA grad で VRAM が埋まる。
- **`num_microbatches=8`** で gradient accumulation。effective batch = 8。
  公式 16 の半分だが、データ規模も小さいので step 数で吸収する。
- **`duration_sec=100`** は公式と同値。今回の対話は最長でも 90 秒前後なので
  全データが 1 サンプルに収まる。

### 学習量: `max_steps=800`

- 100 サンプル / effective batch 8 = 12.5 step/epoch → **約 64 epoch**。
- 公式 example の 2000 step は bs=16 × ~10 epoch ＝ 160000 sample relative の
  曝露で、今回データ規模では過剰 (=  100 epoch 越え）。
- LoRA + 小データの一般的な経験則は **30–80 epoch**（QLoRA, Dettmers et al. 2023
  の Alpaca finetune が 3 epoch / 50k sample の規模感、データ量比から逆算）。
  64 epoch はその上限寄りで、過学習し始めたら ckpt から戻す前提。
- 不足だったら 1500 step に伸ばす、過学習なら 400 に縮める想定。

### 監視と保存

- **`do_eval: true`, `eval_freq: 50`**: 50 step ごとに eval loss を出す。
  公式 example は `do_eval: false` だが、小データではこれが無いと
  best ckpt を選べない。
- **`ckpt_freq: 100`, `num_ckpt_keep: 5`**: 100, 200, ..., 800 と
  ある程度残し、最新 5 個を保持。eval loss を見て後で選別。
- **`save_adapters: true`**: LoRA adapter のみ保存。`full_finetuning: false` の
  必須セット。
- **`log_freq: 5`**: 5 step ごとに loss/lr を吐く。短時間で挙動を見たい初期実験向け。

### モデル損失関連: `first_codebook_weight_multiplier=100`, `text_padding_weight=0.5`

- 公式 example と同値。Moshi の最初の codebook (semantic) は他より重要なので
  loss を 100 倍。text padding の loss 重みは 0.5。**根拠は moshi-finetune 上流の
  公式設定**で、独自に動かす特段の理由がない限り変更しない。

## 評価データ分割 (9:1)

- launcher が manifest をシャッフル後 9:1 で split。
- 100 対話 → train 90, eval 10。
- `seed=0` 固定で再現性を確保。

## 期待される結果と判断基準

- 動作確認の意味で、まず eval loss が **単調減少 → 平坦化** することを確認。
- 平坦化したら 1 step 前後の ckpt を best とみなす。明確に上昇に転じたら手前を取る。
- 200 step 経っても eval loss がほぼ動かない場合は `lr` か `rank` の見直し
  （次の実験は `lr=5e-6` か `rank=64` を試す）。

## 参考文献

- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
  https://arxiv.org/abs/2106.09685
- Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large
  Learning Rates." 2017. https://arxiv.org/abs/1708.07120 (OneCycleLR)
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." 2023.
  https://arxiv.org/abs/2305.14314 (LoRA 小データ epoch 感覚)
- Défossez et al. "Moshi: a speech-text foundation model for real-time dialogue."
  2024. https://arxiv.org/abs/2410.00037 (codebook 構造)
- moshi-finetune 公式 example:
  https://github.com/kyutai-labs/moshi-finetune/blob/main/example/moshi_7B.yaml
