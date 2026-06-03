# moshimoshi-J

日本語孤独・孤立相談窓口向けに Moshi (llm-jp/llm-jp-moshi-v1) を LoRA で
ドメイン適応するためのパイプライン。合成データの生成から fine-tune まで一気通貫で回せる。

## パイプライン概要

```
1. use_cases.jsonl          ← 軸の組み合わせで多様な相談ケースを生成
       ↓ build_use_cases.py
2. dialogues.jsonl          ← Gemma 4 が感情・沈黙付き対話スクリプトを生成
       ↓ generate_synthetic_moshi_training_data.py
3. ステレオ WAV + manifest  ← Qwen3-TTS で音声合成 (moshi=左ch, user=右ch)
       ↓ generate_qwen3_tts_data.py
4. Moshi LoRA fine-tune     ← moshi-finetune で学習
```

## セットアップ

```bash
# moshi-finetune を兄弟ディレクトリに clone
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune

# 依存の sync
uv sync                           # Moshi + Qwen3-TTS
uv sync --project gemma_runtime   # Gemma 4
```

**要件**: Python 3.11+, NVIDIA GPU (A100 80GB 推奨), CUDA 対応 PyTorch

## 使い方

### フルパイプライン (データ生成 → 学習)

```bash
# 100 対話 (~1h) を生成して学習まで一気に回す
bash scripts/run_pipeline.sh

# 対話数やステップを環境変数で調整
NUM_CASES=250 bash scripts/run_pipeline.sh
STEPS=use_cases,dialogues bash scripts/run_pipeline.sh   # 音声化前まで
```

### 実験ベース (推奨)

ハイパラを変えて比較する場合は `experiments/` 配下で管理する。

```bash
# 既存データで実験を起動
bash scripts/run_experiment.sh exp001_lora_baseline ./data/runs/2026-06-02_130539

# データ生成 + 実験をまとめて実行
bash scripts/generate_and_run.sh exp002_lora_3h_data 250
```

詳細は [experiments/README.md](experiments/README.md) 参照。

### 個別ステップ

```bash
# 1. use case 生成
uv run python scripts/build_use_cases.py --out-path data/v1/use_cases.jsonl --num 100

# 2. Gemma で対話生成
uv run python scripts/generate_synthetic_moshi_training_data.py \
  --out-dir data/v1/gemma_dialogues \
  --use-cases-jsonl data/v1/use_cases.jsonl \
  --num-dialogues 100 --mode dialogues-only \
  --gemma-backend transformers-subprocess --allow-template-fallback

# 3. Qwen3-TTS で音声合成
uv run python scripts/generate_qwen3_tts_data.py \
  --out-dir data/v1/training_set \
  --dialogues-jsonl data/v1/gemma_dialogues/dialogues.jsonl
```

## 実験一覧

| 実験 | データ量 | 狙い |
|---|---|---|
| `exp001_lora_baseline` | ~100 対話 (~1h) | LoRA rank=32 ベースライン |
| `exp002_lora_3h_data` | ~250 対話 (~3h) | データ量の効果を検証 (ハイパラ同一) |
| `exp100_full_ft` | ~100 対話 (~1h) | フル fine-tuning で LoRA との比較 |

## Response Recorder

Moshi に固定音声を入力して応答を録音する実験ツール。
ドメイン適応の前後比較に使う。

```bash
uv run python response_recorder.py \
  --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/

# テキストプロンプトも可
echo "こんにちは" | uv run python response_recorder.py --out-dir results/stdin/
```

`--hf-repo` でモデル指定、`--silence-sec` で応答待ち時間を調整。
詳細は `python response_recorder.py --help` を参照。

## ディレクトリ構成

```
scripts/
  run_pipeline.sh                  # E2E パイプライン
  run_experiment.sh                # 実験単体起動
  generate_and_run.sh              # データ生成 + 実験起動
  build_use_cases.py               # use case 生成
  generate_synthetic_moshi_training_data.py  # Gemma 対話生成
  generate_qwen3_tts_data.py       # Qwen3-TTS 音声合成
experiments/
  exp001_lora_baseline/            # 各実験の config.yaml + HYPERPARAMS.md
  exp002_lora_3h_data/
configs/
  moshi_lora_jp_loneliness.yaml    # パイプライン用 FT config
data/runs/                         # 実行ごとの出力 (git ignore)
```
