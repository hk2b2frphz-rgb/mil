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

### 既存の学習データを指定して実行

生成済みの学習データを使う場合は、`SRC_RUN_DIR` に
`training_set/synthetic_moshi_train.jsonl` を含む run ディレクトリを指定する。

```bash
# 例:
#   ./data/runs/3h_dataset/training_set/synthetic_moshi_train.jsonl

export SRC_RUN_DIR=./data/runs/3h_dataset

# full fine-tuning sweep. Default is f01 only; override patterns for tuning.
qsub scripts/fullft_sweep.pbs

# 必要なら sweep pattern を上書きする。
qsub -v SRC_RUN_DIR=./data/runs/3h_dataset,SWEEP_PATTERNS=f01,f04 scripts/fullft_sweep.pbs
```

PBS を使わずにシェルから直接試す場合:

```bash
SRC_RUN_DIR=./data/runs/3h_dataset \
SWEEP_PATTERNS="f01 f02" \
bash scripts/run_fullft_sweep_pair.sh
```

同じ `SRC_RUN_DIR` は LoRA sweep と単発PBSにも使える。

```bash
qsub -v SRC_RUN_DIR=./data/runs/3h_dataset,SWEEP_PATTERNS=h01,h02 scripts/sweep_lora.pbs
qsub scripts/run_experiment.pbs
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

## 学習済みモデルの利用

### LoRA チェックポイントのマージ

LoRA 実験のチェックポイントをベースモデルにマージして、推論に使える単一ファイルを生成する。

```bash
# マージ
uv run --project ../moshi-finetune python scripts/merge_lora.py \
  --lora-ckpt experiments/exp001_lora_baseline/checkpoints/<ts>/checkpoint_000500/consolidated/lora.safetensors \
  --out merged_model/consolidated.safetensors

# マージ済みモデルで推論
uv run python response_recorder.py \
  --moshi-weight merged_model/consolidated.safetensors \
  --inputs prompts/hello.wav --out-dir results/lora_merged/
```

`--scaling` は checkpoint の `config.json` から自動取得される。

### フル FT チェックポイント

フル FT (exp100) の場合はマージ不要。チェックポイントをそのまま指定する。

```bash
uv run python response_recorder.py \
  --moshi-weight experiments/exp100_full_ft/checkpoints/<ts>/checkpoint_000500/consolidated/consolidated.safetensors \
  --inputs prompts/hello.wav --out-dir results/full_ft/
```

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
  merge_lora.py                    # LoRA adapter をベースモデルにマージ
experiments/
  exp001_lora_baseline/            # 各実験の config.yaml + HYPERPARAMS.md
  exp002_lora_3h_data/
  exp100_full_ft/
configs/
  moshi_lora_jp_loneliness.yaml    # パイプライン用 FT config
data/runs/                         # 実行ごとの出力 (git ignore)
```

## Full-FT / LoRA repository split

- Full fine-tuning: `nu-dialogue/moshi-finetune`, default checkout
  `../moshi-finetune-nu-dialogue`.
- LoRA fine-tuning: `kyutai-labs/moshi-finetune`, default checkout
  `../moshi-finetune`.

`scripts/run_fullft_sweep_pair.sh` uses the nu-dialogue repo and will clone it
if `../moshi-finetune-nu-dialogue` is missing. `scripts/run_sweep_pair.sh` and
`scripts/run_experiment.sh` keep using the Kyutai repo for LoRA.

Existing generated data can be reused by pointing `SRC_RUN_DIR` at the run
directory that contains `training_set/synthetic_moshi_train.jsonl`:

```bash
SRC_RUN_DIR=/path/to/data/runs/3h_dataset \
SWEEP_PATTERNS="f01" \
NPROC=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_fullft_sweep_pair.sh
```

PBS full-FT tuning uses `scripts/fullft_sweep.pbs` with `res=middle`,
`NPROC=2`, and `CUDA_VISIBLE_DEVICES=0,1` by default. LoRA tuning uses
`scripts/sweep_lora.pbs` with `res=small`. Details are in
`experiments/fullft_3h_sweep_10_patterns.md`.

For full fine-tuning, MLflow receives machine-readable stdout metrics from the
nu-dialogue runner. The important curves are `train.loss`, `train.loss.text`,
`train.loss.audio`, `eval.loss`, `eval.loss.text`, `eval.loss.audio`,
`train.accuracy.*`, `eval.accuracy.*`, and `learning_rate.*`.
