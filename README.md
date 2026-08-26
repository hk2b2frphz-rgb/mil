# moshimoshi-J

日本語の傾聴対話向けに Moshi (`llm-jp/llm-jp-moshi-v1`) をドメイン適応する
パイプライン。合成対話の生成から学習・評価まで。

```
対話生成 -> TTS(ステレオWAV) -> 学習 -> 評価
```

## セットアップ

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune
uv sync
uv sync --project gemma_runtime
```

Python 3.11+ / NVIDIA GPU。計算ノードからの外部取得にプロキシが要る環境では
`PROXY_URL` を明示的に渡す（`qsub -V` で引き継がれる）。

```bash
qsub -v PROXY_URL=http://<proxy-host>:<port>,MODEL_ID=base scripts/run_full_duplex_eval.pbs
```

## 1. 学習データ生成

対話を生成し（A100・vLLM で Qwen3.6-27B）、話者ごとに全発話を連結して1回で
合成、MMS_FA で境界を復元する。

```bash
qsub -V scripts/run_dialogues_qwen_3000.pbs
qsub -V scripts/run_qwen_tts_whole_utterance_3000_4gpu.pbs
```

対話数は `1000` / `3000` / `10000` の3系統。動作確認は `_smoke` を使う。
TTS のバックエンドは 1000 が Qwen3-TTS、3000/10000 は Kokoro（Qwen3-TTS では
walltime 内に終わらないため）。

## 2. 学習

ハイパラは `experiments/<name>/config.yaml` で管理する。

```bash
bash scripts/run_experiment.sh lora_base_config ./data/runs/<RUN_ID>
```

walltime を越える step 数はチェーンで分割する。LR スケジュールを連続させる
ため `max_steps` は全ジョブで `TOTAL_STEPS` 固定。

```bash
qsub -v 'EXP_NAME=lora_base_config,SRC_RUN_DIR=data/runs/<RUN_ID>,TOTAL_STEPS=7200' \
  scripts/run_train_chain.pbs
```

チェーンのロジックだけを先に検証する:

```bash
bash scripts/check_train_chain.sh
```

## 3. 評価

Full-Duplex-Bench-JA。計算ノード側は外部 API を呼ばない。

```bash
qsub -v MODEL_ID=base scripts/run_full_duplex_eval.pbs
qsub -v MODEL_ID=lora_h01,MODEL_WEIGHT=/path/model.safetensors,MODEL_CONFIG=/path/moshi_lm_kwargs.json \
  scripts/run_full_duplex_eval.pbs
```

LoRA は評価前にマージする。

```bash
qsub -v LORA_CKPT=/path/to/checkpoint/consolidated/lora.safetensors scripts/merge_lora.pbs
```

## ディレクトリ

| | |
| --- | --- |
| `scripts/` | 生成・学習・評価の実行スクリプトと PBS ジョブ |
| `eval/` | Full-Duplex-Bench-JA の評価本体 |
| `configs/` | DeepSpeed / TTS / GRPO の設定 |
| `experiments/` | 実験ごとのハイパラ |
| `eval_sets/` | 評価シナリオと正解データ |
| `gemma_runtime/` | 対話生成・カスケード用 Gemma の隔離 venv |
| `agent_hpc/` | A100 上のコーディングモデルをローカルから使う一式 |
