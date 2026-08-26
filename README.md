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

h01 のようなスイープパターンをそのままチェーンで回す場合（`TOTAL_STEPS` は
パターン自身の step 数が既定になる）:

```bash
qsub -v 'EXP_NAME=lora_base_config,SRC_RUN_DIR=data/runs/<RUN_ID>,SWEEP_PATTERN=h01,STEPS_PER_JOB=1000' \
  scripts/run_train_chain.pbs
```

パターンの定義は `scripts/sweep_patterns.sh`。スイープとチェーンで共有する。

チェーンのロジックだけを先に検証する:

```bash
bash scripts/check_train_chain.sh
```

### スイープ

複数パターンを順に回す。**そのまま投げれば walltime を跨げる。** 残り1時間に
なった時点で学習を止め、最後のチェックポイントから続きを次のジョブに引き継ぐ
（同じデータセットを使い回し、完了済みパターンはスキップする）。

```bash
qsub -V scripts/sweep_lora.pbs
qsub -V scripts/fullft_sweep.pbs
```

進捗は `experiments/pbs_logs/<RUN_ID>_sweep_state.tsv`。引き継ぎを止めるなら
`SWEEP_CHAIN=0`、走っているチェーンを終わらせるなら
`touch ~/.miltoka/stop_sweep_chain`。

止める余裕は `TIMEBOX_LEAD_SEC`（既定 3600 秒）。チェックポイントは
`HP_CKPT_FREQ` step ごとに書かれているので、引き継ぎ1回あたり失うのは
最大でその step 数。

full-FT も同じように跨げる。ただし再開の作りが違う: LoRA はアダプタを
読み直すだけだが、full-FT は ZeRO チェックポイントを
`export_fullft_checkpoint.py --intermediate-only` で MoshiForFinetuning 形式に
戻し、それを次ジョブの `NU_MODEL_DIR` に渡す。重みだけが渡り optimizer state は
渡らないので、残り step 数を引いたうえで warmup は初回のみ行う。

full-FT のチェックポイント間隔は `HP_CKPT_FREQ`。1回の walltime 内に最低1つは
書かれる値にしておくこと。1つも無いと再開点が作れず、そのジョブは失敗する。

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
