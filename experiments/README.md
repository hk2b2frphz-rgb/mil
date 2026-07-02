# experiments/

実験フォルダ。1 つの fine-tune ランの構成(config.yaml + HYPERPARAMS.md)を
`<short_descriptive_name>/` 単位で管理する。

- `lora_base_config/` — LoRA sweep のベース設定。`scripts/run_sweep_pair.sh` /
  `scripts/run_experiment.pbs` が `BASE_EXP` としてデフォルトでコピーする
  （sweepパターンは [sweep_10_patterns.md](sweep_10_patterns.md) 参照）。
- `fullft_base_config/` — full fine-tuning sweep のベース設定。
  `scripts/run_fullft_sweep_pair.sh` / `scripts/fullft_sweep.pbs` が
  `BASE_EXP` としてデフォルトでコピーする
  （sweepパターンは [fullft_3h_sweep_10_patterns.md](fullft_3h_sweep_10_patterns.md) 参照）。

実際の sweep 実行結果は `experiments/_sweeps/<RUN_ID>_<pattern>/` や
`experiments/_fullft_sweeps/<RUN_ID>_<pattern>/`（いずれも git ignore）に
生成される。上の2フォルダはその元になるテンプレートであり、それ自体を
直接学習ジョブとして起動することは通常ない。

## 各実験フォルダの構成

```
lora_base_config/
├── config.yaml             # moshi-finetune 用 YAML（テンプレ、パスは launcher が埋める）
├── HYPERPARAMS.md          # 設計意図・参考文献・期待結果（必須）
├── data/                   # 学習データのコピー（launcher が作成、git ignore）
│   ├── training_set/       #   ← data/runs/<RUN_ID>/training_set から hardlink/copy
│   ├── train.jsonl         #   ← 9:1 split の train manifest
│   └── eval.jsonl          #   ← 9:1 split の eval manifest
├── checkpoints/            # train.py が出力（git ignore）
├── args.yaml               # train.py が出力（実際に使われた全 args、振り返り用）
├── run.log                 # launcher が記録した stdout/stderr
└── notes.md                # 任意。実行後の所見をメモ
```

`HYPERPARAMS.md` には **値そのもの・選んだ根拠・参考文献・期待結果と判断基準**
を必ず書く。値だけ書いても後から振り返れない。

## 起動方法

```bash
# 学習データのソース run dir を指定して起動
bash scripts/run_experiment.sh lora_base_config ./data/runs/2026-06-02_130539
```

挙動:

1. `experiments/<EXP>/data/training_set` にソースの `training_set/` を hardlink で
   コピー（同 disk なら容量ゼロ・即時）。失敗したら通常 `cp -r` で fallback。
2. manifest を 9:1 で split → `train.jsonl` / `eval.jsonl` を生成。
3. `config.yaml` を `train.jsonl` / `eval.jsonl` / `checkpoints/` で埋めた
   `_resolved.yaml` を出して torchrun に渡す。
4. stdout/stderr を `experiments/<EXP>/run.log` にも tee。

実行後は `args.yaml` と `checkpoints/` で結果を保存し、`notes.md` に
所感・eval loss 曲線・best ckpt を書き残す運用。

## チェックポイントの推論用変換

LoRAマージ:

```bash
qsub -v \
LORA_CKPT=/path/to/checkpoint/consolidated/lora.safetensors,\
OUT_WEIGHT=/path/to/merged/consolidated.safetensors \
scripts/merge_lora.pbs
```

Full-FT ZeRO checkpointのexport:

```bash
qsub -v \
STEP_DIR=/path/to/checkpoints/nu_<timestamp>/step_120,\
OUT_DIR=/path/to/exported/step_120_clean \
scripts/export_fullft_checkpoint.pbs
```

どちらも既存の出力先は上書きしない。出力先を省略した場合はPBSジョブIDを
含むディレクトリへ保存する。

## 新しい実験を追加するとき

単発の比較実験(sweepパターンの追加ではなく、独立したフォルダとして残したい
もの)を作るときは、何を検証する実験かが名前だけで分かるようにする
(`exp001`のような連番+短縮名ではなく、`lora_rank64_check` のように内容を表す名前にする):

1. `cp -r lora_base_config <short_descriptive_name>`（例: `lora_rank64_check`）
2. `HYPERPARAMS.md` の冒頭と「期待される結果と判断基準」を書き直す（差分が明確になるよう、
   比較対象の実験との対比を書く）
3. `config.yaml` の変更したいハイパラだけ書き換える
4. `bash scripts/run_experiment.sh <short_descriptive_name> <SRC_RUN_DIR>` で起動

データの再生成が要らない場合は **同じ `<SRC_RUN_DIR>`** を渡して、ハイパラ違いの
比較実験を高速に回す想定。ハイパラを1〜2軸だけ振って多数比較したい場合は、
独立フォルダを都度作るのではなく `scripts/sweep_lora.pbs` /
`scripts/fullft_sweep.pbs` の sweep パターンとして追加する方が管理しやすい
（[sweep_10_patterns.md](sweep_10_patterns.md) 参照）。
