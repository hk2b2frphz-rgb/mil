# experiments/

実験フォルダ。`exp<NNN>_<short_name>/` の単位で 1 つの fine-tune ランを管理する。

## 各実験フォルダの構成

```
exp001_lora_baseline/
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
bash scripts/run_experiment.sh exp001_lora_baseline ./data/runs/2026-06-02_130539
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

1. `cp -r exp001_lora_baseline expNNN_<short_name>`
2. `HYPERPARAMS.md` の冒頭と「期待される結果と判断基準」を書き直す（差分が明確になるよう、
   前実験との対比を書く）
3. `config.yaml` の変更したいハイパラだけ書き換える
4. `bash scripts/run_experiment.sh expNNN_<short_name> <SRC_RUN_DIR>` で起動

データの再生成が要らない場合は **同じ `<SRC_RUN_DIR>`** を渡して、ハイパラ違いの
比較実験を高速に回す想定。
