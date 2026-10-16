# Full fine-tuning 3h 10-pattern sweep

Important repository split:

- Full fine-tuning uses [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune).
- LoRA fine-tuning uses [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).

The full-FT launcher keeps these separate by using `../moshi-finetune-nu-dialogue`
by default. If that directory does not exist on the server, it is cloned
automatically. The Kyutai checkout remains `../moshi-finetune` and is used only
by `scripts/run_experiment.sh` / LoRA sweeps.

This sweep uses the current synthetic data generation pipeline with
`NUM_CASES=250` (~3h of dialogue), the data-volume setting adopted as the
default across the LoRA/full-FT sweeps after an early one-off comparison
showed 250 dialogues improved eval loss over the ~100-dialogue baseline.

Use one PBS entrypoint and select full fine-tuning patterns with
`SWEEP_PATTERNS`. The default PBS run executes `f01` only; pass the patterns you
want to compare for hyperparameter tuning.

Submit:

```bash
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=f01 scripts/fullft_sweep.pbs
```

For a small tuning comparison, submit only the conditions you want:

```bash
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=f01,f04 scripts/fullft_sweep.pbs
```

Defaults:

- `BASE_EXP=fullft_base_config`
- `NUM_CASES=250`
- `MLFLOW_EXPERIMENT_NAME=job_fullft_hp`
- `MLFLOW_TRACKING_URI=sqlite:///$PWD/mlruns/mlflow.db`
- `MLFLOW_ARTIFACT_ROOT=file:$PWD/mlruns/artifacts`
- `res=middle` PBS resource, A100 80GB x2 target for full-FT sweep jobs
- `NPROC=2`, `CUDA_VISIBLE_DEVICES=0,1`
- `NU_MOSHI_FT_REPO=$PWD/../moshi-finetune-nu-dialogue`
- `NU_DEEPSPEED_CONFIG=$PWD/configs/deepspeed_zero3_fp16_warmlr_act_ckpt.json`
  for the default A100 80GB x2 run. This uses ZeRO-3
  parameter/optimizer partitioning and a short warmup to the fixed LR.
- `HP_LR=1e-5` for the full-FT baseline (the nu-dialogue `3e-5` default
  overfits this data volume almost immediately; it is kept as the `lr_3e-5`
  reference pattern). LoRA sweeps keep the Kyutai example default `2e-6`.

Using an existing generated 3h dataset:

```bash
export SRC_RUN_DIR=/path/to/data/runs/3h_dataset
qsub scripts/fullft_sweep.pbs
```

Learning-rate-only sweep:

```bash
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset scripts/fullft_lr_sweep.pbs

# Shorter and safer for 12h walltime: submit one LR per PBS job.
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=lr_2e-5 scripts/fullft_lr_sweep.pbs
```

`SRC_RUN_DIR` must contain:

```text
training_set/synthetic_moshi_train.jsonl
```

The launcher converts that existing manifest into the nu-dialogue format under
`data/nu_fullft/<RUN_ID>/`, tokenizes audio/text, writes parquet files, and then
starts `accelerate launch`. Set `REFRESH_NU_DATA=1` if you want to rebuild the
converted/tokenized nu data.

After writing parquet, the launcher checks train/eval row counts, chunk counts
after `min_length`/`max_length`, and non-padding main-speaker text labels. If
eval has no valid chunks or labels, training stops before producing misleading
NaN eval metrics. If a previous run failed during nu text tokenization, rerun
with `REFRESH_NU_DATA=1` once after pulling.

During nu data preparation, transcript text is normalized before tokenization:
ellipsis variants (`...`, `．．．`, `・・・`) are unified to `…`, ASCII sentence
punctuation is converted to Japanese punctuation where appropriate, and spaces
around punctuation are removed. The before/after examples and rule counts are
written to `data/nu_fullft/<RUN_ID>/manifest.json` under `text_normalization`.

Direct test without PBS:

```bash
SRC_RUN_DIR=/path/to/data/runs/3h_dataset \
SWEEP_PATTERNS="f01" \
NPROC=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_fullft_sweep_pair.sh
```

The full-FT baseline is data-matched and **epoch-denominated**: the schedule is
specified in epochs and converted to steps by the runner from the real
train-chunk count (`steps_per_epoch = ceil(train_count / global_batch)`). The
runner logs the conversion (`train_count`, `steps_per_epoch`, resulting
`max_steps`/`eval_steps`/`save_steps`/`warmup_steps`).

Baseline: `lr=1e-5`, `batch=1`, `micro=8`, `duration=60`, `max_epochs=12`,
`warmup=1 epoch`, `eval=every 0.5 epoch`, `ckpt=every 1 epoch`. At ~250 cases
(train~225, global batch 16 -> ~15 steps/epoch) this is roughly `max_steps~180`,
`warmup~15`, `eval_steps~8`, `save_steps~15` -> ~12 checkpoints and ~24 eval
points. LR is `1e-5` (not the nu-dialogue `3e-5` default), which overfits this
data volume almost immediately; the `3e-5` reference lives in `lr_3e-5`.

| pattern | change from full-FT baseline | purpose |
|---|---|---|
| `f01` | (baseline) | data-matched fixed-LR baseline |
| `f02` | `duration=80` | longer context, more activation memory |
| `f03` | `duration=40` | shorter context, lower activation memory |
| `f04` | `weight_decay=0.01` | weaker regularization |
| `f05` | `weight_decay=0.2` | stronger regularization |
| `f06` | `micro=16` | larger effective batch without increasing per-GPU memory |
| `f07` | `duration=30` | emergency low-memory context |
| `f08` | `max_norm=0.5` | tighter gradient clipping |
| `f09` | `max_epochs=8` | shorter exposure |
| `f10` | `max_epochs=20` | longer exposure |
| `lr_3e-5` | `lr=3e-5` | nu-dialogue default LR reference |
| `lr_2e-5` | `lr=2e-5` | slightly lower LR |
| `lr_1e-5` | `lr=1e-5` | lower LR |
| `lr_5e-6` | `lr=5e-6` | conservative LR |

Epoch knobs can also be overridden per run, e.g. to capture the very early
overfitting region at higher resolution:

```bash
HP_MAX_EPOCHS=12 HP_EVAL_EVERY_EPOCH=0.25 HP_CKPT_EVERY_EPOCH=0.5 \
SRC_RUN_DIR=/path/to/data/runs/3h_dataset SWEEP_PATTERNS=f01 \
NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_fullft_sweep_pair.sh
```

Each pattern runs in its own `experiments/_fullft_sweeps/<RUN_ID>_<pattern>/`
directory and logs to MLflow as `<RUN_ID>_<pattern>`. For nu-dialogue full-FT,
MLflow metrics are parsed from the machine-readable `MILTO_METRICS` stdout log.
The important curves are `train.loss`, `train.loss.text`, `train.loss.audio`,
`eval.loss`, `eval.loss.text`, `eval.loss.audio`, `train.accuracy.*`,
`eval.accuracy.*`, and `learning_rate.*`. The launch config, raw training log,
nu config, and dataset health JSON are logged as artifacts. Live sync runs while
training when `MLFLOW_LIVE_SYNC_INTERVAL` is nonzero.

#### A100 x2 OOM対策

whole-utterance TTS で作った対話は1件あたりの音声が長くなりやすく、
`duration` の上限付近のシーケンスが増えるとアクティベーションメモリの
ピークが上がり、A100 80GB x2 でも OOM することがある。対策として:

- `configs/deepspeed_zero3_fp16_warmlr_act_ckpt.json` に ZeRO-3 の
  `offload_optimizer`（Adam の optimizer state を CPU にオフロード）を
  デフォルトで有効化し、`sub_group_size` / `stage3_max_live_parameters` /
  `stage3_max_reuse_distance` を `1e9` から `5e8` に下げてパラメータの
  ワーキングセットも縮小した。速度は多少落ちるが GPU メモリを大きく削減できる。
- それでも OOM する場合は、パラメータも CPU オフロードする
  `configs/deepspeed_zero3_fp16_warmlr_act_ckpt_full_offload.json` に切り替える
  （さらに遅くなるが、より安全）:

  ```bash
  NU_DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero3_fp16_warmlr_act_ckpt_full_offload.json" \
  SRC_RUN_DIR=/path/to/data/runs/... SWEEP_PATTERNS=f01 \
  NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_fullft_sweep_pair.sh
  ```

- それでも厳しい場合は `f07`（`duration=30`）などの低メモリ・パターンに
  切り替えるか、`HP_NUM_MICROBATCHES` を増やして per-microbatch のメモリを
  さらに下げる。
