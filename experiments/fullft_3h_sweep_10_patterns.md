# Full fine-tuning 3h 10-pattern sweep

Important repository split:

- Full fine-tuning uses [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune).
- LoRA fine-tuning uses [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).

The full-FT launcher keeps these separate by using `../moshi-finetune-nu-dialogue`
by default. If that directory does not exist on the server, it is cloned
automatically. The Kyutai checkout remains `../moshi-finetune` and is used only
by `scripts/run_experiment.sh` / LoRA sweeps.

This sweep uses the current synthetic data generation pipeline with
`NUM_CASES=250`, matching the existing ~3h data setting used by
`exp002_lora_3h_data`.

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

- `BASE_EXP=exp100_full_ft`
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
- `HP_LR=3e-5` for full-FT by default, matching the nu-dialogue default.
  LoRA sweeps keep the Kyutai example default `2e-6`.

Using an existing generated 3h dataset:

```bash
export SRC_RUN_DIR=/path/to/data/runs/3h_dataset
qsub scripts/fullft_sweep.pbs
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

| pattern | change from full-FT 3h baseline | purpose |
|---|---|---|
| `f01` | `lr=3e-5`, `batch=1`, `micro=8`, `duration=60`, `steps=1200` | A100-safe fixed-LR baseline |
| `f02` | `duration=80` | longer context, more activation memory |
| `f03` | `duration=40` | shorter context, lower activation memory |
| `f04` | `weight_decay=0.01` | weaker regularization |
| `f05` | `weight_decay=0.2` | stronger regularization |
| `f06` | `micro=16` | larger effective batch without increasing per-GPU memory |
| `f07` | `duration=30` | emergency low-memory context |
| `f08` | `max_norm=0.5` | tighter gradient clipping |
| `f09` | `steps=800` | shorter exposure |
| `f10` | `steps=1600` | longer exposure |

Each pattern runs in its own `experiments/_fullft_sweeps/<RUN_ID>_<pattern>/`
directory and logs to MLflow as `<RUN_ID>_<pattern>`. For nu-dialogue full-FT,
MLflow metrics are parsed from the machine-readable `MILTO_METRICS` stdout log.
The important curves are `train.loss`, `train.loss.text`, `train.loss.audio`,
`eval.loss`, `eval.loss.text`, `eval.loss.audio`, `train.accuracy.*`,
`eval.accuracy.*`, and `learning_rate.*`. The launch config, raw training log,
nu config, and dataset health JSON are logged as artifacts. Live sync runs while
training when `MLFLOW_LIVE_SYNC_INTERVAL` is nonzero.
