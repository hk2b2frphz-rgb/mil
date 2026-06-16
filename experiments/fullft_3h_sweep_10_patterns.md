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

Run five PBS jobs. Each job generates one 250-dialogue dataset, then runs two
full fine-tuning patterns on that same dataset.

Submit:

```bash
qsub scripts/fullft_sweep_01.pbs
qsub scripts/fullft_sweep_02.pbs
qsub scripts/fullft_sweep_03.pbs
qsub scripts/fullft_sweep_04.pbs
qsub scripts/fullft_sweep_05.pbs
```

Defaults:

- `BASE_EXP=exp100_full_ft`
- `NUM_CASES=250`
- `MLFLOW_EXPERIMENT_NAME=job_fullft_3h`
- `MLFLOW_TRACKING_URI=sqlite:///$PWD/mlruns/mlflow.db`
- `MLFLOW_ARTIFACT_ROOT=file:$PWD/mlruns/artifacts`
- `res=middle` PBS resource, A100 80GB x2 target for full-FT sweep jobs
- `NPROC=2`, `CUDA_VISIBLE_DEVICES=0,1`
- `NU_MOSHI_FT_REPO=$PWD/../moshi-finetune-nu-dialogue`
- `NU_DEEPSPEED_CONFIG=$PWD/configs/deepspeed_zero3_fp16_act_ckpt.json`
  for the default A100 80GB x2 run. This uses fixed learning rate, no warmup
  scheduler, and ZeRO-3 parameter/optimizer partitioning.

Using an existing generated 3h dataset:

```bash
export SRC_RUN_DIR=/path/to/data/runs/3h_dataset
qsub scripts/fullft_sweep_01.pbs
```

`SRC_RUN_DIR` must contain:

```text
training_set/synthetic_moshi_train.jsonl
```

The launcher converts that existing manifest into the nu-dialogue format under
`data/nu_fullft/<RUN_ID>/`, tokenizes audio/text, writes parquet files, and then
starts `accelerate launch`. Set `REFRESH_NU_DATA=1` if you want to rebuild the
converted/tokenized nu data.

If a previous run failed during nu text tokenization, rerun with
`REFRESH_NU_DATA=1` once after pulling. The launcher now patches the
nu-dialogue checkout to avoid `chars[0]["speaker"]` failures on utterance-level
Japanese timestamps.

Direct test without PBS:

```bash
SRC_RUN_DIR=/path/to/data/runs/3h_dataset \
SWEEP_PATTERNS="f01" \
NPROC=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_fullft_sweep_pair.sh
```

| pattern | PBS | change from full-FT 3h baseline | purpose |
|---|---:|---|---|
| `f01` | 01 | `lr=3e-5`, `batch=1`, `micro=8`, `duration=60`, `steps=1200` | A100-safe fixed-LR baseline |
| `f02` | 01 | `duration=80` | longer context, more activation memory |
| `f03` | 02 | `duration=40` | shorter context, lower activation memory |
| `f04` | 02 | `weight_decay=0.01` | weaker regularization |
| `f05` | 03 | `weight_decay=0.2` | stronger regularization |
| `f06` | 03 | `micro=16` | larger effective batch without increasing per-GPU memory |
| `f07` | 04 | `duration=30` | emergency low-memory context |
| `f08` | 04 | `max_norm=0.5` | tighter gradient clipping |
| `f09` | 05 | `steps=800` | shorter exposure |
| `f10` | 05 | `steps=1600` | longer exposure |

Each pattern runs in its own `experiments/_fullft_sweeps/<RUN_ID>_<pattern>/`
directory and logs to MLflow as `<RUN_ID>_<pattern>`. For nu-dialogue full-FT,
MLflow metrics are parsed from the training stdout log, so `train.loss`,
`train.loss.text`, `train.loss.audio`, and learning rates update during training
when `MLFLOW_LIVE_SYNC_INTERVAL` is nonzero.
