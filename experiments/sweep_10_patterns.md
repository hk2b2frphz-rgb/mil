# 10-pattern sweep

Use one PBS entrypoint and select LoRA patterns with `SWEEP_PATTERNS`. The
launcher either reuses `SRC_RUN_DIR` or generates one 250-dialogue dataset, then
runs the selected LoRA patterns on the same data.

Submit:

```bash
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=h01,h02 scripts/sweep_lora.pbs
```

To run all ten patterns, submit one or more jobs and split the comma-separated
pattern list as needed:

```bash
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=h01,h02,h03,h04,h05 scripts/sweep_lora.pbs
qsub -v SRC_RUN_DIR=/path/to/data/runs/3h_dataset,SWEEP_PATTERNS=h06,h07,h08,h09,h10 scripts/sweep_lora.pbs
```

Defaults:

- `NUM_CASES=250`
- LoRA baseline exposure: `max_steps=1200`, `ckpt_freq=120`, `eval_freq=60`
- `MLFLOW_EXPERIMENT_NAME=job_lora_hp`
- `MLFLOW_TRACKING_URI=sqlite:///$PWD/mlruns/mlflow.db`
- `MLFLOW_ARTIFACT_ROOT=file:$PWD/mlruns/artifacts`
- base experiment config: `exp001_lora_baseline`

| pattern | change from baseline | purpose |
|---|---|---|
| `h01` | baseline | reference |
| `h02` | `lr=5e-6` | check faster adaptation / instability |
| `h03` | `lr=1e-6` | check under-training |
| `h04` | `lora.rank=64` | more adapter capacity |
| `h05` | `lora.rank=16` | smaller adapter / overfit check |
| `h06` | `lora.scaling=1.0` | lower LoRA update scale |
| `h07` | `lora.scaling=4.0` | higher LoRA update scale |
| `h08` | `batch_size=4`, `num_microbatches=2` | same effective batch, lower per-forward memory |
| `h09` | `pct_start=0.10` | longer warmup |
| `h10` | `weight_decay=0.01` | weaker regularization |

Each MLflow run name is `<RUN_ID>_<pattern>`.
Each pattern runs in its own `experiments/_sweeps/<RUN_ID>_<pattern>/`
directory, so multiple PBS jobs can run concurrently without sharing
`_resolved.yaml` or experiment data directories.
