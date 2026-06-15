# Full fine-tuning 3h 10-pattern sweep

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
- A100 80GB x2 target for PBS sweep jobs

| pattern | PBS | change from full-FT 3h baseline | purpose |
|---|---:|---|---|
| `f01` | 01 | `lr=5e-7`, `batch=1`, `micro=8`, `steps=1200` | 3h low-memory baseline |
| `f02` | 01 | `lr=1e-6` | faster adaptation / forgetting risk |
| `f03` | 02 | `lr=2e-7` | conservative update |
| `f04` | 02 | `weight_decay=0.01` | weaker regularization |
| `f05` | 03 | `weight_decay=0.2` | stronger regularization |
| `f06` | 03 | `pct_start=0.10` | longer warmup |
| `f07` | 04 | `duration_sec=80` | shorter sequence, lower activation memory |
| `f08` | 04 | `max_norm=0.5` | tighter gradient clipping |
| `f09` | 05 | `steps=800` | shorter exposure |
| `f10` | 05 | `steps=1600` | longer exposure |

Each pattern runs in its own `experiments/_fullft_sweeps/<RUN_ID>_<pattern>/`
directory and logs to MLflow as `<RUN_ID>_<pattern>`.
