# 10-pattern sweep

Run five PBS jobs. Each job generates one dataset with the current data
generation pipeline, then runs two training patterns on the same generated data.

Submit:

```bash
qsub scripts/sweep_01.pbs
qsub scripts/sweep_02.pbs
qsub scripts/sweep_03.pbs
qsub scripts/sweep_04.pbs
qsub scripts/sweep_05.pbs
```

Defaults:

- `NUM_CASES=100`
- `MLFLOW_EXPERIMENT_NAME=job_sweep`
- `MLFLOW_TRACKING_URI=file:$PWD/mlruns`
- base experiment config: `exp001_lora_baseline`

| pattern | PBS | change from baseline | purpose |
|---|---:|---|---|
| `h01` | 01 | baseline | reference |
| `h02` | 01 | `lr=5e-6` | check faster adaptation / instability |
| `h03` | 02 | `lr=1e-6` | check under-training |
| `h04` | 02 | `lora.rank=64` | more adapter capacity |
| `h05` | 03 | `lora.rank=16` | smaller adapter / overfit check |
| `h06` | 03 | `lora.scaling=1.0` | lower LoRA update scale |
| `h07` | 04 | `lora.scaling=4.0` | higher LoRA update scale |
| `h08` | 04 | `batch_size=4`, `num_microbatches=2` | same effective batch, lower per-forward memory |
| `h09` | 05 | `pct_start=0.10` | longer warmup |
| `h10` | 05 | `weight_decay=0.01` | weaker regularization |

Each MLflow run name is `<RUN_ID>_<pattern>`.
Each pattern runs in its own `experiments/_sweeps/<RUN_ID>_<pattern>/`
directory so the five PBS jobs can run concurrently without sharing `_resolved.yaml`
or experiment data directories.
