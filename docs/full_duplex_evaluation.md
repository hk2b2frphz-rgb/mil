# Full-Duplex-Bench Japanese evaluation

This repository uses a Japanese adaptation of the static Full-Duplex-Bench
v1/v1.5 task structure. The upstream reference is pinned to commit
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`.

The Japanese profile preserves the benchmark dimensions:

- pause handling
- backchannel behavior
- smooth turn taking
- user interruption
- user backchannel overlap
- speech directed to another person
- background speech

The inputs are Japanese synthetic speech generated locally with
`pyopenjtalk`. Output transcription uses Moshi's own Japanese text-token stream,
so the server run does not depend on an English ASR model.

These adapted scores are suitable for comparing models in this project. They
are not directly comparable with the official English leaderboard.

The fixed evaluation utterances are not training data. The independent
Gemma/Qwen3-TTS training pipeline, including real stereo overlap generation, is
documented in [`full_duplex_training_data.md`](full_duplex_training_data.md).

## Server: V100 PBS evaluation

The PBS job runs inference, deterministic timing/behavior evaluation, and
packing for later Azure evaluation. It never calls OpenAI or Azure.

Base model:

```bash
qsub -v MODEL_ID=base scripts/run_full_duplex_eval.pbs
```

Merged LoRA:

```bash
qsub -v \
MODEL_ID=lora_h01,\
MODEL_WEIGHT=/path/to/consolidated.safetensors,\
MODEL_CONFIG=/path/to/moshi_lm_kwargs.json \
scripts/run_full_duplex_eval.pbs
```

Exported full-FT:

```bash
qsub -v \
MODEL_ID=full_f01,\
MODEL_WEIGHT=/path/to/exported/model.safetensors,\
MODEL_CONFIG=/path/to/exported/moshi_lm_kwargs.json \
scripts/run_full_duplex_eval.pbs
```

The queue is `xvn_s`; fp16 is the default for V100.

Outputs:

```text
eval_runs/full_duplex/<RUN_ID>/
├── run.log
├── inference/
│   ├── run_config.json
│   └── <task>/<case>/seed_<N>/
├── benchmark_results/
│   ├── per_case.jsonl
│   └── summary.json
└── azure_judge_input.jsonl
```

## Local PC: Azure content evaluation

Copy only `azure_judge_input.jsonl` to the local PC, then run:

```powershell
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT="<deployment>"

python eval/judge_full_duplex_azure.py `
  --provider azure `
  --input eval_runs/full_duplex/<RUN_ID>/azure_judge_input.jsonl `
  --out eval_runs/full_duplex/<RUN_ID>/azure_judged.jsonl

python eval/summarize_eval.py `
  --input eval_runs/full_duplex/<RUN_ID>/azure_judged.jsonl `
  --out eval_runs/full_duplex/<RUN_ID>/azure_summary.json
```

The judge refuses to run under PBS/Slurm/LSF unless `--allow-server` is
explicitly supplied.
