# Local Evaluation Tools

The OpenAI or Azure OpenAI API must be called from the local PC, not from the
training server.

Server side:

```bash
python eval/pack_response_recorder_results.py \
  --recorder-dir results/f01_eval \
  --scenarios eval_sets/loneliness_support.jsonl \
  --system-id f01 \
  --out eval_runs/f01_responses.jsonl
```

Local PC side:

```powershell
pip install openai

$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT="<deployment-name>"
# Optional for classic dated Azure OpenAI API deployments:
# $env:AZURE_OPENAI_API_VERSION="2024-02-01"

python eval/judge_openai.py \
  --provider azure \
  --input eval_runs/f01_responses.jsonl \
  --out eval_runs/f01_judged.jsonl

python eval/summarize_eval.py \
  --input eval_runs/f01_judged.jsonl \
  --out eval_runs/f01_summary.json
```

The judge script exits if it sees `PBS_JOBID`, `SLURM_JOB_ID`, or `LSB_JOBID`,
unless `--allow-server` is passed.

## Full-duplex Japanese evaluation

For the V100/PBS Full-Duplex-Bench-JA workflow, see
[`docs/full_duplex_evaluation.md`](../docs/full_duplex_evaluation.md).

The server job completes inference and deterministic benchmark evaluation, then
writes `azure_judge_input.jsonl`. Azure judging is a separate local-PC command.

The independent full-duplex **training-data** pipeline is documented in
[`docs/full_duplex_training_data.md`](../docs/full_duplex_training_data.md).
The fixed evaluation cases are not reused as training text.
