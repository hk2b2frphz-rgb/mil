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

$env:AZURE_OPENAI_KEY="..."
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

## Comparison baselines

The PBS batch accepts `FDB_SYSTEM=cascade` and `FDB_SYSTEM=speechllm` rows in
the model manifest.  Both run on the same real-dialogue data and write the
same result layout as a Moshi row.  `speechllm` is the Qwen2-Audio
audio-in/text-out baseline; use `SPEECHLLM_MODEL`, `SPEECHLLM_DEVICE_MAP`, and
the other `SPEECHLLM_*` settings in that manifest row.  The worker stays loaded
for the full run, rather than loading the 7B model once per test case.

`FDB_SYSTEM=cascade_synthetic` and `FDB_SYSTEM=speechllm_synthetic` select the
separate Full-Duplex-Bench-JA track.  Do not mix that track's scores with the
real-dialogue table.

## GPT Realtime (local PC only)

GPT Realtime uses an outbound WebSocket and is intentionally not reachable
from `scripts/run_full_duplex_eval_batch.pbs` or any PBS wrapper.  From a local
Bash shell with the repository dependencies available:

```bash
python -m pip install websocket-client
export OPENAI_API_KEY="..."
MODEL_ID=gpt_realtime bash scripts/run_real_gpt_realtime_eval.sh
```

It streams each test WAV to a fresh Realtime API session, keeps the received
audio and transcript in the normal `inference/` layout, then invokes the same
response-rate/latency/MOS and judge-input steps.  The script refuses to run
when PBS, Slurm, or LSF environment variables are present.  Set
`REAL_CASES_PER_TASK=1` for a low-cost smoke run.  `GPT_REALTIME_INPUT_MODE=fast`
is available for API plumbing checks; the default `realtime` mode paces input
at wall-clock speed so response/interruption timing is meaningful.

## Full-duplex Japanese evaluation

For the V100/PBS Full-Duplex-Bench-JA workflow, see
[`docs/full_duplex_evaluation.md`](../docs/full_duplex_evaluation.md).

The server job completes inference and deterministic benchmark evaluation, then
writes `azure_judge_input.jsonl`. Azure judging is a separate local-PC command.

The independent full-duplex **training-data** pipeline is documented in
[`docs/full_duplex_training_data.md`](../docs/full_duplex_training_data.md).
The fixed evaluation cases are not reused as training text.
