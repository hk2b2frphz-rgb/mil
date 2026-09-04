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

Before adding either row to the batch manifest, run its one-case smoke test in
an interactive GPU allocation.  It checks the correct isolated worker import,
executes audio input through to `output.wav`, writes the normal deterministic
summary, and skips only optional UTMOS/backchannel work:

```bash
bash scripts/debug_real_baseline.sh cascade
bash scripts/debug_real_baseline.sh speechllm
```

The debug output is under `eval_runs/debug/`.  Use `DEBUG_CASES_PER_TASK=2`
to test more cases; batch evaluation restores the normal output alignment,
UTMOS, and backchannel settings.

`FDB_SYSTEM=cascade_synthetic` and `FDB_SYSTEM=speechllm_synthetic` select the
separate Full-Duplex-Bench-JA track.  Do not mix that track's scores with the
real-dialogue table.

### Qwen2.5-Omni on a 32GB V100

Use the native audio-input/audio-output 3B checkpoint as a separate row:

```text
qwen25_omni_3b||||FDB_SYSTEM=qwen25_omni;QWEN25_OMNI_MODEL=Qwen/Qwen2.5-Omni-3B;QWEN25_OMNI_DTYPE=float16||qwen25_omni_3b
```

It does **not** use the cascade TTS: `Qwen2.5-Omni-3B` provides the response
waveform itself (24 kHz), which the evaluator resamples and writes as the
normal `output.wav`.  On V100, retain `float16`; do not select bf16.  Before
the first PBS run, refresh the isolated worker environment once:

```bash
uv sync --project gemma_runtime
```

Start with `REAL_CASES_PER_TASK=1`, then use the normal batch command.  The
7B Omni checkpoint is intentionally not the default: its documented minimum
memory is already 31.11 GB for a short 15-second example before the practical
runtime overhead, whereas the 3B checkpoint's corresponding figure is 18.38
GB.

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
