# Evaluation Plan

This project evaluates Japanese Moshi dialogue models in two separated phases:

1. **Server / GPU phase**
   - Run the model and record responses.
   - Compute deterministic timing metadata.
   - Do not call OpenAI or Azure OpenAI APIs.

2. **Local PC judge phase**
   - Copy compact JSONL logs to the local PC.
   - Run LLM-as-a-judge with OpenAI or Azure OpenAI API keys stored locally.
   - Aggregate scores and compare systems.

## Evaluation Dimensions

| Dimension | What it measures | Primary metric |
|---|---|---|
| Response speed | How quickly the model starts responding | p50/p90 first text/audio latency |
| Real-time behavior | Whether generation keeps up with interaction | wall time, output duration, empty response rate |
| Contextual relevance | Whether the response answers the user's actual utterance | LLM judge 1-5 |
| Topic stability | Whether the response drifts away from the user topic | topic drift rate, LLM judge 1-5 |
| Conversation naturalness | Whether the response sounds like natural Japanese dialogue | LLM judge 1-5, human MOS later |
| Empathy / acknowledgement | Whether feelings are received without dismissal | LLM judge 1-5 |
| Backchannel naturalness | Whether aizuchi is natural, not too much or too little | LLM judge 1-5 |
| Safety / boundaries | Crisis handling, no diagnosis, no unsafe reassurance | unsafe flag, boundary score |
| Specificity | Whether the response is not generic boilerplate | LLM judge 1-5 |

## Metrics to Report First

- `n`
- `empty_response_rate`
- `first_response_latency_sec_p50`
- `first_response_latency_sec_p90`
- `audible_start_after_input_sec_p50`
- `audible_start_after_input_sec_p90`
- `judge.overall_mean`
- `judge.contextual_relevance_mean`
- `judge.conversation_naturalness_mean`
- `judge.topic_stability_mean`
- `judge.backchannel_naturalness_mean`
- `judge.empathy_mean`
- `judge.safety_boundary_mean`
- `topic_drift_rate`
- `unsafe_rate`

## Workflow

### 1. Run responses on the server

```bash
uv run python response_recorder.py \
  --text-file eval_sets/loneliness_support_prompts.txt \
  --seeds 0 \
  --out-dir results/f01_eval
```

### 2. Pack response-recorder output on the server

```bash
python eval/pack_response_recorder_results.py \
  --recorder-dir results/f01_eval \
  --scenarios eval_sets/loneliness_support.jsonl \
  --system-id f01 \
  --out eval_runs/f01_responses.jsonl
```

Copy `eval_runs/f01_responses.jsonl` to the local PC.

### 3. Run LLM-as-a-judge on the local PC only

Install the OpenAI Python SDK on the local PC:

```bash
pip install openai
```

For Azure OpenAI:

```powershell
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT="<deployment-name>"
# Optional for classic dated Azure OpenAI API deployments:
# $env:AZURE_OPENAI_API_VERSION="2024-02-01"
python eval/judge_openai.py \
  --provider azure \
  --input eval_runs/f01_responses.jsonl \
  --out eval_runs/f01_judged.jsonl
```

For OpenAI:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="<model-name>"
python eval/judge_openai.py \
  --provider openai \
  --input eval_runs/f01_responses.jsonl \
  --out eval_runs/f01_judged.jsonl
```

The judge script refuses to run inside common batch-job environments unless
`--allow-server` is explicitly passed.

### 4. Aggregate

```bash
python eval/summarize_eval.py \
  --input eval_runs/f01_judged.jsonl \
  --out eval_runs/f01_summary.json
```

### 5. Pairwise comparison

For model comparison such as `f01` vs `f04`, pack both runs and run:

```bash
python eval/pairwise_openai.py \
  --provider azure \
  --a eval_runs/f01_responses.jsonl \
  --b eval_runs/f04_responses.jsonl \
  --a-name f01 \
  --b-name f04 \
  --out eval_runs/f01_vs_f04_pairwise.jsonl \
  --two-pass
```

Then summarize:

```bash
python eval/summarize_eval.py \
  --pairwise eval_runs/f01_vs_f04_pairwise.jsonl \
  --out eval_runs/f01_vs_f04_summary.json
```
