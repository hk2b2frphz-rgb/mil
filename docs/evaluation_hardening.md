# Evaluation hardening

## Dataset and batch reproducibility

`Full-Duplex-Bench-JA` is a controlled Japanese adaptation of the pinned v1/v1.5 protocol, not an English leaderboard submission. Every generated dataset manifest records the rendering configuration, the SHA-256 of `scenarios.jsonl`, and an input fingerprint. The runner verifies them before reuse and rebuilds a stale dataset.

In a parallel mixed batch, all first-time input-TTS builds share a global lock. This prevents greeting and no-greeting variants from consuming the same GPU allocation concurrently.

## v1.5 acoustic adaptation

The server writes `benchmark_results/acoustic_adaptation.json` for every paired overlap trial. It contains WPM, pitch, intensity, abrupt-cutoff count, optional UTMOSv2, paired t-tests, and Holm-adjusted p values for pre/post and clean/post comparisons. UTMOSv2 is installed from the official GitHub implementation pinned in `pyproject.toml` (it is not available as a normal PyPI distribution). The normal runner enables UTMOSv2; use `FDB_WITH_UTMOS=0` only when the optional model weights or runtime cannot be installed.

After local semantic judging, filter acoustic significance tests by action class:

```powershell
python eval/summarize_full_duplex_adaptation.py `
  --adaptation eval_runs/full_duplex/<RUN_ID>/benchmark_results/acoustic_adaptation.json `
  --judged eval_runs/full_duplex/<RUN_ID>/azure_judged.jsonl `
  --action C_RESPOND `
  --out eval_runs/full_duplex/<RUN_ID>/acoustic_adaptation_c_respond.json
```

`C_RESPOND`, `C_RESUME`, `C_UNCERTAIN_HANDLING`, and `C_UNKNOWN` are mandatory for each overlap task.

## Durable local judging

All synchronous judges accept `--resume`. Each successful API call is flushed to JSONL before the next request, and completed IDs are skipped when resuming. Required score/flag fields are validated instead of being silently omitted. Pairwise judging rejects duplicate or unequal case/seed sets.

## OpenAI / Azure Batch API

The Batch API workflow is separate from the synchronous judge and supports both providers. For Azure, deploy a `Global-Batch` model and set `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, and the deployment name in `--model`.

```powershell
python eval/judge_openai_batch.py --action submit `
  --input eval_runs/f01_responses.jsonl `
  --out eval_runs/f01_judged.jsonl `
  --state eval_runs/f01_openai_batch.json `
  --model <openai-model>

python eval/judge_openai_batch.py --action status `
  --state eval_runs/f01_openai_batch.json

python eval/judge_openai_batch.py --action collect `
  --state eval_runs/f01_openai_batch.json `
  --out eval_runs/f01_judged.jsonl
```

For Azure, add `--provider azure` to all three commands and pass the Azure deployment name to `--model` during submit.

The state file retains the provider, batch and input-file IDs, plus the request-to-row mapping. Collection writes successful rows incrementally and saves failures as `<out>.batch_errors.json`. Batch API submission remains local-PC only.
