# Local cascade / SpeechLLM comparison baselines

Two local, turn-based baselines exist purely to get comparable numbers
against the full-duplex Moshi system, on the same eval_sets and through the
same judge/summarize pipeline:

| System | Architecture | Scripts |
|---|---|---|
| `cascade` | ASR (faster-whisper) -> LLM (Gemma) -> TTS (Qwen3-TTS) | `eval/local_baseline_common.py`, `scripts/gemma_dialogue_worker.py` |
| `speechllm` | SpeechLLM (Qwen2-Audio, audio-in/text-out) -> TTS (Qwen3-TTS) | `eval/local_baseline_common.py`, `scripts/speechllm_worker.py` |

Both are "typical" implementations of their category, not attempts to
reimplement full-duplex behavior: they only ever produce one response,
starting after the entire input utterance has been synthesized (and, for
cascade, transcribed). That is the point of the comparison -- it puts a
number on the latency full-duplex Moshi avoids.

## Why these specific pieces

- **TTS**: reuses `Qwen3TTS`/`initialize_tts`/`synthesize` from
  `eval/build_full_duplex_ja_dataset.py`, the same voice/backend already
  used to build eval input audio, so baseline output audio is acoustically
  comparable and no new TTS integration was needed.
- **LLM (cascade)**: reuses `scripts/gemma_dialogue_worker.py` via subprocess
  in the isolated `gemma_runtime/` uv environment -- the same pattern
  `generate_synthetic_moshi_training_data.py` already uses to keep the main
  Moshi environment's huggingface-hub/transformers pins untouched. No new
  LLM-calling code was written; `eval/local_baseline_common.py`'s
  `GemmaLLM` just wraps the existing worker.
- **ASR (cascade only)**: `faster-whisper`, in-process. It has no
  `transformers` dependency, so it runs in the main project environment
  without conflicting with Moshi.
- **SpeechLLM**: `scripts/speechllm_worker.py` (new), same subprocess
  pattern as the Gemma worker, running Qwen2-Audio-7B-Instruct in
  `gemma_runtime/` (now also depends on `numpy`, added to
  `gemma_runtime/pyproject.toml`).

## Install

```bash
# Main project env: ASR + TTS (TTS is already available if you've run the
# Full-Duplex-Bench-JA dataset builder before).
pip install faster-whisper

# Isolated LLM/SpeechLLM env (same one Gemma dialogue generation already uses).
uv sync --project gemma_runtime
```

Qwen2-Audio-7B-Instruct is a ~8B-parameter model; budget VRAM accordingly.
Use a smaller `--llm-model`/`--asr-model` for a quick local smoke test
(e.g. `--asr-model base`, `--llm-model google/gemma-2-2b-it`).

## loneliness_support quality/latency/safety comparison

Turn-based, so this is the pipeline where the comparison is most
apples-to-apples: same fixed utterances, same judge, same metrics.

```bash
uv run python eval/run_local_baseline_loneliness.py --system cascade \
    --out-dir results/cascade_gemma2b_eval --model-id cascade_gemma2b

python eval/pack_response_recorder_results.py \
    --recorder-dir results/cascade_gemma2b_eval \
    --scenarios eval_sets/loneliness_support.jsonl \
    --system-id cascade_gemma2b \
    --out eval_runs/cascade_gemma2b_responses.jsonl

# Local PC only, per docs/evaluation_plan.md:
python eval/judge_openai.py --provider azure \
    --input eval_runs/cascade_gemma2b_responses.jsonl \
    --out eval_runs/cascade_gemma2b_judged.jsonl
python eval/summarize_eval.py \
    --input eval_runs/cascade_gemma2b_judged.jsonl \
    --out eval_runs/cascade_gemma2b_summary.json
```

Run the same for `--system speechllm --model-id speechllm_qwen2audio`, and
for a Moshi run packed via `pack_response_recorder_results.py` the same way,
then compare `summary.json` files side by side. `first_response_latency_sec`
and `audible_start_after_input_sec` are where the architectural difference
shows up most: cascade/speechllm pay ASR (cascade only) + LLM + TTS wall
time in full before any audio starts, while Moshi streams almost
immediately.

### PBS: cascade, both pipelines in one job

`scripts/run_local_baseline_cascade_eval.pbs` runs `--system cascade` over
*both* pipelines above in one V100 job (queue `xvm_s`, `res=small`),
including building `data/full_duplex_ja_nogreeting` if it doesn't exist yet,
and stops before any judging -- this job never calls OpenAI/Azure, same
rule as `run_full_duplex_eval.pbs`.

```bash
qsub -v CASCADE_MODEL_ID=cascade_gemma2b scripts/run_local_baseline_cascade_eval.pbs
```

See the script's header comment for overrides (`CASCADE_ASR_MODEL`,
`CASCADE_LLM_MODEL`, `SKIP_FULL_DUPLEX=1`, `SKIP_LONELINESS=1`, etc.). It
writes `eval_runs/full_duplex/<CASCADE_MODEL_ID>/azure_judge_input.jsonl`
and `eval_runs/<CASCADE_MODEL_ID>_responses.jsonl`; copy both to the local
PC and judge them per the sections above. There is no equivalent PBS
wrapper for `--system speechllm` yet -- run
`run_local_baseline_loneliness.py`/`run_local_baseline_full_duplex.py`
directly with `--system speechllm`.

## Full-Duplex-Bench-JA (turn-taking) comparison

```bash
# Build the dataset once WITHOUT the Moshi-only opening greeting -- a
# cascade/SpeechLLM baseline was never trained to say it and has no
# streaming behavior to reserve lead-in time for.
uv run python eval/build_full_duplex_ja_dataset.py \
    --scenarios eval_sets/full_duplex_ja/scenarios.jsonl \
    --out-dir data/full_duplex_ja_nogreeting --no-opening-greeting

uv run python eval/run_local_baseline_full_duplex.py --system cascade \
    --dataset-dir data/full_duplex_ja_nogreeting \
    --out-dir eval_runs/full_duplex/cascade_gemma2b/inference \
    --model-id cascade_gemma2b

uv run python eval/evaluate_full_duplex_ja.py \
    --run-dir eval_runs/full_duplex/cascade_gemma2b/inference \
    --out-dir eval_runs/full_duplex/cascade_gemma2b/benchmark_results
```

`evaluate_full_duplex_ja.py` runs completely unchanged -- the baseline
driver writes `output.wav`/`output.json`/`output.meta.json` in the exact
format `run_full_duplex_bench.py` produces for Moshi, with the response
audio placed at `input_duration_sec + (asr + llm + tts wall time)` so the
existing latency/TOR metrics automatically reflect real cascade/SpeechLLM
latency instead of understating it.

**Read the caveats in `eval/run_local_baseline_full_duplex.py`'s module
docstring before trusting every number.** In short: `pause_handling`,
`smooth_turn_taking`, and `user_interruption` faithfully show these
baselines always responding late by roughly the full processing time (that
*is* the comparison, not a bug); `backchannel_frequency`/`jsd` stay
structurally not meaningful for a system that can only produce one
response total; and no `clean_output.wav` is produced for
`user_backchannel`/`talking_to_other`/`background_speech`, so the
clean-vs-noisy delta metrics compare against an empty baseline.

The Azure/LLM-judge dimensions (`judge_full_duplex_azure.py`,
`judge_llmjp_style.py`) stay fully meaningful regardless, since they judge
response content, not turn-taking.
