# Full-Duplex-Bench Japanese evaluation

This evaluation is based on Full-Duplex-Bench v1/v1.5. Only changes required
for English-to-Japanese adaptation are applied. The upstream implementation is
pinned to commit
[`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`](https://github.com/DanielLin94144/Full-Duplex-Bench/commit/3e799c45a045256f47d5f1c9cda90157e2d2ec9e).

The benchmark covers pause handling, backchannel behavior, smooth turn taking,
user interruption, user backchannel overlap, speech directed to another
person, and background speech. Deterministic metrics follow the upstream
definitions. Semantic action classification and response-quality ratings
remain delegated to the Azure judge.

These adapted scores are suitable for controlled comparisons among Japanese
models evaluated by this repository. They are not directly comparable to the
English official leaderboard because the language, tokenizer/counting unit,
input speech, and backchannel ground truth differ.

## Deterministic metric definitions

All threshold names and values below are copied from the pinned upstream
evaluation.

| Metric or constant | Definition |
|---|---|
| `turn_duration_threshold` | `1.0` second |
| `turn_num_words_threshold` | `3` counting units |
| `window_size` | `0.2` second for the backchannel occurrence distribution |
| `time_threshold` | `3.0` seconds; a longer backchannel speech segment is a takeover |
| `epsilon` | `1e-10` for distribution normalization |
| TOR | If no aligned output chunks exist, `0`. Otherwise, duration is the last chunk end minus the first chunk start. If duration is below `1.0` second, TOR is `0` for at most 3 chunks and `1` otherwise. For duration at least `1.0` second, TOR is `1`. |
| Response latency | When TOR is `1`, first output chunk start minus the relevant input end, clamped to zero. Otherwise the latency is `null`. |
| Backchannel frequency | Number of non-takeover Silero VAD speech segments divided by total output audio seconds. |
| Backchannel JSD | Jensen-Shannon distance between the normalized `0.2`-second prediction occurrence distribution and the speaker's GT distribution. If no prediction exists, JSD is `1`. If Japanese GT is unavailable, JSD is `null` and excluded from numeric aggregation. |

Task-specific deterministic outputs:

| Task | Main deterministic metrics |
|---|---|
| `pause_handling` | `TOR` over Moshi chunks overlapping the pause. `pause_overlap_sec` remains supplementary. |
| `smooth_turn_taking` | `TOR` and `response_latency_sec`, measured after the final user utterance. |
| `user_interruption` | `TOR` and `response_latency_after_interrupt_sec`, measured after the interruption event ends. Stop timing remains supplementary; semantic quality remains an Azure rating. |
| `backchannel` | `TOR`, `backchannel_frequency`, and optional `jsd`. Speech segments come from Silero VAD. |
| `user_backchannel`, `talking_to_other`, `background_speech` | Timing-only measurements including `stop_latency_sec`, overlap speech, and post-overlap speech. Respond/Resume action classification remains with the Azure judge. |

For backchannel evaluation, each Silero segment is checked in time order. A
segment longer than `3.0` seconds is a takeover. Otherwise, overlapping Moshi
text tokens are counted. More than 3 tokens is a takeover; below `1.0` second,
at most 2 tokens is a backchannel and 3 tokens is a takeover; a segment of at
least `1.0` second is a takeover. Collection stops at the first takeover.

## English-to-Japanese adaptation points

No benchmark behavior is intentionally changed beyond the following required
language adaptations.

| Upstream English evaluation | Japanese adaptation |
|---|---|
| Space-delimited English ASR words are counted. | Each time-aligned Moshi Japanese text piece in `output.json` is one counting unit. Thresholds remain 3 and 2. The unit therefore changes from ASR word to Moshi text token. |
| External English ASR, including CrisperWhisper/parakeet, supplies transcripts. | Moshi's own Japanese text-token stream, written by `run_full_duplex_bench.py`, supplies transcripts and timestamps. |
| `icc_gt_distribution.json[spk]` supplies the English human-annotated backchannel timing distribution. | No Japanese GT distribution currently exists. JSD is optional and returns `null` when GT or the case speaker key is absent. Activate it with `--backchannel-gt PATH` after a Japanese speaker-keyed GT JSON is prepared and scenario metadata contains `spk` or `speaker`. |
| Silero VAD detects output speech. | Silero VAD remains the primary, language-agnostic detector. The existing energy VAD is used automatically with a one-line warning only when Silero cannot be imported, loaded, or run in an offline environment. |
| English TTS creates benchmark input audio. | Qwen3-TTS (same as training pipeline). Auto-falls back to `pyopenjtalk` when GPU/qwen-tts is unavailable. |

Matching train/eval TTS reduces acoustic mismatch between training and benchmark input speech.

## Server: V100 PBS evaluation

The PBS job runs inference, deterministic timing evaluation, and packing for
later Azure evaluation. It never calls OpenAI or Azure.

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

Exported full fine-tuning model:

```bash
qsub -v \
MODEL_ID=full_f01,\
MODEL_WEIGHT=/path/to/exported/model.safetensors,\
MODEL_CONFIG=/path/to/exported/moshi_lm_kwargs.json \
scripts/run_full_duplex_eval.pbs
```

The queue is `xvn_s`; fp16 is the default for V100. If `silero-vad` or its
model cannot be loaded on the offline node, evaluation logs one fallback line
and uses energy VAD.

Outputs:

```text
eval_runs/full_duplex/<RUN_ID>/
|-- run.log
|-- inference/
|   |-- run_config.json
|   `-- <task>/<case>/seed_<N>/
|-- benchmark_results/
|   |-- per_case.jsonl
|   `-- summary.json
`-- azure_judge_input.jsonl
```

The fixed evaluation utterances are not training data. The independent
Gemma/Qwen3-TTS training pipeline is documented in
[`full_duplex_training_data.md`](full_duplex_training_data.md).

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
