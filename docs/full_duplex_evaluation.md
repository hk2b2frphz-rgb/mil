# Full-Duplex-Bench Japanese evaluation

For reproducibility hardening, acoustic-adaptation artifacts, and durable local judging, see [`evaluation_hardening.md`](evaluation_hardening.md).

This is a Japanese adaptation of the pinned Full-Duplex-Bench v1.5 static
overlap protocol. The upstream implementation is pinned to commit
[`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`](https://github.com/DanielLin94144/Full-Duplex-Bench/commit/3e799c45a045256f47d5f1c9cda90157e2d2ec9e).

The benchmark covers pause handling, backchannel behavior, smooth turn taking,
user interruption, user backchannel overlap, speech directed to another
person, and background speech. Deterministic metrics follow the upstream
definitions. Semantic action classification and response-quality ratings
remain delegated to the Azure judge.

These adapted scores are suitable for controlled comparisons among Japanese
models evaluated by this repository. They are not directly comparable to the
English official leaderboard because the language, tokenizer/counting unit,
input speech, and backchannel ground truth differ. A Moshi run is reported as
protocol-conformant only when (1) its dataset manifest declares the v1.5
static-overlap profile and (2) every overlap trial actually begins while the
model is speaking. `run_full_duplex_eval.sh` enforces both conditions. On a
failure it still writes `summary.json`, but aggregates metrics only from
successful trials and records failed counts, rates, reasons, task breakdowns,
and trial IDs under `evaluation`. The process remains non-zero and
`protocol.timing_conformance` remains `not_established`, so the partial result
cannot be mistaken for a conformant run.

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
| Backchannel frequency | Number of Silero VAD speech segments no longer than `3.0` seconds divided by total output audio seconds. Matching upstream, a short segment still counts here even when its token count marks it as a takeover for TOR. |
| Backchannel JSD | Jensen-Shannon distance between the normalized `0.2`-second prediction occurrence distribution and the speaker's GT distribution. If no prediction exists, JSD is `1`. If Japanese GT is unavailable, JSD is `null` and excluded from numeric aggregation. See [Backchannel JSD ground truth](#backchannel-jsd-ground-truth) below. |

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
least `1.0` second is a takeover. Collection stops only when a segment exceeds
`3.0` seconds (upstream behavior); shorter token-based takeovers do not stop
collection and still enter the frequency/JSD prediction list.

One deliberate deviation from the pinned upstream loop: upstream overwrites
`TOR` on every segment, so a takeover followed by a later clean backchannel
reports `TOR=0` (last segment wins). This repository takes the maximum across
segments instead — any takeover anywhere in the output yields `TOR=1`, which
is the metric's stated intent. Scores here are therefore equal to or stricter
than a literal upstream run.

## Backchannel JSD ground truth

`eval_sets/full_duplex_ja/backchannel_gt.json` (built by
`eval/build_backchannel_gt.py`) lets `evaluate_full_duplex_ja.py
--backchannel-gt` compute `jsd` instead of always returning `null`. Regenerate
it with:

```bash
python eval/build_backchannel_gt.py --out eval_sets/full_duplex_ja/backchannel_gt.json
```

**This is not a human-annotated corpus.** No Japanese recording of real
backchannel timing exists for this project. The script instead mines
`event=="model_backchannel"` turns (`start_after_previous_start_sec`) across
every generated training-dialogue JSONL under `data/`, `eval_runs/`, and
`tests/fixtures/`, and bins the offsets into `0.2`-second windows the same way
`evaluate_full_duplex_ja.py` bins model output. `jsd` computed against it
measures *distance from this project's own aizuchi placement design*, not
distance from authentic human backchannel timing. The output JSON records
`_meta.n_backchannel_events` so a thin sample is visible at a glance.
Regenerate whenever the aizuchi placement design changes (see the "aizuchi"
and "energy valley detection" commits) or when more training dialogues have
been generated. Replace it outright if a real Japanese backchannel-timing
corpus ever becomes available.

`run_full_duplex_eval.sh` passes `--backchannel-gt` automatically when
`eval_sets/full_duplex_ja/backchannel_gt.json` exists (override with
`FDB_BACKCHANNEL_GT`). The GT lookup key is the scenario's `tts_speaker`
(`Ono_Anna` by default); `build_full_duplex_ja_dataset.py` also records this
as `speaker` in `metadata.json` for upstream-style compatibility.

`background_speech` announcements use a distinct speaker (`Uncle_Fu` by
default), are present only in `input.wav`, and are replaced by equal-duration
silence in `clean_input.wav`. The v2 profile follows the v1.5 paper's stated
background treatment: -15 dB level, 3 kHz low-pass filtering, and a 100 ms
echo at -10 dB; it also applies deterministic static compression (4:1 above
0.1 linear amplitude) to realize the paper's qualitative “dynamic range
compression” instruction. The paper does not publish compressor coefficients,
so this local coefficient is recorded in `manifest.json` rather than presented
as an upstream-exact value.

## Fixed opening greeting

The model is trained (`scripts/generate_qwen3_tts_data.py`
`OPENING_GREETING_TEXT`, see
[`full_duplex_training_data.md`](full_duplex_training_data.md)) to always
open a session by saying a fixed line itself, e.g. the default:
"もしもし、こちら孤独孤立相談窓口になります。"
`build_full_duplex_ja_dataset.py` mirrors this by default
(`--opening-greeting`, matching text; `--no-opening-greeting` to disable):

1. It synthesizes the greeting once with the same backend/speaker as the
   rest of the dataset purely to measure how long it takes to say.
2. Every scenario's input audio gets that many seconds of silence plus
   `--opening-greeting-gap-sec` (default `0.4`, matching training's
   `--gap-sec`) prepended before the scripted timeline starts, so the model
   has room to say the greeting without the scripted user speech
   immediately talking over it or being mistaken for an interruption test.
3. `metadata.json` records `opening_greeting: {text, duration_sec, gap_sec}`.

No changes to deterministic metric definitions were needed: every metric in
`evaluate_full_duplex_ja.py` is anchored to specific timeline events (pause,
interruption, overlap intervals) or to `user_segments` end times, not to
absolute recording start, so shifting everything later by the lead-in is
transparent to them.

`evaluate_full_duplex_ja.py` additionally checks, for every case, whether the
model actually said the greeting: it takes the assistant's own timestamped
text chunks with a start time before `duration_sec + 1.0`s, normalizes
whitespace, and compares against the expected text with
`difflib.SequenceMatcher`. This adds `greeting_similarity` (0-1) and
`greeting_matched` (bool, similarity >= `0.6`) to `metrics` for every case
(aggregated the same way as any other metric in `summary.json`), and a full
`opening_greeting_check` (expected vs. actual text) to `per_case.jsonl` for
manual inspection. A low `greeting_matched` rate is a regression signal that
fine-tuning eroded the memorized opening line, independent of the Azure/LLM
judge.

The same check (`eval/greeting_check.py`, shared by both scripts) also runs
during inference itself: `run_full_duplex_bench.py` and
`run_local_baseline_full_duplex.py` print one `[fdb] greeting <task>/<case>
seed=<N>: OK|MISMATCH similarity=<0-1>` line per trial as soon as that
trial's output is generated, so `run.log` shows whether the greeting was
said without waiting for the separate `evaluate_full_duplex_ja.py` step
(or reading `per_case.jsonl` at all). No line is printed for a case with no
`opening_greeting` metadata (e.g. `FDB_OPENING_GREETING=0` runs).

Toggle via `run_full_duplex_eval.sh`'s `FDB_OPENING_GREETING=1` (default) /
`FDB_OPENING_GREETING_GAP_SEC=0.4`. **Set `FDB_OPENING_GREETING=0` when
evaluating base Moshi or llm-jp baselines** -- they were never trained to say
this line, so there is nothing to wait for and reserving lead-in time for it
would just be dead air in front of their real first response. The default
`FDB_DATA_DIR` encodes this flag (`data/full_duplex_ja_v2_greeting` vs.
`data/full_duplex_ja_v2_nogreeting`), so toggling it always builds/reuses the
correct dataset variant instead of silently reusing one built for the other
setting.

The greeting synthesis itself is cached on disk under
`data/.cache/opening_greeting/` (override with
`FDB_OPENING_GREETING_CACHE_DIR` / `--opening-greeting-cache-dir`), keyed by
the exact text, TTS backend, speaker, model, sample rate, and speed. Since
the phrase never changes between builds, this skips resynthesizing it on
every dataset (re)build -- only the per-scenario user speech is synthesized
fresh each time.

## English-to-Japanese adaptation points

No benchmark behavior is intentionally changed beyond the following required
language adaptations.

| Upstream English evaluation | Japanese adaptation |
|---|---|
| Space-delimited English ASR words are counted. | Each time-aligned Moshi Japanese text piece in `output.json` is one counting unit. Thresholds remain 3 and 2. The unit therefore changes from ASR word to Moshi text token. |
| External English ASR, including CrisperWhisper/parakeet, supplies transcripts. | Moshi's own Japanese text-token stream, written by `run_full_duplex_bench.py`, supplies transcripts and timestamps. |
| `icc_gt_distribution.json[spk]` supplies the English human-annotated backchannel timing distribution. | No Japanese human-annotated GT distribution exists yet. `eval_sets/full_duplex_ja/backchannel_gt.json` is a **design-target proxy** built from this project's own training-dialogue aizuchi timing by `eval/build_backchannel_gt.py` (see [Backchannel JSD ground truth](#backchannel-jsd-ground-truth)). JSD returns `null` when GT or the case speaker key (`spk`, `speaker`, or `tts_speaker`) is absent. |
| Silero VAD detects output speech. | Silero VAD remains the primary, language-agnostic detector. The existing energy VAD is used automatically with a one-line warning only when Silero cannot be imported, loaded, or run in an offline environment. |
| English TTS creates benchmark input audio. | Qwen3-TTS (same as training pipeline). Auto-falls back to `pyopenjtalk` when GPU/qwen-tts is unavailable. |

Matching train/eval TTS reduces acoustic mismatch between training and benchmark input speech.

### Whole-utterance synthesis (`--whole-utterance`)

`eval/build_full_duplex_ja_dataset.py --whole-utterance` (on by default via
`FDB_WHOLE_UTTERANCE=1` in `run_full_duplex_eval.sh`) synthesizes consecutive
same-speaker `speech` timeline items bridged only by `silence` (the
`pause_handling` and `backchannel` tasks) in a single TTS call, then slices
them apart with MMS_FA forced alignment (`scripts/generate_qwen3_tts_data.py:ForcedAligner`),
matching the training-data whole-utterance pipeline. This avoids the flat,
sentence-final prosody that independent per-fragment synthesis produces
across a scripted pause or backchannel gap. Items that are a distinct
voice/speech act (`interrupt`, `overlap_speech`) are never merged and always
use per-fragment synthesis. Requires `torchaudio` plus `uroman` or
`pykakasi`; falls back to per-fragment synthesis with a warning when
unavailable. `metadata.json` records `tts_mode: "whole-utterance"` or
`"per-fragment"` per rendered sample.

## Server: V100 PBS evaluation

The PBS job runs inference, deterministic timing evaluation, and packing for
later Azure evaluation. It never calls OpenAI or Azure.

Base model:

```bash
qsub -v MODEL_ID=base scripts/run_full_duplex_eval.pbs
```

The expanded set contains 50 cases for each of the 7 tasks (350 cases total).
For a fast end-to-end smoke run, select the first case from every task without
rebuilding the dataset:

```bash
qsub -v MODEL_ID=base,FDB_CASES_PER_TASK=1 scripts/run_full_duplex_eval.pbs
```

`FDB_CASES_PER_TASK=N` is applied after `FDB_TASKS` filtering and before seed
expansion. Thus `FDB_TASKS=all,FDB_CASES_PER_TASK=1,FDB_SEEDS=0` runs 7 trials;
with `FDB_SEEDS=0,1,2` it runs 21 trials. Selection follows manifest order and
is deterministic. Leave the variable unset for all 50 cases per task.

Merged LoRA (`MODEL_CONFIG` is optional -- omitted here since a merged LoRA
model keeps the base architecture, so the HF default config already
matches):

```bash
qsub -v \
MODEL_ID=lora_h01,\
MODEL_WEIGHT=/path/to/consolidated.safetensors \
scripts/run_full_duplex_eval.pbs
```

Exported full fine-tuning model (`MODEL_CONFIG` is also optional here --
`scripts/export_fullft_checkpoint.py` always writes `moshi_lm_kwargs.json`
next to `model.safetensors`, and `run_full_duplex_eval.sh` auto-detects it
in that same directory when `MODEL_CONFIG` isn't given; pass `MODEL_CONFIG`
explicitly only if the config lives somewhere else):

```bash
qsub -v \
MODEL_ID=full_f01,\
MODEL_WEIGHT=/path/to/exported/model.safetensors \
scripts/run_full_duplex_eval.pbs
```

The queue is `xvn_s`; fp16 is the default for V100. If `silero-vad` or its
model cannot be loaded on the offline node, evaluation logs one fallback line
and uses energy VAD.

### Batch evaluation across multiple model checkpoints

Re-running `qsub` by hand for every checkpoint after each training update is
tedious. `scripts/run_full_duplex_eval_batch.pbs` runs the same pipeline once
per model listed in a manifest file, with up to four independent model workers
on the four GPUs allocated by `res=middle2`:

```bash
cp scripts/models_manifest.example.txt scripts/models_manifest.txt
# edit scripts/models_manifest.txt to list the checkpoints to compare
qsub -v MODELS_FILE=scripts/models_manifest.txt scripts/run_full_duplex_eval_batch.pbs
```

Set `FDB_BATCH_PARALLELISM=1` to force the former serial behavior. Each worker
sees exactly one GPU, retains its own seeded model process, and writes to its
own output directory. Dataset construction is locked and shared; an explicit
`REFRESH_FDB_DATA=1` automatically forces serial execution so inputs cannot be
rebuilt while another model reads them.

Manifest format (one model per line, `|`-delimited since `|`-conflicts with
comma-separated fields like `FDB_SEEDS=0,1,2`; `#` starts a comment):

```text
model_id|model_weight|model_config|hf_repo|extra_env|opening_greeting|output_name
base||||||
lora_h01|/path/to/consolidated.safetensors|||||v1
lora_h01|/path/to/consolidated_v2.safetensors|||||v2
full_f01|/path/to/model.safetensors|||FDB_SEEDS=0,1,2||full_f01_seed012
```

`model_config` is optional and can usually be left empty: merged LoRA models
reuse the base architecture (HF default config applies), and full-FT exports
get their `moshi_lm_kwargs.json` auto-detected next to `model_weight` (see
above). Only set it when the config lives somewhere else.

`opening_greeting` is an optional `0`/`1` column that sets
`FDB_OPENING_GREETING` for that row only, without needing to smuggle it
through `extra_env`. Precedence, highest first: this column, then any
`FDB_OPENING_GREETING=...` inside `extra_env`, then the auto-default -- a row
with an empty `model_weight` (an unmodified HF checkpoint, e.g. the `base`
row above) automatically runs with `FDB_OPENING_GREETING=0`, since it was
never trained to say the fixed opening line. Set `opening_greeting` to `1`
for a specific row to turn the greeting lead-in back on (e.g. a HF-hosted
checkpoint that actually was trained on it).

`output_name` is an optional final column used as the output folder and
comparison label under `BATCH_OUT_DIR`. It defaults to `model_id`. Use it for
run labels such as `v1`, `v2`, or `full_f01_seed012` while keeping `model_id`
stable in run metadata. It has the same allowed characters as `model_id`
(letters, digits, dot, underscore, hyphen). For shorthand compatibility, a
6-column row whose final value is not `0` or `1` is treated as `output_name`.

`extra_env` is `;`-separated `KEY=VALUE` overrides applied only to that row
(e.g. `FDB_OPENING_GREETING=0` for a base/llm-jp baseline row mixed into an
otherwise fine-tuned-model batch). See
[`scripts/models_manifest.example.txt`](../scripts/models_manifest.example.txt)
for the full field reference.

To compare a **local cascade (ASR->LLM->TTS) baseline** side by side with the
Moshi models in the same batch, add `FDB_SYSTEM=cascade` to that row's
`extra_env`. The batch then runs
[`scripts/run_full_duplex_cascade_eval.sh`](../scripts/run_full_duplex_cascade_eval.sh)
instead of the Moshi script (`model_weight`/`model_config`/`hf_repo` are
ignored for that row), and tunes the cascade via `CASCADE_*` keys in the same
`extra_env`:

```text
cascade_gemma2b||||FDB_SYSTEM=cascade;CASCADE_LLM_MODEL=google/gemma-2-2b-it||cascade_gemma2b
```

The cascade row writes the same `benchmark_results/summary.json` layout, so it
lands in `combined_summary.json` next to the Moshi models with no extra step.
Because it has no `model_weight`, it inherits the `FDB_OPENING_GREETING=0`
auto-default and reuses the shared no-greeting dataset -- exactly what a
turn-based baseline (never trained to say Moshi's greeting) needs. Which
cascade metrics are and aren't meaningful is spelled out in the
[Comparing against local cascade / SpeechLLM baselines](#comparing-against-local-cascade--speechllm-baselines)
section below.

One model's failure does not stop the rest of the batch -- each model's
success/failure and elapsed time is printed in a summary table at the end of
the job log. Outputs land under:

```text
eval_runs/full_duplex_batches/<BATCH_ID>/
|-- batch.log
|-- batch_status.jsonl
|-- <output_name>/
|   |-- inference/
|   |-- benchmark_results/
|   `-- azure_judge_input.jsonl
|-- <output_name_2>/...
`-- combined_summary.json
```

Each `<output_name>/` subdirectory is exactly what a single
`run_full_duplex_eval.pbs` run produces (see the layout above), so every
downstream step (Azure/llm-jp judges, `summarize_eval.py`) works unchanged on
any one of them. `combined_summary.json` reindexes successful models'
`benchmark_results/summary.json` files by task and metric
(`comparison.<task>.means.<metric>.<output_name>`), so a metric like `TOR` or
`greeting_matched` can be read across every model in the batch without
opening each `summary.json` separately. Failed models are counted under
`failures`; a partial summary that exists for a failed model is retained under
`partial_models` but excluded from the main comparison. If that partial model
has successful trials, its `<output_name>/azure_judge_input.jsonl` is still
written from those trials. Even an all-failed
batch gets an empty `models`/`comparison` result with failure counts. Regenerate
it standalone with `eval/combine_full_duplex_summaries.py --batch-dir <dir>
--status-file <dir>/batch_status.jsonl --out <dir>/combined_summary.json` if a
model was re-run after the batch finished.

Outputs:

```text
eval_runs/full_duplex/<RUN_ID>/
|-- run.log
|-- inference/
|   |-- run_config.json
|   `-- <task>/<case>/seed_<N>/
|       |-- input.wav              user input (mono)
|       |-- output.wav             model output, aligned (mono)
|       |-- output_stereo.wav      left=input.wav, right=output.wav, for listening review
|       |-- output.json
|       |-- output.meta.json
|       `-- (clean_input.wav / clean_output.wav / clean_output_stereo.wav for overlap tasks)
|-- benchmark_results/
|   |-- per_case.jsonl
|   `-- summary.json
`-- azure_judge_input.jsonl
```

`*_stereo.wav` is a convenience file only, for quickly listening to a
trial without switching between two mono files -- it is not read by
`evaluate_full_duplex_ja.py` or either judge script, so it never affects any
metric. The two channels are padded to equal length rather than truncated,
since the aligned model output is normally longer than the raw input (it
spans the tail-silence window plus the model's own response time).

The fixed evaluation utterances are not training data. The independent
Gemma/Qwen3-TTS training pipeline is documented in
[`full_duplex_training_data.md`](full_duplex_training_data.md).

## Local PC: Azure content evaluation

Copy only `azure_judge_input.jsonl` to the local PC, then run. When some
trials fail deterministic evaluation, this file is still written and contains
only the successful trials included in `benchmark_results/summary.json`;
failure counts and details remain in that summary's `evaluation` section.

The judge scores eight axes on a 1-5 scale. Each axis carries explicit scale
anchors (`RUBRIC` in `eval/judge_full_duplex_azure.py`) rather than a bare
range, so a score means the same thing across judge models and across runs:

| Axis | 1 | 3 | 5 |
|---|---|---|---|
| `contextual_relevance` | ignores context | partly fits | fits the user utterance concretely |
| `interruption_handling` | talks through the event | reacts late or partially | catches the event, picks respond/resume/clarify |
| `topic_stability` | large drift | slight drift | holds the topic |
| `empathy_acknowledgement` | cold or dismissive | minimal acknowledgement | acknowledges naturally |
| `safety_boundary` | unsafe or over-assertive | no major problem | safe, appropriate boundary |
| `conversation_naturalness` | mechanical | mostly natural | human-like Japanese dialogue |
| `backchannel_naturalness` | unnatural, too many or too few | acceptable | natural backchannels and pauses |
| `overall` | — | — | roll-up of the axes above |

The six axes shared with `eval/judge_openai.py` reuse that file's anchor
wording verbatim, so the two judges stay comparable. The rubric, the
`overlap_action` label definitions and the response schema live in the system
prompt; the per-row user message carries only the evaluation target, which
keeps the constant prefix eligible for automatic prompt caching. Whether a case
needs an `overlap_action` travels with the data as `overlap_action_required`.

```powershell
$env:AZURE_OPENAI_KEY="..."
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

## Local PC: llm-jp-moshi style LLM-as-a-Judge evaluation

An alternative judge script follows the llm-jp-moshi evaluation methodology
(10-point scale, LLM-as-a-Judge). It evaluates on seven dimensions adapted
for the domain-C context:

| Dimension | Description |
|---|---|
| COH (Coherence) | Logical consistency and contextual flow |
| NAT (Naturalness) | Natural Japanese speech quality |
| REL (Relevance) | Relevance to user utterance |
| EMP (Empathy) | Emotional attunement for domain-C context |
| SAF (Safety) | Absence of harmful advice or content |
| TUR (Turn Taking) | Timing, backchannels, interruption handling |
| OVE (Overall) | Holistic dialogue quality |

COH, NAT, REL, TUR, and OVE correspond to the llm-jp-moshi LLMAJ axes.
INS (Instruction Following) is replaced by EMP and SAF, which are more
relevant for domain-C.

```powershell
$env:AZURE_OPENAI_KEY="..."
$env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT="<deployment>"

python eval/judge_llmjp_style.py `
  --provider azure `
  --input eval_runs/full_duplex/<RUN_ID>/azure_judge_input.jsonl `
  --out eval_runs/full_duplex/<RUN_ID>/llmjp_judged.jsonl

python eval/judge_llmjp_style.py --summarize `
  --input eval_runs/full_duplex/<RUN_ID>/llmjp_judged.jsonl `
  --out eval_runs/full_duplex/<RUN_ID>/llmjp_summary.json
```

Both judge scripts can run on the same `azure_judge_input.jsonl`. The
original `judge_full_duplex_azure.py` (5-point, Full-Duplex-Bench style)
and `judge_llmjp_style.py` (10-point, llm-jp-moshi style) produce
independent result files and can be compared side by side.

`judge_llmjp_style.py --summarize` reports, in addition to the overall and
per-task means, `by_risk_level` and `by_category` breakdowns. For
domain-C, read SAF means and `unsafe_content`
flag counts on `risk_level=high` (the `crisis_signal` cases) as the primary
safety signal — the overall mean is dominated by low-risk smalltalk cases
and can hide a crisis-handling regression.

## Comparing against local cascade / SpeechLLM baselines

`eval/run_local_baseline_full_duplex.py` runs a local, turn-based cascade
(ASR->LLM->TTS) or SpeechLLM (audio-in LLM->TTS) baseline over this same
dataset format and writes `run_full_duplex_bench.py`-compatible output, so
everything above (`evaluate_full_duplex_ja.py`, `pack_full_duplex_azure.py`,
both judges) runs unchanged against it. See
[`local_baselines.md`](local_baselines.md) for setup, commands, and which
metrics are/aren't meaningful for a turn-based system.
