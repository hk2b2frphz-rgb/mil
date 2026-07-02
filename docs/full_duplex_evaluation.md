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
least `1.0` second is a takeover. Collection stops at the first takeover.

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

Toggle via `run_full_duplex_eval.sh`'s `FDB_OPENING_GREETING=1` (default) /
`FDB_OPENING_GREETING_GAP_SEC=0.4`. **Set `FDB_OPENING_GREETING=0` when
evaluating base Moshi or llm-jp baselines** -- they were never trained to say
this line, so there is nothing to wait for and reserving lead-in time for it
would just be dead air in front of their real first response. The default
`FDB_DATA_DIR` encodes this flag (`data/full_duplex_ja_greeting` vs.
`data/full_duplex_ja_nogreeting`), so toggling it always builds/reuses the
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
for the counseling domain:

| Dimension | Description |
|---|---|
| COH (Coherence) | Logical consistency and contextual flow |
| NAT (Naturalness) | Natural Japanese speech quality |
| REL (Relevance) | Relevance to user utterance |
| EMP (Empathy) | Emotional attunement for counseling context |
| SAF (Safety) | Absence of harmful advice or content |
| TUR (Turn Taking) | Timing, backchannels, interruption handling |
| OVE (Overall) | Holistic dialogue quality |

COH, NAT, REL, TUR, and OVE correspond to the llm-jp-moshi LLMAJ axes.
INS (Instruction Following) is replaced by EMP and SAF, which are more
relevant for loneliness/isolation counseling.

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

## Comparing against local cascade / SpeechLLM baselines

`eval/run_local_baseline_full_duplex.py` runs a local, turn-based cascade
(ASR->LLM->TTS) or SpeechLLM (audio-in LLM->TTS) baseline over this same
dataset format and writes `run_full_duplex_bench.py`-compatible output, so
everything above (`evaluate_full_duplex_ja.py`, `pack_full_duplex_azure.py`,
both judges) runs unchanged against it. See
[`local_baselines.md`](local_baselines.md) for setup, commands, and which
metrics are/aren't meaningful for a turn-based system.
