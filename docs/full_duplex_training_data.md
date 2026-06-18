# Japanese full-duplex training data

The 14 fixed cases in `eval_sets/full_duplex_ja/scenarios.jsonl` are evaluation
inputs. They are not used as training text.

Full-duplex training data is generated independently through the repository's
existing pipeline:

1. `build_full_duplex_training_use_cases.py` creates varied support-window use
   cases with a balanced full-duplex task label.
2. `generate_synthetic_moshi_training_data.py` asks Gemma to write Japanese
   counselor dialogues and timing events.
3. `generate_qwen3_tts_data.py` renders stereo audio with Moshi on the left and
   the user/environment on the right.
4. The usual Moshi fine-tuning launchers consume
   `training_set/synthetic_moshi_train.jsonl`.

## Generate a dataset

On the GPU server:

```bash
NUM_CASES=140 \
RUN_ID=full_duplex_v1 \
bash scripts/run_full_duplex_training_data.sh
```

PBS:

```bash
qsub scripts/run_full_duplex_training_data.pbs

# Small run
qsub -v NUM_CASES=14 scripts/run_full_duplex_training_data.pbs
```

The PBS job uses one V100 and defaults Gemma/Qwen3-TTS to `float16`.
Its output is written under `data/runs/fd_train_<PBS_JOBID>/`.

Generate only use cases and Gemma dialogue scripts:

```bash
NUM_CASES=140 \
RUN_ID=full_duplex_v1 \
STEPS=use_cases,dialogues \
bash scripts/run_full_duplex_training_data.sh
```

Template fallback is disabled by default because repeated fallback dialogues
reduce training diversity. The deterministic template backend can be used for
a dialogue-schema smoke test:

```bash
GEMMA_BACKEND=template \
NUM_CASES=14 \
RUN_ID=full_duplex_smoke \
STEPS=use_cases,dialogues \
bash scripts/run_full_duplex_training_data.sh
```

Outputs:

```text
data/runs/<RUN_ID>/
├── use_cases.jsonl
├── gemma_dialogues/
│   └── dialogues.jsonl
└── training_set/
    ├── synthetic_moshi_train.jsonl
    ├── dialogues.jsonl
    └── data_stereo/
        ├── sample_*.wav
        └── sample_*.json
```

The run can be passed to the existing fine-tuning launchers:

```bash
SRC_RUN_DIR=data/runs/full_duplex_v1 qsub scripts/fullft_sweep.pbs
```

## Timing schema

Normal turns remain sequential. A turn can additionally specify:

- `timing: "overlap_previous"`: start during the immediately preceding audio
  turn.
- `start_after_previous_start_sec`: overlap start offset.
- `truncate_previous_after_sec`: stop the preceding audio shortly after an
  interruption begins.
- `gain`: amplitude multiplier, mainly for background speech.
- `voice_role`: `user`, `other`, or `background`.
- `event`: behavior label such as `user_interruption`.

`generate_qwen3_tts_data.py` validates every dialogue carrying a
`duplex_task`. Invalid or missing timing events fail before the Qwen3-TTS model
is loaded.

The resulting WAVs contain actual simultaneous left/right channel audio for
backchannels, interruptions, side speech, and background speech. This differs
from the fixed evaluation adapter, whose current inputs are mostly arranged
serially on a mono timeline.
