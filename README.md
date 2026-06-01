# Moshi Fixed-Input Response Recorder

Batch experiment tool that feeds **fixed audio files** into Moshi as the
opening utterance and records Moshi's responses — both audio and text
(inner-monologue transcript) — in a structured directory tree.

Designed for reproducible comparison experiments: multiple input audios ×
multiple random seeds.

---

## Requirements

- **GPU**: NVIDIA GPU with at least ~16 GB VRAM for bfloat16 inference
  (e.g. A100, A40, RTX 4090).
  For the full `moshiko-pytorch-bf16` checkpoint, ~24 GB is comfortable.
- **CUDA**: matching driver and CUDA toolkit for your PyTorch wheel.
- **Python**: 3.11+ recommended (type-union syntax `str | Path` is used).

---

## Installation (on the GPU server)

Run these commands from the cloned `mos` folder:

```bash
cd mos

# 1. Install uv if needed
pip install uv

# 2. Create .venv and install dependencies
uv sync

# 3. Run inside the uv environment
uv run python response_recorder.py --help
```

`pyproject.toml` is configured for CUDA 12.1 PyTorch wheels via the
`pytorch-cu121` index. If your GPU server uses a different CUDA wheel set,
update the `[[tool.uv.index]]` URL and matching `tool.uv.sources` entries.

If the `moshi` PyPI package is not available or is outdated, install from
source into the uv environment:

```bash
uv pip install git+https://github.com/kyutai-labs/moshi.git
```

`uv sync` installs local Python-side TTS dependencies (`pyopenjtalk` and
`pyttsx3`). No sudo is required. Japanese stdin/text prompts use
`pyopenjtalk` first, so TTS works locally without an online service.

```bash
uv sync
uv run python -c "import pyopenjtalk; print('pyopenjtalk OK')"
echo "こんにちは" | uv run python response_recorder.py --out-dir results/stdin/
```

If you intentionally use an already-active virtual environment instead of
`uv run`, install the missing package into that environment:

```bash
uv pip install pyttsx3
```

The script tries local TTS backends in this order: `pyopenjtalk`, `pyttsx3`,
`espeak-ng`, `espeak`, `pico2wave`, then Windows System.Speech. `--tts-voice`
only applies to backends that support voice selection.

---

## Quick-start examples

### Minimal run (one input file, three seeds, default model)

```bash
python response_recorder.py \
    --inputs  prompts/hello.wav \
    --seeds   0,1,2 \
    --silence-sec 15 \
    --out-dir results/
```

### Text prompt with simple TTS

Pipe text on standard input:

```bash
echo "こんにちは。今日の予定を教えてください。" | python response_recorder.py \
    --out-dir results/stdin/
```

This reads the text from stdin, synthesizes it to WAV under
`<out-dir>/_tts_inputs/`, feeds that WAV to Moshi, saves about the first
10 seconds of the response as `response.wav`, and prints a chronological
conversation timeline.

```bash
python response_recorder.py \
    --texts "こんにちは。今日の予定を教えてください。" \
    --seeds 0,1,2 \
    --out-dir results/text-prompt/
```

Multiple text prompts can be passed directly, or loaded from a UTF-8 text file
with one prompt per line:

```bash
python response_recorder.py \
    --text-file prompts.txt \
    --out-dir results/text-file/
```

The generated prompt WAV files are saved under `<out-dir>/_tts_inputs/`.
The script first tries local Japanese TTS with `pyopenjtalk`, then falls back
to other local system TTS backends. For OS-specific voices, select a voice when
needed:

```bash
python response_recorder.py \
    --texts "もしもし、聞こえますか。" \
    --tts-voice "Microsoft Haruka Desktop" \
    --out-dir results/ja-tts/
```

### Multiple input files and more seeds

```bash
python response_recorder.py \
    --inputs  prompts/hello.wav prompts/question.wav \
    --seeds   0,1,2,3,4 \
    --silence-sec 20 \
    --out-dir results/multi/
```

### Entire directory of prompts

```bash
python response_recorder.py \
    --inputs  prompts/ \
    --seeds   0,1,2 \
    --silence-sec 15 \
    --out-dir results/
```

### Explicit model selection

```bash
python response_recorder.py \
    --hf-repo llm-jp/llm-jp-moshi-v1 \
    --inputs  prompts_ja/ \
    --seeds   0,1 \
    --silence-sec 20 \
    --out-dir results/llm-jp-moshi/
```

### Local weights (no HF download)

```bash
python response_recorder.py \
    --hf-repo    kyutai/moshiko-pytorch-bf16 \
    --moshi-weight  /data/moshi/moshi.safetensors \
    --mimi-weight   /data/moshi/mimi.safetensors \
    --tokenizer     /data/moshi/tokenizer.model \
    --config        /data/moshi/config.json \
    --inputs        prompts/ \
    --seeds         0,1,2 \
    --out-dir       results/local/
```

### Override sampling parameters

```bash
python response_recorder.py \
    --inputs  prompts/ \
    --seeds   0 \
    --temp      0.8 \
    --temp-text 0.8 \
    --cfg-coef  1.5 \
    --max-gen-sec 45 \
    --silence-sec 15 \
    --out-dir results/high-temp/
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--hf-repo` | `llm-jp/llm-jp-moshi-v1` | HuggingFace model repo |
| `--moshi-weight` | — | Local path to Moshi weights |
| `--mimi-weight` | — | Local path to Mimi weights |
| `--tokenizer` | — | Local path to tokenizer |
| `--config` | — | Local path to config file |
| `--inputs` | — | WAV files or directories |
| `--texts` | — | Text prompts to synthesize into WAV inputs |
| `--text-file` | — | UTF-8 file with one prompt per line |
| `--stdin` | auto for piped stdin | Read one text prompt from standard input |
| `--tts-voice` | — | Optional TTS voice name |
| `--tts-rate` | `200` | TTS speaking rate for `pyttsx3` |
| `--silence-sec` | `15.0` | Silence appended after input |
| `--seeds` | — | Comma-separated seed list |
| `--num-trials` | — | Use seeds 0..N-1 (fallback when `--seeds` absent) |
| `--out-dir` | *(required)* | Root output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--half` | bfloat16 | Pass to switch to float16 |
| `--temp` | model default | Audio sampling temperature |
| `--temp-text` | model default | Text sampling temperature |
| `--cfg-coef` | model default | CFG coefficient |
| `--max-gen-sec` | `60.0` | Per-trial generation cap (seconds) |
| `--response-sec` | `10.0` | Seconds of response audio to save |
| `--allow-overlap` | off | Allow Moshi to speak while user audio is still being fed |
| `--no-print-transcript` | off | Suppress transcript output |

---

## Why `--silence-sec` matters

Moshi is a full-duplex streaming model.  It generates output **only while
input frames are being fed**.  Without silence appended to the end of the
input utterance, the response would be cut off the moment the speech ends.

`--silence-sec` (default 15 s) pads the input with zeros so Moshi has time
to produce a complete reply.  Increase it for longer expected responses.

---

## Output structure

```
<out-dir>/
├── run_metadata.json          # Global metadata (model, date, all CLI args)
├── <input_stem>/
│   └── seed_<N>/
│       ├── response.wav       # Moshi's response audio (acoustic-delay corrected)
│       ├── conversation_timeline.jsonl
│       ├── conversation_timeline.txt
│       ├── transcript.jsonl   # One JSON line per emitted text token
│       ├── transcript.txt     # Plain-text concatenation of all pieces
│       └── meta.json          # Per-trial metadata (schema below)
...
```

### Timeline output

The console and `conversation_timeline.txt` show user input and Moshi output
on one timeline:

```text
Conversation timeline:
[00:00.000] user  speech_start こんにちは
[00:02.560] user  speech_end
[00:03.120] moshi speech_start (audio starts)
[00:03.200] moshi text_output こんにちは。どうしましたか。
```

`conversation_timeline.jsonl` stores the same events as JSON lines for later
analysis.

By default, the prompt is fed as a dual-stream history: user audio contains the
full prompt, while the Moshi audio stream is forced to silence for the same
duration. Any Moshi text sampled during that prompt window is removed from the
history, so the continuation starts from `user audio + silent Moshi audio`
rather than from a half-generated Moshi utterance. The script then saves only
the continuation generated during the appended silence. Use `--allow-overlap`
to keep Moshi's original full-duplex behavior.

### `transcript.jsonl` format

Each line is a JSON object:

```json
{"step": 42, "time_sec": 3.36, "piece": " hello"}
```

- `step`: frame index within the trial
- `time_sec`: `step / frame_rate` (frame_rate = 12.5 Hz → 80 ms per step)
- `piece`: decoded SentencePiece token (padding ids 0 and 3 already excluded,
  `▁` replaced with a space)

### `meta.json` schema

```json
{
  "input_path": "/abs/path/to/prompt.wav",
  "input_duration_sec": 2.56,
  "silence_sec": 15.0,
  "seed": 0,
  "model_repo": "llm-jp/llm-jp-moshi-v1",
  "dtype": "bfloat16",
  "device": "cuda",
  "temp": NaN,
  "temp_text": NaN,
  "cfg_coef": NaN,
  "frame_rate": 12.5,
  "sample_rate": 24000,
  "total_steps": 225,
  "input_end_step": 32,
  "first_audio_step": 18,
  "first_audio_time_sec": 1.44,
  "audible_response_start_step": 20,
  "audible_response_start_sec": 1.6,
  "audible_start_after_input_sec": -0.96,
  "first_response_step": 18,
  "first_response_latency_sec": 1.44,
  "wall_time_sec": 47.2,
  "output_audio_sec": 14.08
}
```

`NaN` appears for temperature/cfg fields when the model's built-in default
was used (i.e. the corresponding CLI flag was not supplied).

`first_response_latency_sec` is the latency to the **first non-padding text
token**.  It is `null` when no text was generated.

`first_audio_time_sec` is when Moshi first returned audio tokens. Because the
codec has an acoustic delay, `audible_response_start_sec` is the estimated
stream time where that audio becomes audible in `response.wav`.
`audible_start_after_input_sec` compares that estimated audible start with the
end of the prompt audio: negative means Moshi started speaking while the prompt
was still being fed, positive means it started after the prompt ended.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Empty `response.wav` / blank transcript | `--silence-sec` too short | Increase to 20–30 s |
| CUDA OOM | VRAM insufficient | Use a smaller model, or `--half` (float16) |
| `ModuleNotFoundError: moshi` | Package not installed | `pip install moshi` or install from source |
| Garbled text | Expected for some tokens; check `▁` replacement | No fix needed; output is still valid |
| Trial skipped with `FAILED` | Per-trial exception | Check stderr; other trials continue |
