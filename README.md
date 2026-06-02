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

Make the local TTS prompt faster with `--tts-speed`:

```bash
echo "こんにちは" | uv run python response_recorder.py --tts-speed 1.5 --out-dir results/fast-tts/
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

### 約 1 時間分の合成データで Moshi LoRA FT する（推奨フルパイプライン）

孤独・孤立相談窓口向けの日本語ドメイン適応を、4 ステップで一気に回します。
全部 GPU サーバー上で実行する想定です。

```text
1. use_cases.jsonl     (人手レス、軸の組み合わせで 100 件)
        ↓ scripts/build_use_cases.py
2. dialogues.jsonl     (Gemma 4 が emotion / silence 付きで生成)
        ↓ scripts/generate_synthetic_moshi_training_data.py --mode dialogues-only
3. stereo WAV + manifest  (Qwen3-TTS、moshi 固定 / user プール巡回)
        ↓ scripts/generate_qwen3_tts_data.py --dialogues-jsonl ...
4. Moshi LoRA fine-tune
        ↓ torchrun -m train configs/moshi_lora_jp_loneliness.yaml
```

#### 一発で回す

```bash
# moshi-finetune を兄弟ディレクトリに clone しておく
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune

# 各 uv 環境を sync 済みであることを確認
uv sync                        # Moshi + Qwen3-TTS
uv sync --project gemma_runtime  # Gemma 4

# 4 ステップ一気通貫
bash scripts/run_pipeline.sh
```

環境変数で各種調整可能 (`OUT_ROOT`, `NUM_CASES`, `GEMMA_MODEL`, `QWEN_TTS_MODEL`,
`FT_CONFIG`, `MOSHI_FT_REPO`, `NPROC`)。途中で止めたい場合は `STEPS` で絞れます:

```bash
STEPS=use_cases,dialogues bash scripts/run_pipeline.sh   # 音声化前まで
STEPS=audio bash scripts/run_pipeline.sh                 # 音声化だけ再実行
```

#### 個別ステップ

```bash
# 1. 100 件の多様な use case カードを軸組み合わせで作る
python scripts/build_use_cases.py \
  --out-path data/v1/use_cases.jsonl --num 100

# 2. Gemma で対話 JSONL のみ生成（音声化はスキップ）
uv run python scripts/generate_synthetic_moshi_training_data.py \
  --out-dir data/v1/gemma_dialogues \
  --use-cases-jsonl data/v1/use_cases.jsonl \
  --num-dialogues 100 \
  --mode dialogues-only \
  --gemma-backend transformers-subprocess \
  --allow-template-fallback

# 3. Qwen3-TTS で音声化（emotion / silence ターン込み）
uv run python scripts/generate_qwen3_tts_data.py \
  --out-dir data/v1/training_set \
  --dialogues-jsonl data/v1/gemma_dialogues/dialogues.jsonl

# 4. Moshi LoRA FT
cd ../moshi-finetune
torchrun --nproc-per-node 1 -m train \
  "$OLDPWD/configs/moshi_lora_jp_loneliness.yaml"
```

#### 多様性の設計（`build_use_cases.py`）

軸を直積に近い形で組み合わせて 100 件を出します:

| 軸 | バリエーション |
|---|---|
| situation | 20 種（夜の雑談 / 退職後 / 介護疲れ / 入院中 / 喪失 / SNS疲れ / …） |
| age_band | 20代〜70代 |
| gender | 男性 / 女性 |
| risk_level | low:medium:high = 60:30:10 |
| silence_pattern | none:occasional:heavy = 55:30:15（特定 situation は heavy 寄り） |
| opening_kind | smalltalk / feelings / silence |

#### データ量の目安

100 対話 × 平均 35 秒 ≒ **約 1 時間**。長めの heavy silence パターンや medium/high
risk の対話を含めると 1.0〜1.2 時間程度に着地します。

---

### Qwen3-TTS による日本語対話データ生成（推奨・シンプル）

`scripts/generate_qwen3_tts_data.py` は Qwen3-TTS のプリセット話者で日本語対話を
音声合成し、Moshi fine-tune フォーマットで書き出します。
**Gemma 不要**・**依存環境がシンプル**なため、まず最初にこちらを試してください。

仕組み:

1. ハードコードされた短い日本語対話テンプレートを使用（Gemma 生成は不要）。
2. Qwen3-TTS の `generate_custom_voice()` が話者ごとに異なるプリセット声で各ターンを合成。
   - **moshi 側は `--speaker-moshi` で固定**（デフォルト: `Serena`）
   - **user 側は `--user-speaker-pool` のプールから対話ごとに 1 人ずつローテーション**
     （デフォルトプール: `Ono_Anna, Sohee, Vivian, Dylan, Eric, Aiden`）。
   - これにより「いつもの相談員（moshi）が、毎回違う相談者（user）と話す」
     データセットになり、Moshi 側の汎化に効くことを期待。
3. 相談員 (moshi) を左チャンネル、相談者 (user) を右チャンネルのステレオ WAV に配置。
4. Moshi fine-tune manifest (`synthetic_moshi_train.jsonl`) と `dialogues.jsonl` を書き出し。

依存パッケージ `qwen-tts` を `pyproject.toml` に含めているので `uv sync` で入ります
（手動で入れる場合は `pip install -U qwen-tts`）。

```bash
uv sync

uv run python scripts/generate_qwen3_tts_data.py \
  --out-dir data/qwen3_tts_test \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device cuda \
  --dtype bfloat16 \
  --speaker-user Ono_Anna \
  --speaker-moshi Serena \
  --num-dialogues 3
```

実装は `qwen_tts.Qwen3TTSModel` を直接使い、以下の API を呼びます:

```python
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=torch.bfloat16)
wavs, sr = model.generate_custom_voice(
    text="...", language="Japanese", speaker="Ono_Anna", instruct=None,
)
```

Qwen3-TTS CustomVoice 系の利用可能なプリセット話者:

```
Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
```

`--instruct-user` / `--instruct-moshi` で**全ターン共通**のスタイル指示（例:
`"温かく穏やかなトーンで"`）をオプションで渡せます。

#### ターン単位の感情ラベル制御（試験的）

テンプレート対話の各ターンには `emotion` ラベル（`hesitant`, `sad`, `warm`,
`empathetic`, `relieved`, `concerned` など）が付いていて、起動時に
スクリプト内の `EMOTION_PRESETS` で **Qwen3-TTS の `instruct` 文字列に解決**されます。
これにより、同じ対話の中で「ためらう」「沈む」「ほっとする」といった感情変化が
音声側にも反映されます。

利用可能なラベル一覧（試験的、`scripts/generate_qwen3_tts_data.py` 内の
`EMOTION_PRESETS` に定義）:

```
neutral, hesitant, sad, lonely, anxious, relieved, grateful,
warm, gentle, empathetic, encouraging, concerned, reassuring
```

優先順位は **ターンの `emotion`** > `--instruct-user` / `--instruct-moshi` > なし。

| フラグ | 用途 |
|---|---|
| `--no-emotion` | ターンの `emotion` を無視してプレーンに合成（A/B 比較用） |
| `--emotion-map-file path.json` | `{ "lonely": "別の指示文" }` のように一部だけ上書きする JSON |

`emotion-map-file` の例:

```json
{
  "lonely": "声のボリュームを落とし、夜の静けさを背負うように話して",
  "concerned": "急かさず、相手のペースを尊重する穏やかなトーンで尋ねるように"
}
```

出力 WAV と一緒に書かれる JSON のメタデータには、その対話で実際に使われた
`emotion_map_used` がそのまま記録されるので、後から再現・比較できます。

#### ユーザーの沈黙 / moshi 側からの声かけ

孤独・孤立相談窓口を想定し、テンプレートには**ユーザーが言葉に詰まる / 長く沈黙する**
パターンを含めています。沈黙中は moshi 側が穏やかに声をかけるよう、
**moshi の連続ターン** + 沈黙ターンを混在させた構造です:

```python
{"speaker": "user",    "text": "うまく言えないんですけど…えっと…", "emotion": "hesitant"},
{"speaker": "silence", "duration_sec": 3.5, "note": "ユーザーが言い淀んで黙ってしまう"},
{"speaker": "moshi",   "text": "ゆっくりで大丈夫ですよ。",         "emotion": "gentle"},
{"speaker": "silence", "duration_sec": 2.0},
{"speaker": "moshi",   "text": "急がなくて大丈夫です。",            "emotion": "reassuring"},
```

実装上のポイント:

- `speaker: "silence"` のターンは音声を合成せず、`duration_sec` ぶんだけ
  **両チャンネルとも無音**の時間を挿入します
- moshi が連続して話す形（user の応答なし）も自然に書けるので、
  「沈黙→声かけ→さらに沈黙→さらに声かけ」のような窓口対応を再現できます
- 出力 JSON のメタデータには `silences: [{start_sec, end_sec, duration_sec, note}, ...]`
  として全沈黙区間が記録されます

Moshi の学習用途では、これにより
**「相手が話さなくても自分から会話を維持できる」挙動**を学習データに含められます。

出力ディレクトリ構造:

```text
data/qwen3_tts_test/
├── synthetic_moshi_train.jsonl   ← Moshi fine-tune manifest
├── dialogues.jsonl               ← 対話スクリプト
└── data_stereo/
    ├── sample_001_smalltalk_evening_001.wav
    ├── sample_001_smalltalk_evening_001.json
    └── ...
```

manifest の形式:

```json
{"path": "data_stereo/sample_001_smalltalk_evening_001.wav", "duration": 18.4}
```

WAV チャンネル規約:

- 左チャンネル: Moshi / 相談員
- 右チャンネル: user / 相談者

#### `generate_qwen3_tts_data.py` オプション

| フラグ | デフォルト | 説明 |
|---|---|---|
| `--out-dir` | *(必須)* | 出力ディレクトリ |
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Qwen3-TTS の HuggingFace モデル ID |
| `--device` | `cuda` | `cuda` または `cpu` |
| `--dtype` | `bfloat16` | `float16` / `bfloat16` / `float32` |
| `--attn-impl` | `default` | `default` / `flash_attention_2` / `sdpa` / `eager`（flash-attn 未インストール時は自動フォールバック） |
| `--language` | `Japanese` | `generate_custom_voice` に渡す language 文字列 |
| `--speaker-user` | `Ono_Anna` | プール未指定時に使う user 側の固定話者 |
| `--user-speaker-pool` | `Ono_Anna,Sohee,Vivian,Dylan,Eric,Aiden` | 対話ごとに順にローテーションする user 話者のカンマ区切り列。`''` を渡すと `--speaker-user` で固定 |
| `--speaker-moshi` | `Serena` | moshi 側プリセット話者（固定） |
| `--instruct-user` | *(なし)* | user 発話の既定スタイル指示（ターン側 emotion が無い場合のみ使う） |
| `--instruct-moshi` | *(なし)* | moshi 発話の既定スタイル指示（ターン側 emotion が無い場合のみ使う） |
| `--no-emotion` | off | テンプレートの emotion ラベルを無視する |
| `--emotion-map-file` | *(なし)* | 感情ラベル→instruct 文字列の上書き JSON |
| `--num-dialogues` | `3` | 生成する対話数（最大 3、テンプレート数に依存） |
| `--lead-in-sec` | `0.3` | 先頭の無音（秒） |
| `--gap-sec` | `0.4` | ターン間の無音（秒） |
| `--manifest-name` | `synthetic_moshi_train.jsonl` | manifest ファイル名 |

---

### Synthetic training data: Gemma 4 + Moshi TTS

`scripts/generate_synthetic_moshi_training_data.py` は Gemma 4 でスクリプトを生成してから
Moshi TTS で音声化する、より高機能なパイプラインです。
GPU サーバー上での実行を想定しており、Moshi と Gemma を別仮想環境で動かします。

The default mode is `scripted-moshi-tts`:

1. Gemma 4 generates a Japanese loneliness/isolation support dialogue script.
2. The main script calls Gemma through `scripts/gemma_dialogue_worker.py` as a
   subprocess. No separate API server is required.
3. Kyutai/Moshi TTS renders every script turn into audio.
4. The script places counselor/Moshi turns in the left channel and user turns
   in the right channel, then writes stereo WAVs, per-WAV timestamp JSON, a
   manifest JSONL, and a `dialogues.jsonl` log.

Moshi and Gemma still use separate Python environments because current Moshi
packages can require `huggingface-hub<1.0`, while newer Gemma 4 support in
Transformers can require a newer Hugging Face stack.  This is not an external
server: the dataset program launches the Gemma worker process itself.  Set up
both environments once:

Run a tiny three-case test from the cloned `mos` folder:

```bash
# Moshi + TTS environment
uv sync

# Gemma-only runtime, isolated from Moshi dependencies
uv sync --project gemma_runtime

uv run python scripts/generate_synthetic_moshi_training_data.py \
  --out-dir data/synthetic_loneliness_test \
  --num-dialogues 3 \
  --mode scripted-moshi-tts \
  --gemma-backend transformers-subprocess \
  --gemma-model google/gemma-4-E2B-it \
  --moshi-tts-repo kyutai/tts-1.6b-en_fr
```

`HF_TOKEN` is not required by this script. If the model download needs Hugging
Face credentials, run `huggingface-cli login` inside the relevant environment.
For an offline/check-format run, use `--gemma-backend template`, or
pre-generate dialogue JSONL and pass it with `--dialogues-jsonl`.

Important mode distinction:

- `scripted-moshi-tts`: Gemma's script is the transcript, and Moshi/Kyutai TTS
  synthesizes that script. Use this for script-faithful training data.
- `moshi-selfplay`: user turns are rendered with local TTS, then
  `llm-jp/llm-jp-moshi-v1` improvises the counselor stream from the user
  audio. Use this to sample Moshi behavior, not to force a fixed script.
- `scripted-local-tts` or `scripted-stereo`: local TTS renders both speakers
  for quick format checks.

The default Kyutai/Moshi TTS checkpoint above is the official open TTS model.
If you have a Japanese-capable Moshi TTS checkpoint, pass it with
`--moshi-tts-repo`.

`llm-jp/llm-jp-moshi-v1` is a full-duplex spoken dialogue model, not a
script-faithful text-to-speech checkpoint.  In this tool it is used only by
`--mode moshi-selfplay`, where it generates its own counselor response from the
user audio context.

Output shape:

```text
data/synthetic_loneliness_test/
├── synthetic_moshi_train.jsonl
├── dialogues.jsonl
├── generation_run.json
├── data_stereo/
│   ├── sample_001_*.wav
│   ├── sample_001_*.json
│   └── ...
└── sample_metadata/
```

The manifest lines follow the `moshi-finetune` convention:

```json
{"path": "data_stereo/sample_001_smalltalk_evening_001.wav", "duration": 42.1}
```

The WAV channel convention is:

- left channel: Moshi / counselor stream
- right channel: user stream

For a lighter format check that does not load Moshi TTS, render both speakers
from the Gemma script with local TTS:

```bash
uv run python scripts/generate_synthetic_moshi_training_data.py \
  --out-dir data/synthetic_scripted_test \
  --num-dialogues 3 \
  --mode scripted-local-tts \
  --gemma-backend template
```

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
| `--tts-speed` | `1.25` | Speed multiplier for `pyopenjtalk` prompt audio |
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

By default, Moshi runs in its normal full-duplex streaming mode: user audio is
fed frame by frame, and Moshi can respond at any point while the prompt and
appended silence are being fed. To test faster user prompts, increase
`--tts-speed`.

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
