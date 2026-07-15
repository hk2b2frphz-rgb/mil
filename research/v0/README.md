# v0 — 単発LLM対話生成 + Kyutai/Moshi 直接TTS

コミット `b0b30b0`(「Show Gemma subprocess progress in real time」、Qwen3-TTS導入
直前)時点のスナップショット。

## 構成

- `generate_synthetic_moshi_training_data.py` — LLM（Gemma、単発プロンプト）で
  対話スクリプトを生成し、そのまま以下のいずれかのモードで音声化する:
  - `scripted-moshi-tts`（既定）: Kyutai/Moshi TTS でスクリプトをステレオWAV化
  - `moshi-selfplay`: ローカルTTSで user 音声を作り、llm-jp-moshi-v1 に
    それを聞かせて相談員側の応答をモデル自身に生成させる
  - `scripted-local-tts` / `scripted-stereo`: pyopenjtalk/pyttsx3 で両話者を
    ローカルTTSレンダリング（フォーマット確認用の軽量モード）
- `gemma_dialogue_worker.py` — `--llm-backend transformers-subprocess` 用の
  LLMサブプロセスワーカー
- `response_recorder.py` — Moshi のロード・推論・WAVデコードのヘルパー
  （`moshi-selfplay`/`scripted-moshi-tts` の音声合成が依存）

まだ Qwen3-TTS もマルチエージェント対話生成も存在しない、最初の
エンドツーエンド版（対話生成→音声化→Moshi LoRA fine-tune、
`a7ad25e` 相当）。

## 実行について

当時のインターフェースは以下だった:

```bash
uv run python scripts/generate_synthetic_moshi_training_data.py \
    --out-dir ./output_v0 \
    --mode scripted-moshi-tts \
    --num-dialogues 3
```

このファイルは `REPO_ROOT = Path(__file__).resolve().parents[1]` でリポジトリ
ルートを推定し、そこから `response_recorder` と
`tests/fixtures/listening_dialogues.jsonl` を読む前提で書かれている。
`scripts/` 直下に置かれていることが前提のため、この `research/v0/` に置いた
コピーをそのまま実行することはできない（`response_recorder.py` は同梱して
いるが、想定パスが1階層ずれる）。差分参照用のアーカイブと理解すること。

## この後どう変わったか

- Qwen3-TTS が導入され、TTSレンダリングが専用ファイル
  ([v1.0](../v1.0/generate_qwen3_tts_data.py)) に分離された。
- 対話生成が単発プロンプトから userAI/moshiAI/judgeAI/aizuchiAI の
  マルチエージェント方式に置き換わった（[v1.0](../v1.0/)）。
