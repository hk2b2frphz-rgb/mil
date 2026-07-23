# v1.0 — マルチエージェント対話生成 + Qwen3-TTS（per-turn合成）

コミット `4234fdf`(「Add Qwen 3000-dialogue generation PBS script」、
whole-utterance合成導入直前)時点のスナップショット。

## v0 からの変化

1. **対話生成がマルチエージェント方式に**: `generate_synthetic_moshi_training_data.py`
   に userAI（利用者）・moshiAI（応答者）・judgeAI（対話完了判定）・aizuchiAI（相づち
   挿入位置決定）の4エージェントを vLLM 等の OpenAI-compatible サーバー越しに走らせる
   `--dialogue-generation-mode multi-agent` を追加。`--mode dialogues-only` で
   音声化せず `dialogues.jsonl` だけ生成できるようになった。
2. **TTSが Qwen3-TTS に置き換え**: `generate_qwen3_tts_data.py`（新規ファイル）が
   `dialogues.jsonl` を読み、Qwen3-TTS CustomVoice でターンごとに音声合成する。
   話者ごとの発話をまとめて1回で合成する whole-utterance 方式はまだ無い。
3. `build_use_cases.py` / `enrich_dialogue_timing.py` が追加され、対話ケースの
   軸出し（年齢・性格・今日の感情状態など）と、生成後の対話への相づち後付け
   （句切れごとの overlap 挿入）が分離された。

## 構成

- `build_use_cases.py` — 対話ケース(`use_cases.jsonl`)を軸の組み合わせで生成
- `generate_synthetic_moshi_training_data.py` — マルチエージェントで
  `dialogues.jsonl` を生成（`--mode dialogues-only --dialogue-generation-mode
  multi-agent`）
- `enrich_dialogue_timing.py` — 生成済み対話に相づち・タイミングを後付け
- `generate_qwen3_tts_data.py` — Qwen3-TTS でステレオWAV化（per-turn合成）
- `gemma_dialogue_worker.py` — ローカルLLMサブプロセス版バックエンド（互換維持）
- `run_dialogues_qwen_{smoke,1000,3000,10000}.pbs` — 対話生成のみのPBSジョブ
- `run_qwen_tts_{smoke,1000_4gpu}.pbs` — TTSレンダリングのPBSジョブ

## パイプライン

```
use_cases.jsonl → dialogues.jsonl → dialogues_enriched.jsonl → ステレオWAV
build_use_cases.py  generate_synthetic_moshi_training_data.py   enrich_dialogue_timing.py   generate_qwen3_tts_data.py
                     --mode dialogues-only --dialogue-generation-mode multi-agent
```

## この後どう変わったか

話者ごとの発話を連結して1回のTTSで合成し、MMS_FA強制アライメントで
セグメント境界を復元する whole-utterance 方式が導入された（[v1.1](../v1.1/)）。
per-turn合成は韻律が発話ごとに途切れる問題があった。
