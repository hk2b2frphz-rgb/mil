# Full-duplex 直接関連の収集済み資料

作成日: 2026-08-28

`references/` 内の収集済み資料から、**full-duplex音声対話を直接の対象にする論文**だけを
抜粋した一覧です。評価の周辺技術である音声圧縮、TTS、ASR、RL、fine-tuningの論文は含めて
いません。

## 収集済み論文（3本）

| 論文 | 年 | このプロジェクトとの関係 | ローカルPDF |
|---|---:|---|---|
| [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) — Défossez et al. (Kyutai) | 2024 | 同時発話・相槌・割込みを扱うリアルタイム音声対話モデル。日本語Moshiの基盤。 | `references/01_moshi_fullduplex/defossez2024_moshi.pdf` |
| [Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities](https://arxiv.org/abs/2503.04721) — Lin et al. | 2025 | Full-Duplex-Bench-JAの評価プロトコルの原典。ターン交替、割込み、相槌などを評価する。 | `references/01_moshi_fullduplex/lin2025_full-duplex-bench.pdf` |
| [Generative Spoken Dialogue Language Modeling](https://arxiv.org/abs/2203.16502) — Nguyen et al. | 2022 | 発話者ごとの並列ストリームを生成する音声対話モデルで、full-duplex対話の先行研究。 | `references/01_moshi_fullduplex/nguyen2022_dgslm.pdf` |

## リリース・モデル資料

full-duplexに直接関係する公式リリース文書・モデルカードは、現在の収集物にはありません。
このプロジェクトで用いる `llm-jp/llm-jp-moshi-v1` のモデルカードも未保存です。

## 除外した資料

- EnCodec、AudioLM、SpeechGPT、Qwen2-Audio
- TTS、ASR、強制アライメント
- GRPO、RLHF、DPO、LoRA、ZeRO、QLoRA
- 日本語LLM一般および共感対話・孤独支援

これらは実装や学習には関係しますが、full-duplex対話そのものを主題にはしていないため、
本一覧からは除外しています。
