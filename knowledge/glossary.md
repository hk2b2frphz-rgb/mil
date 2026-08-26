# 用語集

状態: 下書き / 更新日: 2026-07-01

プロジェクト内で頻出する用語。定義は本repoでの使われ方に合わせる。詳しい概念は
`concepts/` の各ファイルへ。

| 用語 | 説明 | 関連 |
|---|---|---|
| **全二重 (full-duplex)** | 話者双方が同時に話す/聞ける対話様式。Moshiはユーザ入力を聞きながら発話できる。 | Moshi, Full-Duplex-Bench |
| **相槌 / あいづち (aizuchi)** | 傾聴を示す短い応答（「うん」「そうなんだ」）。`--auto-overlap-aizuchi` で先行ユーザ発話に重ねる。 | backchannel |
| **backchannel** | 相槌の英語表現。話者交替を伴わない聞き手の反応。 | aizuchi |
| **ターンテイキング (turn-taking)** | 発話権の交替。交替潜時・割り込み・沈黙の扱いが評価対象。 | Full-Duplex-Bench |
| **whole-utterance モード** | 話者ごとに全発話を連結して1回TTS合成し、MMS_FA強制アライメントで境界復元。韻律が一貫。 | `generate_qwen3_tts_data.py` |
| **強制アライメント (forced alignment)** | 音声とテキストの時間対応付け。ここではMMS_FA (CTC) を使用。 | MMS, CTC |
| **CTC target too long** | 連結発話が生成長上限で打ち切られると音声<テキストになりCTCが失敗。`--whole-utterance-max-chars` と spare dialogues で対策。 | `generate_qwen3_tts_data.py` |
| **cascade** | ASR→LLM→TTS を繋ぐ非end-to-end方式。比較ベースライン。 | Whisper, Gemma, Qwen3-TTS |
| **SpeechLLM** | 音声を直接理解するLLM(Qwen2-Audio)ベースの比較系。 | Qwen2-Audio |
| **GRPO** | Group Relative Policy Optimization。価値関数不要のRL整合。本ブランチの中核。 | [concepts/grpo.md](concepts/grpo.md) |
| **LoRA / full-FT** | 低ランク適応 / 全パラメータ微調整。二本立てで運用。 | LoRA, ZeRO |
| **ZeRO-3 / offload** | DeepSpeedのメモリ最適化。optimizer/paramをCPUへ退避しA100×2 OOMを回避。 | `configs/deepspeed_zero3_*.json` |
| **keep-best-only** | eval loss最小のcheckpointのみ残す剪定。full-FT/LoRA両方に実装。 | — |
| **opening greeting** | 学習/評価で先頭に固定するmoshiの開始挨拶ターン。`--no-opening-greeting`で無効。 | — |
| **shard / spare dialogues** | TTS並列生成の分割単位と、失敗補填用の予備対話（`SPARE_RATIO`）。 | 4GPU TTS pbs |
