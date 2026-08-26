# v1.1 — whole-utterance Qwen3-TTS 合成

コミット `2e07755`(feature/grpo-interactivity-alignment ブランチの、今回の
v1.2 改修が入る直前の HEAD)時点のスナップショット。

## v1.0 からの変化

- **whole-utterance合成**: `generate_qwen3_tts_data.py --whole-utterance` で、
  話者(user/moshi)ごとに全発話を連結して1回のTTS呼び出しで合成し、
  MMS_FA (torchaudio forced alignment) でセグメント境界を復元する方式に変更。
  ターンごとに独立合成すると韻律が発話ごとに途切れて不自然になる問題を解消。
  相づち(あいづち)は音声波形として相手の発話に自然に重ね(`overlap_previous`)、
  隣接する user ターンは相づちで橋渡しされている限り1つの連続音声にマージされる。
- `--style-preset {counseling_anxious, counseling_sad, ..., random}` で
  対話単位の声のトーン・話速を指定できるようになった（`random` は対話ごとに
  ランダム選択）。
- 本番ジョブが 1000/3000/10000 対話の3規模に整理された
  （**v1.1.0** = 1000, **v1.1.1** = 3000, **v1.1.2** = 10000。コードは同一で
  対話数とPBSジョブ設定のみ異なる）。

## 構成

- `build_use_cases.py` / `generate_synthetic_moshi_training_data.py`
  （マルチエージェント対話生成、v1.0と同じ役割）
- `enrich_dialogue_timing.py` — 相づち後付け（相づち語彙は v1.2 で
  感情別プールに再設計される前の単一プール版）
- `generate_qwen3_tts_data.py` — whole-utterance 合成対応版
- `run_qwen_tts_whole_utterance_smoke.pbs` — 動作確認用の小規模ジョブ
- `run_qwen_tts_whole_utterance_1000_4gpu.pbs` — **v1.1.0**(1000対話)
- `run_qwen_tts_whole_utterance_3000_4gpu.pbs` — **v1.1.1**(3000対話)
- `run_qwen_tts_whole_utterance_10000_4gpu.pbs` — **v1.1.2**(10000対話)

## この後どう変わったか

生成データを聴くと「そうなんですね」「なるほど」の連発が事務的で、声のトーンが
本文の感情と合っていない問題が見つかった。冒頭挨拶の声色固定・感情に応じた
相づち・声のミラーリングを加えたのが [v1.2](../v1.2/)。
