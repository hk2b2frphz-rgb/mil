# 0002: 冒頭挨拶の固定音声化と相づち・ミラーリングの再設計

状態: 実装済み / 更新日: 2026-07-02

## 文脈
生成データを聴くと (1) moshi の「そうなんですね」「なるほど」が多すぎて事務的、
特に「なるほど」が耳につく、(2) 相手の話すスピードや感情に moshi の声が
追従しない、(3) 復唱や「えぇ」「あぁ…」のような感情のこもった反応が無い、
(4) 冒頭挨拶が対話ごとの style preset で毎回違う韻律になる、という問題があった。

## 決定
1. **冒頭挨拶を固定**: `OPENING_GREETING_TEXT =「もしもし、こちら孤独孤立相談窓口になります。」`
   に変更。音声は run の最初に固定話者（`--speaker-moshi`）+ 固定 instruct
   （`--opening-greeting-instruct`）で **1回だけ合成**し、
   `data/.cache/opening_greeting/` にキャッシュして全対話・全 shard で
   **同一波形を使い回す**。whole-utterance 連結からも除外する。
2. **「なるほど」廃止・「そうなんですね」制限**: 生成側の相づち語彙を
   はい／ええ／えぇ／あぁ…／そうでしたか に変更。「なるほど」は生成禁止
   （TTS 側の overlap 検出セットには旧データ互換のため残す）。
   「そうなんですね」は1対話1回まで。
3. **復唱（エコー）**: user のキーワードを短くそのまま返す行
   （「十五年、ですか…。」）を1対話1〜3回入れるようプロンプトと few-shot
   fixture に組み込み。
4. **ミラーリング**: `--style-preset auto` を追加。dialogues.jsonl の
   `emotional_state`（新フィールド）または use case id の状態トークンから
   対話ごとにプリセットを選び、user と moshi の声のテンポ・トーンを対にする
   （high_tension / agitated / withdrawn プリセットを新設）。PBS ジョブの
   既定を `random` → `auto` に変更。
   `enrich_dialogue_timing.py` の相づち注入も感情群別プールに変更
   （沈んだ話には「あぁ…」、明るい話には「へえ」「いいですね」等）。

## 理由
- 「なるほど」は理解の表明としては事務的で、傾聴の相づちとしては冷たく
  聞こえる（利用者フィードバック）。感情に合わせた声漏れ系（あぁ…）や
  静かな同意（ええ／えぇ）の方が自然。
- 相づち・応答のトーンは話者間で同調する（ミラーリング）。preset を
  random にすると本文の感情と声の感情が食い違うサンプルが混ざる。
- 挨拶を毎回合成すると韻律が揺れ、「毎回同じ第一声を言う」学習目標が
  薄まる。固定波形の再利用は一貫性と生成コストの両方で有利。

## 代替案
- 相づちをターン単位で別話者コンテキスト合成 → 韻律の不整合が出るため
  whole-utterance + overlap 方式を維持。
- 「そうなんですね」全面禁止 → 1回までは自然なので残す。

## 影響・振り返り
- 旧 dialogues.jsonl（挨拶なし・なるほど入り）もそのまま TTS 可能
  （overlap 検出セットは上位集合、auto preset は id トークンから解決）。
- 評価側 `greeting_similarity` は新テキストで判定される
  （`eval/build_full_duplex_ja_dataset.py` のフォールバック文字列も同期済み）。
- <TODO: 新データで学習後、挨拶再現率と相づち分布を検証して追記>

## 関連
- コード: [scripts/generate_qwen3_tts_data.py](../../scripts/generate_qwen3_tts_data.py),
  [scripts/generate_synthetic_moshi_training_data.py](../../scripts/generate_synthetic_moshi_training_data.py),
  [scripts/enrich_dialogue_timing.py](../../scripts/enrich_dialogue_timing.py),
  [tests/test_opening_greeting_and_style.py](../../tests/test_opening_greeting_and_style.py)
- 概念: あいづち（Maynard 1989 / Den et al.）、ミラーリング（対人同調）
