# v1.2 — 冒頭挨拶の声色固定・感情ミラーリング・相づち再設計

現在の `scripts/` と同一内容（本フォルダは archive 用の複製）。
設計判断の詳細は
[knowledge/decisions/0002_fixed_greeting_and_aizuchi_redesign.md](../../knowledge/decisions/0002_fixed_greeting_and_aizuchi_redesign.md)
を参照。

## v1.1 からの変化

1. **冒頭挨拶を固定**: `OPENING_GREETING_TEXT` を
   「もしもし、こちら孤独孤立相談窓口になります。」に変更。音声はrunの最初に
   固定話者＋固定instructで1回だけ合成し、`data/.cache/opening_greeting/` に
   キャッシュして全対話で同一波形を使い回す（whole-utterance連結からも除外）。
2. **「なるほど」廃止・「そうなんですね」制限**: 相づち語彙を
   はい/ええ/えぇ/あぁ…/そうでしたか に変更。「なるほど」は生成禁止、
   「そうなんですね」は1対話1回まで。
3. **復唱(エコー)**: 相手のキーワードを短くそのまま返す行を対話プロンプトと
   few-shot例に追加。
4. **感情ミラーリング**: `--style-preset auto` を追加。
   `dialogues.jsonl` の `emotional_state` または use case id の状態トークンから
   対話ごとにプリセットを自動選択し、user/moshiの声のテンポ・トーンを対にする
   （high_tension/agitated/withdrawn プリセットを新設）。
   `enrich_dialogue_timing.py` の相づち注入も感情群別プールに変更。

## 構成

v1.1 と同じファイル構成（`build_use_cases.py` は変更なし、他4ファイルに変更あり）。

## この後

データ生成とは独立の軸として、GRPOによる応答性（間・相づち・割り込み）の
追加ファインチューニングを行う（[v2](../v2/)）。
