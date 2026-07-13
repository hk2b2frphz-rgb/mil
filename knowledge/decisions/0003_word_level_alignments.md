# 0003: alignments を発話単位から単語単位に変更（kyutai Interleaver 対応）

状態: 実装済み / 更新日: 2026-07-14

## 文脈
v1〜v1.1.1（30h/100h）の LoRA 学習でテキスト出力が崩壊し、full-FT でも
相槌・音声の自然性が低かった。原因調査の結果、sidecar JSON の
`alignments` が **発話（ターン）単位** `[ターン全文, [start, end], label]`
だったことが判明。

- **LoRA 経路（kyutai moshi-finetune）**: `Interleaver` は「1 エントリ = 1 単語」
  前提で、エントリ開始フレームから 1 フレーム 1 トークンで書き込む。
  発話単位だと (1) ターン冒頭に全文がバースト配置され音声と非同期、
  (2) 次エントリ開始時に書き残しトークンを **無警告で破棄**
  （`build_token_stream` の `to_append_stack = deque(tokenized)` 置換）、
  (3) 100 秒チャンク境界をまたぐ発話はテキスト丸ごと欠落。
  → 「文の途中で切れたテキスト」を教師として学習 → 出力崩壊。
- **full-FT 経路（nu-dialogue）**: `tools/tokenize_text.py` が発話区間へ
  文字数比例でトークンを分散するため、粗いが同期は保たれる
  （AUTO_PATCH_ROBUST_TEXT_ALIGNMENT）。full と LoRA で症状が違ったのはこのため。

## 決定
1. `scripts/alignment_words.py` を新設。ターン文を単語
   （pyopenjtalk 形態素、無ければ句読点境界+4 文字チャンクの無劣化分割）に
   分割し、[start, end] を文字数比例で内挿する。1 エントリ上限 8 文字。
2. `scripts/generate_qwen3_tts_data.py` は今後 **単語単位の alignments** を
   書き出す。発話単位の原本は `alignments_utterance` キーに保存し、
   `metadata.alignments_granularity = "word"` を付与。
3. 既存データは `scripts/split_alignments_to_words.py` で **音声再合成なしに**
   sidecar JSON だけをインプレース修復できる（冪等・dry-run あり）。
4. **FA 失敗時の proportional fallback を既定で廃止**（`--fa-fallback skip` が
   新既定）。フォールバックが発火すると連結音声の切り出し境界そのものが
   文字数比例の推定になり、ターンが単語の途中で切れた**破損音声**が無警告で
   学習データに混入していた。既定では該当対話を破棄し（`--success-target`
   併用で予備対話から自動補充）、末尾ログに FA 破棄件数を集計する。
   旧挙動は `--fa-fallback proportional`（デバッグ用）で残す。
5. 単語分割器は pyopenjtalk（形態素）→ regex-chunk（句読点+4文字）の順で
   選択し、どちらが使われたかを `metadata.alignments_word_split.segmenter`
   に記録する。時刻はどちらも文字数比例内挿なので品質差は小さいが、
   run 内で粒度を統一したい場合は修復 CLI の `--require-pyopenjtalk` を使う。

## 理由
- Moshi のテキストストリームは音声と同期した inner monologue が本質。
  kyutai の DailyTalk 例も単語単位で、単語単位が想定フォーマット。
- 文字数比例の内挿は nu-dialogue 版と同じ近似で、full-FT でテキストが
  崩壊しなかった実績がある。TTS 再合成不要で 30h/100h 資産を修復できる。

## 代替案
- MMS_FA の文字レベル span から真の単語時刻を出す（`ForcedAligner.align` は
  内部で持っている）→ 精度は上がるが whole-utterance 経路の改修が大きい。
  まず比例内挿で再学習し、不足なら次段で検討。
- `Interleaver(keep_and_shift=True)` に変更 → 破棄は防げるがバースト
  （非同期）は残るため不採用。

## 影響・振り返り
- `prepare_nu_fullft_dataset.py`（nu 経路）は単語単位でもそのまま動く
  （エントリごとに文字数比例分散するため結果は実質同一）。
- `grpo/segment_extractor.py` は `metadata.dialogue.turns` を参照するため影響なし。
- 修復後は LoRA を再学習して検証する。手順:
  `split_alignments_to_words.py --manifest <RUN>/training_set/synthetic_moshi_train.jsonl`
  → `run_sweep_pair.sh` の h01 相当を 1 本 → テキスト崩壊の有無を確認。
- <TODO: 再学習後、テキスト崩壊が解消したか・相槌/音声自然性の変化を追記>
- 留意: full-FT の「学習テキスト丸暗記」はデータ多様性とエポック過多の問題で
  本件とは別軸（エポック削減・早期終了・実対話混合で対処）。

## 関連
- コード: [scripts/alignment_words.py](../../scripts/alignment_words.py),
  [scripts/split_alignments_to_words.py](../../scripts/split_alignments_to_words.py),
  [scripts/generate_qwen3_tts_data.py](../../scripts/generate_qwen3_tts_data.py),
  [tests/test_alignment_word_split.py](../../tests/test_alignment_word_split.py)
- 参照: kyutai moshi-finetune `finetune/data/interleaver.py`（build_token_stream）,
  nu-dialogue `tools/tokenize_text.py`（文字数比例分散）
- 関連決定: [0002](0002_fixed_greeting_and_aizuchi_redesign.md)
