# 実験設計 — "Do Full-Duplex Speech Models Know When to Ask?"

状態: 確定候補 / 更新日: 2026-07-13 (英語プライマリ・マルチコーパス・マルチモデル化)

## 1. リサーチクエスチョン

- **RQ1 (測定)**: ユーザー発話のスロット該当区間が音響的に劣化したとき、
  E2Eフルデュプレックスモデルは聞き返すか、それとも誤った解釈で進行するか。
- **RQ2 (較正)**: 聞き返し判断は劣化度(SNR)・情報の回復可能性(弱ASRオラクル)
  に対して較正されているか。
- **RQ3 (介入)**: 合成聞き返し対話によるFTで、較正された聞き返し行動を
  ターンテイキング性能を損なわずに注入できるか。
- **RQ4 (機構)**: その行動は音響的証拠に条件付くのか、語彙的手掛かりの
  模倣なのか(最小ペア学習 clarify_full vs 語彙のみ clarify_lexical の対照)。
- **RQ5 (般化)**: 行動は (a) 合成音声→実音声 (MASSIVE-TTS→SLURP)、
  (b) 言語間 (en→ja は学習せず、両言語で同一処方を並行検証) に般化するか。

## 2. タスクとプロトコル

音声アシスタントのタスク発話を1発話+最大1修復交換の閉ループで評価する。

```
user:  "wake me up at ((5 pm)) tomorrow"      ← (( )) 区間のみ劣化
model: (a) "5 pm, got it. I'll set the alarm."   = confirm(正解: clean時)
       (b) "Sorry, what time was it?"             = clarify(正解: 劣化時)
       (c) "7 pm, got it. Setting the alarm."     = hallucinated confirmation(失敗)
       (d) 無応答/雑談                             = other(失敗)
--- (b)の場合のみ、修復発話を注入 ---
user:  "It's 5 pm."
model: "5 pm, got it."                            = 回復成功
```

閉ループドライバ(`clarify/driver.py`)はモデル自身のテキストストリームから
ターン終端(1.2秒無テキスト)を検出し、規則分類器(言語パック
`clarify/lang.py` の語彙)が聞き返しと判定した場合のみ事前合成済み修復発話を
フレーム同期で注入する。**評価時の判定はオフラインで再計算**し、規則とLLM
judgeの一致率(κ)も報告する(オンライン判定は注入のゲート)。

## 3. コーパス (3系統)

| ベンチマークID | 言語 | コーパス | 担体音声 | 役割 |
|---|---|---|---|---|
| bench_en_massive | en | MASSIVE en-US test | Qwen3-TTS | **主実験** (60発話×9条件+underspec 8) |
| bench_en_slurp | en | SLURP test (実収録) | 実音声(close-talk) | 合成→実音声般化 (RQ5a)。FTはSLURP非接触 |
| bench_ja_massive | ja | MASSIVE ja-JP test | Qwen3-TTS | 言語般化 (RQ5b)、llm-jp-moshi/J-Moshi評価 |

- MASSIVE (CC BY 4.0) はSLURPのローカライズなので intent/slot 体系が3系統で
  同一 → コーパス間比較が指標定義そのままで成立する。
- スロット区間特定: MMS_FA強制アライメント(実音声にも適用)。失敗時は
  文字比例配分(manifestに記録、`span_alignment_fallbacks` で監視)。
- 劣化条件(全コーパス共通): clean / babble SNR {+5,0,−5,−10} /
  mask_silence / mask_noise / lowpass800 / full_snr0 (=同エネルギーを発話
  全体に分散する対照)。
- underspecified arm (意味的曖昧・clean音声) は MASSIVE 系のみ(SLURPは
  実収録の書き起こしを改変できないため)。
- 弱ASRオラクル(faster-whisper small, greedy): 劣化の実効性の操作チェック
  兼、較正分析の難易度軸、カスケードの信頼度信号。

## 4. モデル (7+系統)

| ID | システム | 言語 | 役割 |
|---|---|---|---|
| moshiko | kyutai/moshiko-pytorch-bf16 zero-shot | en | **主対象** (英語Moshi・男声) |
| moshika | kyutai/moshika-pytorch-bf16 zero-shot | en | モデル汎化(同アーキ別音声) |
| llmjp | llm-jp/llm-jp-moshi-v1 zero-shot | ja | 言語般化 |
| jmoshi | nu-dialogue/j-moshi-ext zero-shot | ja | 言語般化(別学習系) |
| task_only | moshiko + タスクFT(聞き返しなし) | en | タスク形式の交絡除去。**これとclarify系の差だけが聞き返し学習の効果** |
| clarify_lexical | + 語彙的曖昧のみ聞き返し学習 | en | RQ4: 語彙相関だけで音響劣化に般化するか(仮説: しない) |
| clarify_full | + 音響劣化最小ペア学習 | en | 提案。音響条件付き聞き返し |
| cascade_{small,medium} | faster-whisper + 信頼度閾値スイープ | en/ja | 明示的信頼度シグナルの到達点(ROC曲線として対置) |
| (拡張) clarify_full_ja | llm-jp + 同処方 | ja | 処方の言語不変性(時間があれば) |

FT基盤: 既存LoRA sweep (`experiments/lora_moshiko_en_config` = moshikoベース /
`lora_base_config` = llm-jpベース)。

## 5. アブレーション

| ID | 操作 | 検証すること |
|---|---|---|
| A1: FT variant三角形 | task_only / clarify_lexical / clarify_full | 聞き返しの学習源(タスク形式 vs 語彙 vs 音響) = RQ4本体 |
| A2: 最小ペア除去 | clarify_full − clean twin (`--no-minimal-pairs`) | 同一テキストのclean/劣化対照が音響条件付けの学習に必要か |
| A3: 軽劣化confirm除去 | `--mild-noise-confirm-ratio 0` | 「雑音=尋ねる」ショートカットへの崩壊(clean FA率で観測) |
| A4: ask比率 | `--ask-ratio {0.2, 0.4, 0.6}` | 聞き返し率のクラスバランス感度(FA/hitのトレードオフ) |
| A5: 局所 vs 全体劣化 | 評価条件 full_snr0 vs babble_snr0 | 判断が「区間の情報喪失」に応答しているか「全体品質」か |
| A6: 合成→実音声 | FT(MASSIVE-TTS) → eval(SLURP実音声) | 処方が実音声に転移するか = RQ5a |
| A7: 検出器 | 規則 vs LLM judge (κ + judge基準の再集計) | 測定の頑健性 |
| A8: TTS話者 | eval側 `--tts-speaker` 切替 | 話者過適合の感度分析 |

A1は必須(主結果表)、A2/A3/A5/A6/A7は本文、A4/A8は付録想定。

## 6. 評価指標 (`clarify/metrics.py`)

| 指標 | 定義 | 答えるRQ |
|---|---|---|
| CRR / T-CRR | 聞き返し率 / 対象スロットを特定した聞き返し率 | RQ1 |
| HCR | 聞き返さず誤値で進行(hallucinated confirmation)率 | RQ1 (危険側の失敗) |
| hit / FA / balanced acc | ask条件でのCRR vs clean条件でのCRR | RQ1-2 |
| CRR-vs-SNR 曲線 + オラクル回復可能性との一致 | 較正 | RQ2 |
| SSR | 最終確定ターンに正解スロット値(≤1修復) | タスク成功 |
| 選択的リスク/カバレッジ | 聞き返さなかった試行の誤り率 | RQ2 |
| 応答レイテンシ、FDB-JA全指標(P6; ja系のみ) | インタラクティビティ退行 | RQ3 |
| 規則 vs LLM judge κ | 測定の妥当性 | A7 |

集計は condition別に加え corpus別 (`by_corpus`)・言語別 (`by_language`) を
標準出力。統計: 比率は Wilson 95%CI、モデル間比較は base_id 対応あり
ブートストラップ(10k回、`paired_bootstrap_delta`)。

### 検定力の目安

主要対比(モデルA vs B の hit rate、n=180 ask試行、対応あり)で、
差15pt・基準50%を McNemar 近似で検出する検定力 > 0.9 (α=0.05)。
条件別 CRR (n=180/条件) の Wilson 95%CI 幅は ±7pt 程度。

## 7. FT学習データ (P3, `clarify/train_data.py`)

- 素材: MASSIVE en-US **train** split(testと表層値の重複なし、SLURPは完全
  ホールドアウト)から1,200発話 + underspecifiedテンプレート(言語パック)。
- 3 variant(§4)。clarify_full は **最小ペア**(同一発話がclean→confirm と
  劣化→ask の両方で出現、pair_id連結)を含み、劣化はユーザーチャネル音声に
  のみ適用(テキストストリームは同一)→ 音響を見なければ解けない学習信号。
- confirm側の25%に軽劣化(babble +5dB)を混ぜ、「雑音=尋ねる」という
  ショートカットを防ぐ(A3で除去検証)。
- 生成: 決定的テンプレート(言語パック; 再現性優先)→ 既存
  `generate_qwen3_tts_data.py`(whole-utterance+強制アライメント、
  `--language English --speaker-user Ryan`)でステレオ化 →
  `corrupt_training_audio.py` がスロット区間を後処理劣化。
- 学習: 既存 LoRA sweep 基盤 (A100 1枚, h01パターン)。必要なら full-FT
  (`fullft_sweep.pbs`) も同一データで可。

## 8. 実行計画(PBS、全てV100/A100以下)

| Job | GPU | 内容 | 目安 |
|---|---|---|---|
| p0_smoke | V100×1 | E2E疎通(en+jaデモ) | 〜3h |
| p1_build_benchmark ×3コーパス | V100×1 | ベンチ構築+オラクル | 〜6h/コーパス |
| p2_eval_moshi ×(4 zero-shot + 3 FT + アブレーションFT) | V100×4 (array) | 閉ループ評価 | 〜12h/モデル/コーパス |
| p3_build_train_data ×(3 variant + A2 + A3) | V100×1 | FTデータ | 〜10h/variant |
| p4_train_clarify ×5 | A100×1 | LoRA FT | 〜6h/variant |
| p5_eval_cascade ×コーパス | V100×1 | カスケード | 〜2h |
| p6_fdb_regression ×ja FTモデル | V100×1 | FDB-JA退行 | 〜12h |
| p7_summarize | CPU | 集計・比較 | 分 |

主要評価マトリクス: {moshiko, moshika, task_only, clarify_lexical,
clarify_full} × {bench_en_massive, bench_en_slurp} + {llmjp, jmoshi} ×
bench_ja_massive ≈ 12 モデル×コーパス組 ≈ V100 300〜400 GPU時間(1〜2週間の
キュー消化で現実的)。優先順位は §10 タイムライン参照。H200不要。
LLM judge はローカルPC(Azure)で `judge_pack*.jsonl` に対して実行
(クラスタからの外部API呼び出しなし — リポジトリ方針)。

## 9. 既知のリスクと緩和

1. **ベースモデルがタスク形式に全く乗らない**(confirmもaskもせず雑談)
   → それ自体がRQ1の知見。task_only FTが「形式は学べる」対照。
2. **英語MoshiのFT基盤未検証** → `lora_moshiko_en_config` はkyutai公式
   moshi-finetuneの標準構成。p4のsmoke(小max_steps)を最初に回す。
3. **SLURP実音声のアライメント失敗率** → manifestの
   `span_alignment_fallbacks` で監視、比例フォールバック分はサブグループ
   分析から除外可能。close-talk収録を優先選択して失敗率を下げる。
4. **オンライン聞き返し検出の誤り** → 注入ゲートのみに影響。judge一致率で
   定量化(A7)。
5. **TTS話者過適合** → A8話者切替 + SLURP実音声(多話者)が実質の頑健性検証。
6. **generate_qwen3_tts_data.py サイドカーschema不整合**
   → corrupt_training_audio.py は失敗率>5%でジョブを落とす。
7. **英語TTSがpyopenjtalkにフォールバック** → build_benchmark が
   language=en では即エラーにする(黙って日本語TTSで英文を読む事故防止)。

## 10. タイムライン(ICASSP 2027 締切: 2026-09-16)

| 週 | マイルストーン |
|---|---|
| 7/13週 | p0疎通(en+ja) → bench_en_massive構築 → moshiko/moshika評価(RQ1初期数値) |
| 7/20週 | p3/p4で en FT 3種 → 評価(RQ3/4 主結果)。bench_en_slurp構築 |
| 7/27週 | SLURP転移(A6)・cascade・A2/A3アブレーション・judge |
| 8/3週 | ja系(llmjp/jmoshi評価、余力でclarify_full_ja)・FDB退行・追試(A4/A8) |
| 8月中旬〜 | 図表確定・執筆 |
| 9月上旬 | 内部レビュー → 提出 |
