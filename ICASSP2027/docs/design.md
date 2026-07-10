# 実験設計 — "Do Full-Duplex Speech Models Know When to Ask?"

状態: 確定候補 / 更新日: 2026-07-10

## 1. リサーチクエスチョン

- **RQ1 (測定)**: ユーザー発話のスロット該当区間が音響的に劣化したとき、
  E2Eフルデュプレックスモデルは聞き返すか、それとも誤った解釈で進行するか。
- **RQ2 (較正)**: 聞き返し判断は劣化度(SNR)・情報の回復可能性(弱ASRオラクル)
  に対して較正されているか。
- **RQ3 (介入)**: 合成聞き返し対話によるFTで、較正された聞き返し行動を
  ターンテイキング性能を損なわずに注入できるか。
- **RQ4 (機構)**: その行動は音響的証拠に条件付くのか、語彙的手掛かりの
  模倣なのか(最小ペア学習 clarify_full vs 語彙のみ clarify_lexical の対照)。

## 2. タスクとプロトコル

音声アシスタントのタスク発話(MASSIVE ja-JP)を1発話+最大1修復交換の
閉ループで評価する。

```
user:  「明日の【15時】にアラームをかけて」   ← 【】区間のみ劣化
model: (a)「15時ですね。設定しますね」        = confirm(正解: clean時)
       (b)「すみません、何時でしたか？」      = clarify(正解: 劣化時)
       (c)「5時ですね。設定しますね」          = hallucinated confirmation(失敗)
       (d) 無応答/雑談                          = other(失敗)
--- (b)の場合のみ、修復発話を注入 ---
user:  「15時です」
model: 「15時ですね。設定しますね」            = 回復成功
```

閉ループドライバ(`clarify/driver.py`)はモデル自身のテキストストリームから
ターン終端(1.2秒無テキスト)を検出し、規則分類器が聞き返しと判定した場合のみ
事前合成済み修復発話をフレーム同期で注入する。**評価時の判定はオフラインで
再計算**し、規則とLLM judgeの一致率も報告する(オンライン判定は注入のゲート)。

## 3. ベンチマーク構築 (P1)

| 要素 | 設計 |
|---|---|
| 素材 | MASSIVE ja-JP **test** split、9 intent × 9 slot型からラウンドロビン60発話 + underspecified 8テンプレート |
| TTS | Qwen3-TTS (学習と同一; pyopenjtalkフォールバック)。話者は既定 Ono_Anna、話者汎化はseed間でなくTTS話者切替オプションで別途 |
| スロット区間特定 | MMS_FA強制アライメント(既存 ForcedAligner)。失敗時は文字比例配分(manifestに記録・除外分析可能) |
| 劣化条件 | clean / babble SNR {+5,0,−5,−10} / mask_silence / mask_noise / lowpass800 / full_snr0 (全9条件) |
| 対照条件 | full_snr0 = 同エネルギーの雑音を発話全体に分散 → 局所情報喪失と全体品質低下の切り分け |
| 意味的曖昧arm | underspecified 8件(clean音声・スロット欠落) → 「音響がきれいでも尋ねるべき」ケース |
| 期待行動ラベル | clean→act / mask_*・underspec→ask / SNR系→graded(オラクルで採点) |
| オラクル | faster-whisper small (greedy) の転写でスロット回復可否 + avg_logprob。劣化の実効性の操作チェック兼較正軸 |
| 再現性 | 全劣化はケースID由来の固定seed。データセットはコード+seedから決定的に再構築可能 |

規模: 60×8条件 + 60 clean + 8 underspec ≈ **548ケース × 3 seeds ≈ 1,644試行/モデル**。
1試行 ≈ 実時間20〜40秒(V100) → 1モデルあたり約11〜18 GPU時間 → **4-way
アレイジョブで1晩** (`p2_eval_moshi.pbs`、`-J 0-3`)。

### 検定力の目安

主要対比(モデルA vs B の hit rate、n=180 ask試行、対応あり)で、
差15pt・基準50%を McNemar 近似で検出する検定力 > 0.9 (α=0.05)。
条件別 CRR (n=180/条件) の Wilson 95%CI 幅は ±7pt 程度。

## 4. 評価指標 (`clarify/metrics.py`)

| 指標 | 定義 | 答えるRQ |
|---|---|---|
| CRR / T-CRR | 聞き返し率 / 対象スロットを特定した聞き返し率 | RQ1 |
| HCR | 聞き返さず誤値で進行(hallucinated confirmation)率 | RQ1 (危険側の失敗) |
| hit / FA / balanced acc | ask条件でのCRR vs clean条件でのCRR | RQ1-2 |
| CRR-vs-SNR 曲線 + オラクル回復可能性との一致 | 較正 | RQ2 |
| SSR | 最終確定ターンに正解スロット値(≤1修復) | タスク成功 |
| 選択的リスク/カバレッジ | 聞き返さなかった試行の誤り率 | RQ2 |
| 応答レイテンシ、FDB-JA全指標(P6) | インタラクティビティ退行 | RQ3 |
| 規則 vs LLM judge κ | 測定の妥当性 | 方法論 |

統計: 比率は Wilson 95%CI、モデル間比較は base_id 対応ありブートストラップ
(10k回、`paired_bootstrap_delta`)。

## 5. 比較システム

| ID | システム | 役割 |
|---|---|---|
| base | llm-jp-moshi-v1 zero-shot | E2Eの現状(仮説: CRR≈0, HCR高) |
| jmoshi | nu-dialogue/j-moshi-ext zero-shot | モデル汎化(同アーキ別学習) |
| task_only | base + タスクFT(聞き返しなし学習) | タスク形式の交絡除去。**これとclarify系の差だけが聞き返し学習の効果** |
| clarify_lexical | + 語彙的曖昧のみ聞き返し学習 | RQ4: 語彙相関だけで音響劣化に般化するか(仮説: しない) |
| clarify_full | + 音響劣化最小ペア学習 | 提案。音響条件付き聞き返し |
| cascade_small / (large) | faster-whisper + 信頼度閾値スイープ | 明示的信頼度シグナルの到達点(ROC曲線として対置) |

## 6. FT学習データ (P3, `clarify/train_data.py`)

- 素材: MASSIVE ja-JP **train** split(testと表層値の重複なし)から1,200発話
  + underspecifiedテンプレート。
- 3 variant(上表)。clarify_full は **最小ペア**(同一発話がclean→confirm と
  劣化→ask の両方で出現、pair_id連結)を含み、劣化はユーザーチャネル音声にのみ
  適用(テキストストリームは同一)→ 音響を見なければ解けない学習信号。
- confirm側の25%に軽劣化(babble +5dB)を混ぜ、「雑音=尋ねる」という
  ショートカットを防ぐ(較正の学習信号)。
- 生成: 決定的テンプレート(Gemma不要・再現性優先)→ 既存
  `generate_qwen3_tts_data.py`(whole-utterance+強制アライメント)で
  ステレオ化 → `corrupt_training_audio.py` がスロット区間を後処理劣化。
- 学習: 既存 LoRA sweep 基盤(kyutai moshi-finetune, A100 1枚, h01パターン)。
  必要なら full-FT (`fullft_sweep.pbs`) も同一データで可。

## 7. 実行計画(PBS、全てV100/A100以下)

| Job | GPU | 内容 | 目安 |
|---|---|---|---|
| p0_smoke | V100×1 | E2E疎通(демо4件) | 〜2h |
| p1_build_benchmark | V100×1 | ベンチ構築+オラクル | 〜6h |
| p2_eval_moshi ×モデル | V100×4 (array) | 閉ループ評価 | 〜12h/モデル |
| p3_build_train_data ×3 variant | V100×1 | FTデータ | 〜10h/variant |
| p4_train_clarify ×3 | A100×1 | LoRA FT | 〜6h/variant |
| p5_eval_cascade | V100×1 | カスケード | 〜2h |
| p6_fdb_regression ×FTモデル | V100×1 | FDB-JA退行 | 〜12h |
| p7_summarize | CPU | 集計・比較 | 分 |

合計 ≈ V100 100〜150 GPU時間 + A100 20 GPU時間。H200不要。
LLM judge はローカルPC(Azure)で `judge_pack*.jsonl` に対して実行
(クラスタからの外部API呼び出しなし — リポジトリ方針を踏襲)。

## 8. タイムライン(ICASSP 2027 締切: 2026-09-16)

| 週 | マイルストーン |
|---|---|
| 7/13週 | p0疎通 → p1本構築 → base/jmoshi評価(RQ1初期数値) |
| 7/20週 | p3/p4でFT3種 → p2でFT評価(RQ3/4 主結果) |
| 7/27週 | cascade・FDB退行・judge・分析、追試(seed/話者感度) |
| 8月 | 図表確定・執筆・(任意)英語Moshi+MASSIVE en-US一般化実験 |
| 9月上旬 | 内部レビュー → 提出 |

## 9. 既知のリスクと緩和

1. **ベースモデルがタスク形式に全く乗らない**(confirmもaskもせず雑談)
   → それ自体がRQ1の知見。task_only FTが「形式は学べる」対照。
   ベンチマークはFT群の比較で成立する設計。
2. **強制アライメント失敗率が高い** → manifestに記録、比例フォールバック、
   `span_alignment_fallbacks` が高ければ手動検査対象(短文なので失敗は稀と予想)。
3. **オンライン聞き返し検出の誤り** → 注入ゲートのみに影響。judge一致率で
   定量化し、誤ゲート試行はサブグループ分析で除外可能。
4. **TTS話者1名への過適合** → 評価TTS話者切替(`--tts-speaker`)での感度分析を
   追試に含める。
5. **generate_qwen3_tts_data.py のサイドカーschemaとの不整合**
   → corrupt_training_audio.py は失敗率>5%でジョブを落とす設計
   (静かに未劣化データで学習が走ることを防ぐ)。p0/小規模で先に検証。
