# 論文構想 (ICASSP 2027, 4ページ + 参考文献)

状態: 下書き / 更新日: 2026-07-10

## 仮タイトル案

1. **"Do Full-Duplex Speech Dialogue Models Know When to Ask?
   Measuring and Instilling Clarification under Acoustic Uncertainty"**
2. "When to Ask: Acoustically-Grounded Clarification in Full-Duplex
   Spoken Dialogue Models"
3. "Hearing That You Didn't Hear: Calibrated Clarification Requests in
   End-to-End Spoken Dialogue"

## アブストラクト骨子

カスケード型SDSはASR信頼度で確認・聞き返しを行ってきたが、E2Eフルデュプレックス
モデル(Moshi系)には明示的な信頼度信号がない。スロット区間限定の音響劣化と
閉ループ修復プロトコルによる初のベンチマークで、公開日本語フルデュプレックス
モデルが劣化時にほぼ聞き返さず誤値で進行すること(hallucinated confirmation)を
示す。最小ペア合成対話によるLoRA FTで、聞き返しが音響的証拠に較正されて出現し
(hit +Xpt / FA +Ypt)、タスク成功が Z pt 改善、ターンテイキング指標は退行しない
ことを示す。語彙のみのFT対照は音響劣化に般化せず、行動が音響条件付きである
ことを支持する。

## セクション構成

### 1. Introduction (0.75p)
- フック: フルデュプレックスモデルは「いつ話すか」を学んだが
  「聞き取れなかったと気づいて尋ねる」ことを学んだか?
- カスケードの信頼度ベース確認戦略(古典) → E2E化で消えた暗黙機能
- 貢献 (1)タスク+閉ループプロトコル (2)ベンチマーク数値 (3)FT処方と
  最小ペアによる音響条件付けの検証 (4)カスケードとの決定品質比較

### 2. Related Work (0.4p)
related_work.md §2 を圧縮。表は入れず1段落×4。

### 3. Benchmark & Protocol (0.9p)
- 図1: パイプライン全体図(劣化・閉ループ・判定) ← 最重要図
- 劣化条件・期待行動・弱ASRオラクル・指標定義(CRR/T-CRR/HCR/SSR/hit/FA)
- 閉ループドライバ(フレーム同期・ターン検出・修復注入)

### 4. Instilling Clarification (0.5p)
- 3 variant FTデータ(task_only / clarify_lexical / clarify_full)
- 最小ペア設計(音響のみ異なる) + 軽劣化confirm(ショートカット防止)

### 5. Experiments (1.2p)
- 表1: 主結果 — 条件×モデル CRR/SSR/HCR
  (moshiko/moshika/llmjp/jmoshi zero-shot, task_only, clarify_lexical,
  clarify_full, cascade)。コーパス列 = MASSIVE-en / SLURP / MASSIVE-ja
- 図2: CRR vs SNR(較正曲線; cascadeのROC上に各モデルの動作点)
- 図3: 選択的リスク-カバレッジ
- 表2: アブレーション(A2最小ペア除去 / A3軽劣化除去 / A6合成→実転移)
  + FDB-JA退行チェック + judge一致κ
- 分析: full_snr0 vs 局所劣化(局所性の必要性)、underspecified
  (意味的曖昧)への転移、T-CRRの内訳

### 6. Conclusion (0.15p)

## 主張と証拠の対応表

| 主張 | 証拠 |
|---|---|
| E2Eは聞き返せない(モデル・言語横断) | moshiko/moshika/llmjp/jmoshi の CRR≈0 + HCR高(表1) |
| タスクFTだけでは尋ねない | task_only の hit低 |
| 語彙学習は音響に般化しない | clarify_lexical: underspec hit高 / acoustic hit低 |
| 音響最小ペアで較正された聞き返し | clarify_full: SNR単調なCRR(図2)+FA低; A2除去で崩れる(表2) |
| ショートカットでなく情報喪失に応答 | A3除去でclean FA上昇 / full_snr0 vs 局所の対比 |
| 実音声へ転移 | MASSIVE-TTS学習 → SLURP実音声eval(A6, 表2) |
| 明示信頼度(カスケード)との差 | cascade ROC と各モデル動作点の距離 |
| 副作用なし | FDB-JA指標維持(表2) |

## 図表制作メモ

- 図1はp0出力の実波形(clean/corrupted)+実対話例を使う
- 図2はseed×caseのWilson CI帯付き折れ線
- スタイル: ICASSP 2列。図は matplotlib、`summarize_results.py` の
  comparison.json から生成(プロットスクリプトは執筆期に追加)

## 投稿情報

- ICASSP 2027 (Toronto, 2027-05-16〜21)、締切 **2026-09-16**
- トピック: Speech and Language Processing — Spoken Dialogue Systems /
  Human-Computer Interaction
- 再現性: コード+seed+manifest公開(TTS合成のためデータ再構築可能、
  MASSIVE は CC BY 4.0)
