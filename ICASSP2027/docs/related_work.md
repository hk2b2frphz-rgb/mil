# 先行研究調査 — 聞き返し(Clarification)と不確実性 × フルデュプレックス音声対話

状態: 下書き / 更新日: 2026-07-10

## 1. リサーチギャップの要約

**「E2Eフルデュプレックス音声対話モデルは、聞き取れなかったときに聞き返せるか」を
測定・改善した研究は存在しない(2026年7月時点)。**

近接領域は以下の4象限に整理できる。本研究は右下の空白を埋める。

| | カスケード型 (ASR→LLM) | E2Eフルデュプレックス (Moshi系) |
|---|---|---|
| **頑健性(黙って耐える)** | ノイズ頑健ASR、GER訂正 (Denoising GER 等) | IRAF (arXiv:2606.06559): 干渉耐性fusion。「耐える」のみで「尋ねる」は扱わない |
| **聞き返し(尋ねる)** | Proactive for Uncertainty (arXiv:2605.25404): ASR潜在表現から誤り原因を診断し聞き返し。**ASR信頼度という明示信号に依存** | **← 本研究。明示的信頼度信号を持たないE2Eモデルが、音響的不確実性を行動(聞き返し)に変換できるかは未検証** |

## 2. 主要関連研究

### 2.1 フルデュプレックス音声対話モデル

- **Moshi** (Défossez et al., 2024): Mimi codec + 二重ストリームLM。
  本研究のベース。テキストストリームを自己書き起こしとして持つ点が
  評価設計上の利点(出力ASR不要)。
- **J-Moshi / Towards a Japanese Full-duplex Spoken Dialogue System**
  (Ohashi et al., Interspeech 2025; arXiv:2506.02979): 日本語初の公開
  フルデュプレックスモデル。llm-jp-moshi-v1 (本リポジトリのベース) も同系。
- **BayLing-Duplex** (arXiv:2606.14528)、**DuplexSLA** (arXiv:2605.20755)、
  **ASPIRin** (arXiv:2604.10065, RLでインタラクティビティ最適化):
  いずれもturn-taking/割り込みが焦点。理解の不確実性→聞き返しは扱わない。

### 2.2 フルデュプレックス評価ベンチマーク

- **Full-Duplex-Bench v1/v1.5** (Lin et al., 2025): pause/backchannel/
  turn-taking/interruption のタイミング評価。本リポジトリに日本語適応版
  (FDB-JA) が実装済み — 本研究のP6(インタラクティビティ退行チェック)で再利用。
- **Full-Duplex-Bench v2** (arXiv:2510.07838): 自動examinerによるマルチターン
  評価(correction handling, entity tracking)。**聞き返し・不確実性の評価軸は
  含まれない**(著者らの記述でも correction=ユーザー起点の訂正への対応であり、
  モデル起点の聞き返しではない)。
- **ICASSP 2026 HumDial Challenge** (arXiv:2604.21406): フルデュプレックス
  対話が ICASSP コミュニティの公式チャレンジになった。ICASSP 2027 への
  時流として好適。「動的なターン交渉」の次の課題として不確実性対応を
  位置づけられる。

### 2.3 聞き返し・不確実性(テキスト/カスケード)

- 古典: Purver (2004) CRの類型論、San-Segundo et al. / Bohus & Rudnicky の
  confidence-based confirmation戦略 — カスケードSDSの標準装備だった機能が
  E2E化で失われた、というのが本研究のframing。
- **Proactive for Uncertainty** (arXiv:2605.25404): 最近接。ASR潜在表現から
  perception/comprehension/deletion を診断→LLMが聞き返し戦略選択。
  カスケード限定。本研究はこれをE2Eフルデュプレックスに拡張する問いを立て、
  かつカスケード(ASR信頼度閾値)を明示ベースラインとして対置する。
- **CLAM / AbstentionBench / selective prediction 系**: LLMの「答えない」
  行動の較正。音響モダリティ・リアルタイム対話には未展開。
- **SpokenWOZ** (arXiv:2305.13040): 音声TODベンチマーク。ASR誤りは扱うが
  評価はテキスト側でモデルに聞き返しの機会を与えない。

### 2.4 音声LLMの不確実性・頑健性

- **Walking Through Uncertainty** (arXiv:2604.25591): audio-aware LLMの
  不確実性推定の実証研究(QA形式)。行動(聞き返し)への接続なし。
- **VocalBench-DF** (arXiv:2510.15406): 非流暢性への頑健性。
- **Cascade Equivalence Hypothesis** (arXiv:2602.17598): 雑音下では
  カスケードがE2Eを上回る、という報告。本研究のRQ4(音響条件付けの学習
  可能性)の動機。
- **NOVA** (arXiv:2601.11004): ノイズ考慮の言語的信頼度較正(RAG)。

### 2.5 SLU資源

- **MASSIVE** (FitzGerald et al., 2022; CC BY 4.0): 51言語・60 intents・
  55 slots、ja-JP完備。SLURPのローカライズなのでintent/slot体系は
  音声アシスタント文脈で標準的。ベンチマークのスロット素材に採用。
- SLURP (英語・実音声): 英語Moshiへの一般化実験(オプションP2)の候補。

## 3. 新規性の主張(論文のContribution候補)

1. **タスク定義**: acoustically-grounded clarification — スロット区間限定の
   音響劣化に対し「行動としての聞き返し」を要求する初のフルデュプレックス
   評価タスク。
2. **閉ループ評価プロトコル**: モデルが聞き返した場合のみ修復発話を注入する
   フレーム同期ドライバ(全自動・オフライン・再現可能)。既存ベンチマーク
   (FDB系)は固定入力のみで、モデル行動に応答する評価はFDB-v2のexaminer
   (商用API依存)以外になく、ローカル再現可能な実装は初。
3. **知見**: (仮説) ベースのフルデュプレックスモデルは聞き返さず誤った値で
   進行する(hallucinated confirmation)。カスケードはASR信頼度で尋ねられる。
   この「E2Eの自己認識ギャップ」の定量化。
4. **処方**: 最小ペア(同一発話・音響のみ劣化)を含む合成データによるFTで、
   語彙相関ではなく音響的証拠に条件付けられた聞き返しが学習可能かの検証
   (clarify_lexical vs clarify_full の対照で切り分け)。

## 4. 想定される査読上の攻撃と防御

| 攻撃 | 防御 |
|---|---|
| 「TTS合成音声で音響劣化を語れるのか」 | 弱ASRオラクルで劣化の実効性を独立に検証(操作チェック)。TTS品質はclean条件のSSRで担保を示す。実音声(SLURP)拡張をFuture workまたはP2で。 |
| 「規則ベースの聞き返し検出は恣意的」 | LLM judgeとの一致率(κ)を報告し、判定はjudge側でも再現。オンライン検出器は修復注入のゲートのみで、主結果はオフライン再計算。 |
| 「1つのモデル(Moshi系)の話では」 | J-Moshi/llm-jp-moshiの2チェックポイント+FT群+カスケード2種で系統比較。英語Moshi追加はP2。 |
| 「FTしたらタスクを暗記しただけ」 | ベンチマークはMASSIVE testスロット値・未知話者条件、trainとの surface 重複なし。さらに clean 条件のFA率で過剰聞き返しを罰する。 |
| 「聞き返し率が上がっただけでは」 | 主指標は決定品質(hit/FA・balanced acc)と選択的リスク+タスク成功(SSR)。FDB-JA退行チェックでインタラクティビティ維持も報告。 |

## 5. 検索ログ(主要ソース)

- Full-Duplex-Bench v2: https://arxiv.org/pdf/2510.07838
- HumDial Challenge (ICASSP 2026): https://arxiv.org/abs/2604.21406
- Proactive for Uncertainty: https://arxiv.org/pdf/2605.25404
- IRAF: https://arxiv.org/pdf/2606.06559
- ASPIRin: https://arxiv.org/pdf/2604.10065
- BayLing-Duplex: https://arxiv.org/html/2606.14528
- DuplexSLA: https://arxiv.org/pdf/2605.20755
- Walking Through Uncertainty: https://arxiv.org/pdf/2604.25591
- VocalBench-DF: https://arxiv.org/pdf/2510.15406
- Cascade Equivalence Hypothesis: https://arxiv.org/pdf/2602.17598
- J-Moshi: https://arxiv.org/abs/2506.02979
- SpokenWOZ: https://arxiv.org/abs/2305.13040
- MASSIVE: https://huggingface.co/datasets/AmazonScience/massive
- ICASSP 2027 CFP (deadline 2026-09-16): https://2027.ieeeicassp.org/call-for-papers/
