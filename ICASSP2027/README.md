# ICASSP2027 — Do Full-Duplex Speech Models Know When to Ask?

E2Eフルデュプレックス音声対話モデル(Moshi系)が、ユーザー発話のスロット
該当区間が音響的に劣化したとき**自らの不確実性に気づいて聞き返せるか**を
測定し、FTで注入する実験一式。**英語プライマリ**(kyutai Moshi)、日本語は
言語般化軸(llm-jp-moshi / J-Moshi)。

- リサーチクエスチョンと実験設計(コーパス/モデル/アブレーション行列):
  [docs/design.md](docs/design.md)
- 先行研究と新規性: [docs/related_work.md](docs/related_work.md)
- 論文構想: [docs/paper_outline.md](docs/paper_outline.md)
- 締切: **ICASSP 2027 = 2026-09-16**

## 30秒サマリ

```
user:  "wake me up at ((5 pm)) tomorrow"      ← (( )) 区間だけ babble/mask 劣化
期待:   clean → "5 pm, got it. I'll set the alarm."      (confirm)
        劣化  → "Sorry, what time was it?"                (clarify)
                → 修復発話 "It's 5 pm." を注入 → confirm
危険な失敗:     "7 pm, got it."                (hallucinated confirmation)
```

評価はモデル自身のテキストストリームで完結し(出力ASR不要)、聞き返し時のみ
修復発話をフレーム同期注入する**閉ループドライバ**が方法論の新規要素。
言語依存要素(検出語彙・正規化・テンプレート)は [clarify/lang.py](clarify/lang.py)
の LanguagePack に集約。

## 実験マトリクス

| 軸 | 内容 |
|---|---|
| コーパス | MASSIVE en-US (TTS, 主実験) / **SLURP** (en, 実音声, 合成→実転移) / MASSIVE ja-JP (TTS, 言語般化) |
| モデル | moshiko / moshika (en zero-shot)、llm-jp-moshi-v1 / j-moshi-ext (ja zero-shot)、FT 3種 (task_only / clarify_lexical / clarify_full)、whisperカスケード(閾値スイープ) |
| アブレーション | FT variant三角形 (A1)、最小ペア除去 (A2)、軽劣化confirm除去 (A3)、ask比率 (A4)、局所vs全体劣化 (A5)、合成→実音声 (A6)、規則vs judge (A7)、TTS話者 (A8) |

## 実行手順(すべてPBS、V100/A100のみ)

```bash
# 0. 疎通(en+jaデモ E2E; 最初に必ず)
qsub -V ICASSP2027/pbs/p0_smoke.pbs

# 1. ベンチマーク構築(コーパスごとに1回)
qsub -v BENCH_ID=bench_en_massive,LANGUAGE=en,CORPUS=massive \
     ICASSP2027/pbs/p1_build_benchmark.pbs
bash ICASSP2027/scripts/download_slurp.sh          # ログインノード(要ネット)
qsub -v BENCH_ID=bench_en_slurp,LANGUAGE=en,CORPUS=slurp,\
SLURP_JSONL=ICASSP2027/data/slurp/metadata/test.jsonl,\
SLURP_AUDIO_DIR=ICASSP2027/data/slurp/audio/slurp_real \
     ICASSP2027/pbs/p1_build_benchmark.pbs
qsub -v BENCH_ID=bench_ja_massive,LANGUAGE=ja,CORPUS=massive \
     ICASSP2027/pbs/p1_build_benchmark.pbs

# 2. 閉ループ評価(4-way array; モデル×ベンチごとに投入)
qsub -v BENCH_ID=bench_en_massive,MODEL_ID=moshiko ICASSP2027/pbs/p2_eval_moshi.pbs
qsub -v BENCH_ID=bench_en_massive,MODEL_ID=moshika,\
HF_REPO=kyutai/moshika-pytorch-bf16 ICASSP2027/pbs/p2_eval_moshi.pbs
qsub -v BENCH_ID=bench_ja_massive,MODEL_ID=llmjp,\
HF_REPO=llm-jp/llm-jp-moshi-v1 ICASSP2027/pbs/p2_eval_moshi.pbs
qsub -v BENCH_ID=bench_ja_massive,MODEL_ID=jmoshi,\
HF_REPO=nu-dialogue/j-moshi-ext ICASSP2027/pbs/p2_eval_moshi.pbs

# 3. FT学習データ(en 3 variant + アブレーション)
qsub -v VARIANT=clarify_full,LANGUAGE=en    ICASSP2027/pbs/p3_build_train_data.pbs
qsub -v VARIANT=clarify_lexical,LANGUAGE=en ICASSP2027/pbs/p3_build_train_data.pbs
qsub -v VARIANT=task_only,LANGUAGE=en       ICASSP2027/pbs/p3_build_train_data.pbs
qsub -v VARIANT=clarify_full,LANGUAGE=en,NO_MINIMAL_PAIRS=1,\
RUN_DIR=ICASSP2027/runs/train_en_ablate_nopairs ICASSP2027/pbs/p3_build_train_data.pbs
qsub -v VARIANT=clarify_full,LANGUAGE=en,MILD_NOISE_RATIO=0,\
RUN_DIR=ICASSP2027/runs/train_en_ablate_nomild ICASSP2027/pbs/p3_build_train_data.pbs

# 4. LoRA FT(A100; LANGUAGE=en は moshiko ベースを自動選択)
qsub -v SRC_RUN_DIR=ICASSP2027/runs/train_en_clarify_full,LANGUAGE=en \
     ICASSP2027/pbs/p4_train_clarify.pbs
#    → merge_lora.pbs でマージ後、p2 に MODEL_WEIGHT を渡して
#      bench_en_massive と bench_en_slurp(A6転移) の両方で評価

# 5. カスケードベースライン(信頼度閾値スイープ)
qsub -v BENCH_ID=bench_en_massive ICASSP2027/pbs/p5_eval_cascade.pbs
qsub -v BENCH_ID=bench_en_slurp   ICASSP2027/pbs/p5_eval_cascade.pbs

# 6. インタラクティビティ退行チェック(ja FTモデルのみ; 既存FDB-JA)
qsub -v MODEL_ID=clarify_full_ja,MODEL_WEIGHT=/path/consolidated.safetensors \
     ICASSP2027/pbs/p6_fdb_regression.pbs

# 7. 集計・モデル比較(CPUのみ; ログインノード直実行も可)
#    p2の出力は eval_<MODEL_ID>_<BENCH_ID>/
qsub -v RUNS="moshiko=ICASSP2027/runs/eval_moshiko_bench_en_massive clarify_full=ICASSP2027/runs/eval_clarify_full_bench_en_massive",\
OUT_ID=comparison_en_massive ICASSP2027/pbs/p7_summarize.pbs
```

LLM judge(規則分類の独立検証、A7)はローカルPCで
`ICASSP2027/runs/eval_*/judge_pack*.jsonl` を Azure に投げる
(クラスタから外部APIは呼ばない — リポジトリ方針)。

## 構成

```
clarify/                コアパッケージ(重依存はすべて遅延import)
  lang.py               言語パック(en/ja): 検出語彙・正規化・テンプレート
  corpora.py            MASSIVE(多言語)/SLURP(実音声)ローダ
  corruptions.py        スロット区間限定の音響劣化(決定的seed)
  slots.py              スロット抽出(コーパス共通bracket形式)・選択
  scenario.py           ケーススキーマ + manifest I/O
  detector.py           聞き返し/復唱の規則分類 + ターン分割
  driver.py             閉ループ状態機械 + Moshiストリーミング駆動
  metrics.py            CRR/T-CRR/HCR/SSR/hit/FA/選択的リスク
                        + corpus/言語別集計 + 対応ブートストラップ
  asr_oracle.py         弱ASR回復可能性オラクル(faster-whisper)
  cascade.py            ASR信頼度閾値カスケードベースライン
  train_data.py         FT対話生成(3 variant + A2/A3/A4 アブレーション)
  judge_pack.py         LLM judge用パック(en/jaプロンプト) + κ計算
scripts/                CLI + download_slurp.sh
pbs/                    p0〜p7(上記手順)
tests/                  pytest 62件(CPUのみで実行可)
runs/                   出力(git ignore)
```

## テスト

```bash
python -m pytest ICASSP2027/tests -q     # ローカル(Windows/CPU)で可
```

## 実装メモ

- `run_closed_loop_eval.py` は `--shard/--num-shards` でケースを
  ラウンドロビン分割。`summarize_results.py` が `scores*.jsonl` を
  重複排除しつつ統合するので、シャード欠落・再実行に頑健。
- 検出言語はケースの `base.language` から自動決定(CLIフラグ不要)。
  ベンチマークは言語混在manifestも扱える。
- 劣化はすべて `case_seed(base_id, condition)` 由来の固定seed →
  ベンチマークはコードから決定的に再構築可能。
- SLURPでは close-talk (headset) 収録を優先選択して劣化を載せる。
  underspecified arm は実収録を改変できないため MASSIVE 系のみ。
- `corrupt_training_audio.py` は span を特定できない対話が5%を超えると
  失敗する(未劣化データで静かに学習が走る事故の防止)。
- p2 の `MODEL_WEIGHT` 指定時、`moshi_lm_kwargs.json` を重みの隣から
  自動検出(full-FT export 規約と同じ)。
- 英語FTのベースconfigは `experiments/lora_moshiko_en_config`
  (p4 が LANGUAGE=en で自動選択)。
