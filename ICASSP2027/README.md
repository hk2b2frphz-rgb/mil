# ICASSP2027 — Do Full-Duplex Speech Models Know When to Ask?

E2Eフルデュプレックス音声対話モデル(Moshi / llm-jp-moshi / J-Moshi)が、
ユーザー発話のスロット該当区間が音響的に劣化したとき**自らの不確実性に
気づいて聞き返せるか**を測定し、FTで注入する実験一式。

- リサーチクエスチョンと実験設計: [docs/design.md](docs/design.md)
- 先行研究と新規性: [docs/related_work.md](docs/related_work.md)
- 論文構想: [docs/paper_outline.md](docs/paper_outline.md)
- 締切: **ICASSP 2027 = 2026-09-16**

## 30秒サマリ

```
user:  「明日の【15時】にアラームをかけて」   ← 【】区間だけ babble/mask 劣化
期待:   clean     → 「15時ですね。設定しますね」(confirm)
        劣化      → 「すみません、何時でしたか？」(clarify) → 修復発話注入 → confirm
危険な失敗:        「5時ですね。設定しますね」(hallucinated confirmation)
```

素材は MASSIVE ja-JP (CC BY 4.0) のスロット注釈、音声化は既存の
Qwen3-TTS + MMS_FA 強制アライメント基盤を再利用。評価はモデル自身の
テキストストリームで完結し(出力ASR不要)、聞き返し時のみ修復発話を
フレーム同期注入する**閉ループドライバ**が新規要素。

## 実行手順(すべてPBS、V100/A100のみ)

```bash
# 0. 疎通(デモ4件 E2E; 最初に必ず)
qsub -V ICASSP2027/pbs/p0_smoke.pbs

# 1. ベンチマーク構築(+弱ASRオラクル)
qsub -v BENCH_ID=bench_v1 ICASSP2027/pbs/p1_build_benchmark.pbs

# 2. 閉ループ評価(4-way array; モデルごとに投入)
qsub -v BENCH_ID=bench_v1,MODEL_ID=base ICASSP2027/pbs/p2_eval_moshi.pbs
qsub -v BENCH_ID=bench_v1,MODEL_ID=jmoshi,HF_REPO=nu-dialogue/j-moshi-ext \
     ICASSP2027/pbs/p2_eval_moshi.pbs

# 3. FT学習データ 3 variant (task_only / clarify_lexical / clarify_full)
qsub -v VARIANT=clarify_full ICASSP2027/pbs/p3_build_train_data.pbs

# 4. LoRA FT (A100; 既存sweep基盤を再利用)
qsub -v SRC_RUN_DIR=ICASSP2027/runs/train_clarify_full \
     ICASSP2027/pbs/p4_train_clarify.pbs
#    → merge_lora.pbs でマージ後、p2 に MODEL_WEIGHT を渡して評価

# 5. カスケードベースライン(信頼度閾値スイープ)
qsub -v BENCH_ID=bench_v1 ICASSP2027/pbs/p5_eval_cascade.pbs

# 6. インタラクティビティ退行チェック(既存FDB-JAを再利用)
qsub -v MODEL_ID=clarify_full,MODEL_WEIGHT=/path/consolidated.safetensors \
     ICASSP2027/pbs/p6_fdb_regression.pbs

# 7. 集計・モデル比較(CPUのみ; ログインノード直実行も可)
qsub -v RUNS="base=ICASSP2027/runs/eval_base clarify_full=ICASSP2027/runs/eval_clarify_full" \
     ICASSP2027/pbs/p7_summarize.pbs
```

LLM judge(規則分類の独立検証)はローカルPCで
`ICASSP2027/runs/eval_*/judge_pack*.jsonl` を Azure に投げる
(クラスタから外部APIは呼ばない — リポジトリ方針)。

## 構成

```
clarify/                コアパッケージ(重依存はすべて遅延import)
  corruptions.py        スロット区間限定の音響劣化(決定的seed)
  slots.py              MASSIVE ja-JP スロット抽出・日本語正規化・修復発話
  scenario.py           ケーススキーマ + manifest I/O
  detector.py           聞き返し/復唱の規則分類 + ターン分割
  driver.py             閉ループ状態機械 + Moshiストリーミング駆動
  metrics.py            CRR/T-CRR/HCR/SSR/hit/FA/選択的リスク/対応ブートストラップ
  asr_oracle.py         弱ASR回復可能性オラクル(faster-whisper)
  cascade.py            ASR信頼度閾値カスケードベースライン
  train_data.py         FT対話生成(3 variant、最小ペア)
  judge_pack.py         LLM judge用パック + κ計算
scripts/                CLI(build_benchmark / run_closed_loop_eval /
                        run_cascade_eval / build_training_data /
                        corrupt_training_audio / summarize_results)
pbs/                    p0〜p7(上記手順)
tests/                  pytest 40件(CPUのみで実行可)
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
- 劣化はすべて `case_seed(base_id, condition)` 由来の固定seed →
  ベンチマークはコードから決定的に再構築可能。
- `corrupt_training_audio.py` は span を特定できない対話が5%を超えると
  失敗する(未劣化データで静かに学習が走る事故の防止)。
- p2 の `MODEL_WEIGHT` 指定時、`moshi_lm_kwargs.json` を重みの隣から
  自動検出(full-FT export 規約と同じ)。
