# ねじ問い合わせ「疑似先輩」PoC

工場の若手作業者が、ねじの見分け方・ピッチ・締付トルク・下穴・トラブル対応を音声で質問できる Moshi 用PoCです。既存miltokaの音声合成・学習処理を再利用しますが、知識、対話、生成物、実験結果はすべて `screw_poc/` 配下に分離します。

## 設計

- 技術知識は [`knowledge/screw_knowledge.csv`](knowledge/screw_knowledge.csv) の30項目で一元管理します。
- システム発話は、知識表の承認済み回答、固定の確認質問、固定の停止・エスカレーション文だけから生成します。
- ユーザー発話だけを言い換え、技術値の揺れや創作を防ぎます。
- TTSでは規格名・品番・小数・単位を固定の読みへ置き換えます。表示・採点用の原文は維持し、音声だけを正確に読み上げます。
- 情報がないときは、専門用語を繰り返し要求せず質問を3段階で簡単にします。
  - 例: 「強度区分は？」→「頭の数字は？」→「頭に見える文字や数字をそのまま読んでください」
- 3回確認しても特定できない場合は、推測で答えず作業停止または責任者確認へ誘導します。
- システム音声は `Ono_Anna` 固定、ユーザーはそれ以外の8音声を循環利用します。

これは方式検討用です。一般規格だけから任意の製品の締付トルクを断定しません。CSVにある具体的なトルク値は、出典メーカー・品番・条件に限定した回答です。実運用時は、工場の作業標準・図面・メーカー仕様を最優先データとして差し替えてください。

## フォルダ

```text
screw_poc/
  config/       対話ポリシー、音声、LoRA設定
  knowledge/    30項目の承認済み知識DB
  scripts/      生成、検証、TTS、学習ラッパー
  tests/        知識ロックと状態遷移のテスト
  artifacts/    生成データ（git管理外）
  experiments/  学習時に作られるチェックポイント（git管理外）
```

## 1. 対話データを作る

リポジトリ直下で実行します。

```bash
python screw_poc/scripts/generate_dialogues.py
python screw_poc/scripts/validate_dataset.py screw_poc/artifacts/train_dialogues.jsonl --expected-count 1000
python screw_poc/scripts/validate_dataset.py screw_poc/artifacts/evaluation_dialogues.jsonl --expected-count 200
python screw_poc/scripts/build_training_subsets.py
```

主な出力は次のとおりです。

- `artifacts/train_dialogues.jsonl`: 学習用1,000対話
- `artifacts/evaluation_dialogues.jsonl`: 未使用表現を含む評価用200対話
- `artifacts/subsets/train_0100.jsonl`: 30知識をすべて含む100対話
- `artifacts/subsets/train_0300.jsonl`: 30知識をすべて含む300対話
- `artifacts/subsets/train_1000.jsonl`: 基準となる1,000対話

100/300/1000件は同じ正解文を共有するため、件数による効果を比較できます。標準の1,000件は、知識内900件、領域外100件からなる通常の逐次対話です。話者の発話は重ねません。

## 2. Qwen3-TTSで音声学習データを作る

まず少数で確認する場合:

```bash
NUM_DIALOGUES=10 bash screw_poc/scripts/run_tts.sh
```

全件を作る場合:

```bash
bash screw_poc/scripts/run_tts.sh
```

PowerShellでは次を使えます。

```powershell
.\screw_poc\scripts\run_tts.ps1 -NumDialogues 10
```

既存の `scripts/generate_qwen3_tts_data.py` を呼び出し、左チャンネルをシステム、右チャンネルをユーザーとする Moshi 学習用WAVとmanifestを `screw_poc/artifacts/tts/` に生成します。

## 3. Moshi LoRAを学習する

TTS生成後に実行します。

```bash
bash screw_poc/scripts/run_train.sh
```

既存の安全策を含む学習ランチャーを再利用しつつ、設定、データ、ログ、チェックポイントは `screw_poc/experiments/lora_base_config/` に置かれます。100件・300件を試す場合は、各サブセットを `run_tts.sh` の第1引数に渡し、TTS出力先を第2引数で分けてください。

## PBSで実行する

PBSサーバーでは、リポジトリ直下から投入します。まず3対話の音声確認を行います。

```bash
qsub -V screw_poc/pbs/run_tts_smoke.pbs
```

問題がなければ、4 GPUで1,000対話を生成します。

```bash
qsub -V screw_poc/pbs/run_tts_1000_4gpu.pbs
```

TTS終了後、Moshi LoRAとFull Fine-tuningをそれぞれ投入できます。

```bash
qsub -V screw_poc/pbs/run_train.pbs
qsub -V screw_poc/pbs/run_train_full.pbs
```

TTS成功後にLoRAとFull FTの両方を自動開始する依存関係付き投入は、次の1コマンドです。

```bash
bash screw_poc/pbs/submit_pipeline.sh
```

この場合、`afterok` 依存を使うため、TTSが失敗した場合に学習は始まりません。LoRAとFull FTは同じ音声データを読み、互いに独立して実行されます。

すでにTTSが完成している場合は、次のコマンドで両方の学習だけを投入できます。

```bash
bash screw_poc/pbs/submit_train_both.sh
```

出力先は次のように分かれます。

- TTS: `screw_poc/artifacts/tts_1000_sequential/`
- LoRA: `screw_poc/experiments/lora_base_config/`
- Full FT: `screw_poc/experiments/fullft_base_config/`
- PBSログ: `screw_poc/artifacts/pbs_logs/`
- Full FT前処理データ: `screw_poc/artifacts/nu_fullft/`

Full FTは既存のnu-dialogue経路を使い、既定でA100 2 GPU、学習率`1e-5`、12 epochです。LoRAより計算時間とチェックポイント容量が大きくなります。

100・300対話の比較実験では、たとえば300対話を次のように投入できます。

```bash
qsub -V -v DIALOGUES_JSONL=screw_poc/artifacts/subsets/train_0300.jsonl,NUM_DIALOGUES=300,OUT_ROOT=screw_poc/artifacts/tts_0300 screw_poc/pbs/run_tts_1000_4gpu.pbs
qsub -V -v TTS_RUN_DIR=screw_poc/artifacts/tts_0300/merged screw_poc/pbs/run_train.pbs
qsub -V -v TTS_RUN_DIR=screw_poc/artifacts/tts_0300/merged screw_poc/pbs/run_train_full.pbs
```

## 4. テスト

```bash
uv run pytest screw_poc/tests/test_screw_poc.py -q
```

テストでは、30知識の存在、音声割当、1,000/200件の件数、技術回答のDB固定、3段階質問、100/300/1000件の知識網羅、逐次対話であることを確認します。

## 5. 正解率を測る

モデル音声を既存のASR処理で文字起こしし、対話IDとシステム発話をJSONLにします。

```json
{"id":"evaluation_k01_000","response_texts":["呼び径は分かりますか？","M3の並目ピッチは0.5ミリです。"]}
```

最終回答だけを採点する場合は `response_text` を使えます。採点コマンドは次のとおりです。

```bash
python screw_poc/scripts/score_predictions.py predictions.jsonl
```

評価値として、評価200件のカバレッジ、最終回答の正規化完全一致率、対話内の全システム発話系列一致率、最終回答の文字誤り率を出します。応答パターン分類も付与した場合は、その系列一致率も同時に測れます。
