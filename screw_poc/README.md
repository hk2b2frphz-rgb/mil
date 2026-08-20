# ねじ問い合わせ「疑似先輩」PoC

工場の若手作業者が、ねじの見分け方・ピッチ・締付トルク・下穴・トラブル対応を音声で質問できる Moshi 用PoCです。既存miltokaの音声合成・Full-Duplex・学習処理を再利用しますが、知識、対話、生成物、実験結果はすべて `screw_poc/` 配下に分離します。

## 設計

- 技術知識は [`knowledge/screw_knowledge.csv`](knowledge/screw_knowledge.csv) の30項目で一元管理します。
- システム発話は、知識表の承認済み回答、固定の確認質問、固定の停止・エスカレーション文だけから生成します。
- ユーザー発話だけを言い換え、技術値の揺れや創作を防ぎます。
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

100/300/1000件は同じ正解文を共有するため、件数による効果を比較できます。標準の1,000件は、知識内900件、領域外100件、Full-Duplex事象300件です。

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

TTSから学習まで、投入するのはこの1本だけです。

```bash
qsub -v PROXY_URL=http://<host>:<port> screw_poc/pbs/run_tts_1000_4gpu.pbs
```

`PROXY_URL` は必須です（HFキャッシュが既にある場合を除く）。指定しないと
`setup_proxy.sh` がプロキシを無効化し、4シャードとも重み取得で落ちます。認証ありなら
`http://user:pass@host:port`、`PROXY_HOST`/`PROXY_PORT`/`PROXY_USER`/`PROXY_PASS`
の形でも構いません。この値はTTSジョブが連鎖させるLoRA/Full FTへも `-v` で
引き継がれます。

シェルに `PROXY_URL` などを export 済みなら、ラッパー経由でも同じことができます
（設定済みの `PROXY_*` を集めて `-v` に載せます）。

```bash
export PROXY_URL=http://<host>:<port>
bash screw_poc/pbs/submit_pipeline.sh
```

投入されるのはTTSジョブだけです。LoRAとFull FTは、**TTSジョブが自分の最後で
`qsub` して投入します**（`run_tts_1000_4gpu.pbs` の `chain: submit training`
ステップ）。マージ済みmanifestの検証を通った時点で初めて投入されるため、TTSが
失敗したときは学習ジョブがそもそもキューに入りません。以前の
`qsub -W depend=afterok` 方式では、TTS失敗時に2本がholdのままキューに残り、
手で `qdel` する必要がありました。

連鎖したジョブIDは、TTSログ末尾の `[chain]` 行に出ます。

```bash
tail -20 screw_poc/artifacts/pbs_logs/tts_1000_<jobid>.log
```

環境変数は小文字の `-v` で渡します。`-V` は投入元のプロキシ設定とPATHを子ジョブへ
運び、`-v` がジョブごとに変わる値（`TTS_RUN_DIR`、`NPROC`）を上書きします。これに
より、4 GPUのTTSジョブの `CUDA_VISIBLE_DEVICES` が1 GPUのLoRAジョブへ漏れません。
各ジョブはノードに入った直後に `scripts/setup_proxy.sh` を無条件で読み込むので、
連鎖先でもプロキシは毎回張り直されます。

### プロキシ

計算ノードから huggingface.co に出られないと、Qwen3-TTSの重み取得に失敗して
4シャードとも落ちます（`MaxRetryError HTTPSConnectionPool(host='huggingface.co',
port=443)`）。`scripts/setup_proxy.sh` は `PROXY_URL`（または
`PROXY_HOST`/`PROXY_PORT`/`PROXY_USER`/`PROXY_PASS`/`PROXY_SCHEME`）が
無いとプロキシを無効化して全変数をunsetするため、**投入シェルで設定しておく必要が
あります**。

`submit_pipeline.sh` と `submit_train_both.sh` は、設定されている `PROXY_*` を
`screw_poc/pbs/proxy_env.sh` 経由で集めて小文字 `-v` に載せます。TTSジョブは
受け取った値をそのまま連鎖先のLoRA/Full FTへ `-v` で引き継ぐので、**チェーンの
どのジョブでもプロキシが張られます**。ジョブ側の実際の状態はログで確認できます。

```bash
grep -n "\[proxy\]" screw_poc/artifacts/pbs_logs/tts_1000_*.log
# [proxy] enabled: proxy.example.ac.jp:8080   <- 正常
# [proxy] disabled                            <- 落ちる
```

未設定のまま投入しようとすると、投入時点で警告が出ます。`NO_PROXY` は値にカンマを
含み `qsub -v` で運べないため転送対象から外しており、ジョブ内で
`setup_proxy.sh` が再構成します。パスワードを含む値はログにも端末にも出しません
（変数名だけを表示します）。

TTSジョブの step 2/5 は、シャードを起動する前にチェックポイントを1回だけ取得します。
到達できない場合はこの時点で1つの明確なエラーで停止し、4シャードが個別にリトライして
死ぬことはありません。取得済みなら即通過し、4シャードは温まったキャッシュから読みます。

連鎖を止めてTTSだけ流したい場合は `CHAIN_TRAIN=0` を渡します。

```bash
qsub -V -v CHAIN_TRAIN=1 screw_poc/pbs/run_tts_1000_4gpu.pbs   # 既定（連鎖する）
qsub -V -v CHAIN_TRAIN=0 screw_poc/pbs/run_tts_1000_4gpu.pbs   # TTSのみ
```

すでにTTSが完成している場合は、次のコマンドで両方の学習だけを投入できます。

```bash
bash screw_poc/pbs/submit_train_both.sh
```

出力先は次のように分かれます。

- TTS: `screw_poc/artifacts/tts_1000/`
- LoRA: `screw_poc/experiments/lora_base_config/`
- Full FT: `screw_poc/experiments/fullft_base_config/`
- PBSログ: `screw_poc/artifacts/pbs_logs/`
- Full FT前処理データ: `screw_poc/artifacts/nu_fullft/`

Full FTは既存のnu-dialogue経路を使い、既定でA100 2 GPU、学習率`1e-5`、12 epochです。LoRAより計算時間とチェックポイント容量が大きくなります。

100・300対話の比較実験では、たとえば300対話を次のように投入できます。

```bash
DIALOGUES_JSONL=screw_poc/artifacts/subsets/train_0300.jsonl \
NUM_DIALOGUES=300 \
OUT_ROOT=screw_poc/artifacts/tts_0300 \
    bash screw_poc/pbs/submit_pipeline.sh
```

これらの変数は `-v` でTTSジョブへ転送され、連鎖側は対応する
`TTS_RUN_DIR=screw_poc/artifacts/tts_0300/merged` を自動で引き継ぎます。手で
3本投入する必要はありません。

## 4. テスト

```bash
uv run pytest screw_poc/tests/test_screw_poc.py -q
```

テストでは、30知識の存在、音声割当、1,000/200件の件数、技術回答のDB固定、3段階質問、100/300/1000件の知識網羅、既存Full-Duplex検証との互換性を確認します。

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
