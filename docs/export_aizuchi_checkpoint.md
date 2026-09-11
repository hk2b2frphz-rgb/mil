# 学習途中の相槌モデルをexportする

Full-FT（nu-dialogue / DeepSpeed ZeRO）の保存済み `step_<N>` を指定します。
リポジトリのルートで実行してください。LoRA用ではありません。

```bash
qsub -v STEP_DIR=/absolute/path/checkpoints/nu_timestamp/step_120 \
  scripts/export_aizuchi_checkpoint.pbs
```

出力先も指定する場合（既存のディレクトリは上書きしません）：

```bash
qsub -v STEP_DIR=/absolute/path/checkpoints/nu_timestamp/step_120,OUT_DIR=/absolute/path/exports/aizuchi_step120 \
  scripts/export_aizuchi_checkpoint.pbs
```

標準の保存先は以下です。`<JOB_ID>` はPBSジョブ番号です。

```text
exported_model/aizuchi_export_step_120_<JOB_ID>/
  checkpoint/step_120/     # 元チェックポイントの独立したコピー
  exported/model.safetensors
```

`SNAPSHOT_DIR` でコピー先も指定できます。コピー先・export先は学習の
`checkpoints` ツリーの外側に置いてください。元チェックポイントや `latest`、
学習設定には書き込みません。変換処理はコピーだけを使用します。
学習と共有するPython環境の依存パッケージ更新も無効にしています。
事前に通常のFull-FT/export環境（このリポジトリとnu-dialogue側）を構築しておいてください。

保存完了済みのstepを選んでください。標準で30秒間のファイル一覧・サイズ・更新情報の
安定性と、コピー前後の変化を確認し、変化があればexportを中止します。
これは保存完了の厳密な証明ではないため、書き込み中の最新stepは避けてください。
`STABLE_SECONDS` で確認間隔を変更できます。学習側の古いcheckpoint削除と重なった場合も
失敗することがあります。その場合は別の保存済みstepと新しい出力先で再実行してください。

コピーは成功時・失敗時とも残します。変換中はチェックポイント一式、FP32中間モデル、
exportモデルを保持できる空き容量が必要です。コピーによる共有ストレージの読み取り負荷と、
別PBSジョブ分の計算資源は発生します。

変換設定は既存の `scripts/export_fullft_checkpoint.pbs` と共通です。
`NU_MOSHI_FT_REPO`、`MOSHI_LM_KWARGS`、`MODEL_DTYPE`（標準 `float16`）、
`REMOVE_USER_STREAM`、`KEEP_INTERMEDIATE` を必要に応じて `-v` で指定できます。
学習時に独自のモデル設定を使った場合は、対応する `MOSHI_LM_KWARGS` を明示してください。
exportのログは `experiments/pbs_logs/aizuchi_export_step_120_<JOB_ID>.log`、
コピーを含むジョブ全体のログはPBSの標準出力に保存されます。
