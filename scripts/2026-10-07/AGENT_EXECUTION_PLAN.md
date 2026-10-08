# 別Agentへの実行引き継ぎ：3条件を順次実行する方針

更新日：2026-10-08（日本時間）。実行対象のPBSはこの `2026-10-07` フォルダにあります。

目的はreal-v2・従来の重畳TTS・AI配置の3モデルを同じ対話集合で比較することです。**重い計算は同時に1ジョブだけ**にし、次のPBSは前段の成功を確認してから投入してください。この文書の追加作業ではサーバーへの投入を行っていません。引き継ぐAgentは実際のサーバーの状態から着手段階を判断します。

## 最初に確認すること

1. [README.md](README.md)を読み、実験設定と出力先を確認する。AI配置は従来方式と同じ両側Qwen TTS＋重畳を使い、バンク・KABURIを使わない。バンク・KABURIを使うのはreal-v2だけ。
2. サーバーのリポジトリ直下でブランチ・更新内容・ローカル変更を確認する。未コミット変更を破棄せず、既に実行中のジョブが参照するコードを途中で更新しない。
3. `qstat -u "$USER"` で既存ジョブを確認する。今回の実験の待機・実行中ジョブがあれば重複投入しない。同じユーザーの別の重い実験が実行中なら新規投入を待つ。他の実験のジョブを取り消さない。
4. 共有ストレージの空き容量、既存の共通対話、バンク、clone参照を確認する。既存の同一設定の生成物を優先して使う。バンクを流用する場合は語彙全語の音声とclone参照の一致を確認する。
5. `~/.miltoka/stop_fullft_chain` があれば停止の意図を確認し、そのまま着手を止める。自動で削除しない。

## 共通設定と負荷の抑え方

リポジトリ直下で設定し、すべての個別投入に同じ値を引き継ぎます。

```bash
export COMPARE_ID=aizuchi_compare_2026-10-08_continuous_10000
export COMPARE_N=10000
export COMPARE_SEED=0
export COMPARE_EVAL_SEEDS=0
export COMPARE_SHARDS=4
export COMPARE_NPROC=2

# LLM requests and CPU workers: reduce concurrency, keep model settings fixed.
export MULTI_AGENT_CONCURRENCY=8
export COMPARE_AI_CONCURRENCY=4
export NU_TOKENIZE_TEXT_WORKERS=2
export NU_TOKENIZE_AUDIO_WORKERS=2
export NU_DATASET_WORKERS=4
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

export MAX_LINKS=8
unset CHAIN_INDEX
export STOP_FILE="$HOME/.miltoka/stop_fullft_chain"
export PLAN_ROOT="$PWD/data/runs/$COMPARE_ID"
mkdir -p "$PLAN_ROOT/jobs"
```

既定のTTSは4GPU、学習は2GPUです。負荷を抑える主な方法は**条件間・段階間を並行実行しないこと**とCPU・LLMの並列数の制限です。GPU数、学習batch、音声生成条件を条件ごとに変更すると比較条件や再開設定が変わるので、勝手に減らしたり、途中で変えたりしません。

**`submit_comparison.sh` はこの順次実行では使いません。** このスクリプトは複数条件を並行投入します。全PBSを先にキューへ積むことも避け、未完了の段階は1つだけにします。ログインノードで `run_stage.sh`、TTS、学習、評価を直接実行せず、重い処理は個別PBSで実行します。

今回の共通対話は挨拶・終話なしの会話途中の抜粋です。話者発話数2〜12回と各発話の短・中・長を混ぜ、話者AIに残り回数を知らせません。旧v2対話は流用しません。新方針の生成済み対話だけは `COMPARE_SOURCE_INPUT` にその `shared/dialogues.jsonl` の絶対パスを設定して使えます。`listener_protocol` が一致することを確認します。バンク・clone参照を変更するなら、同じ参照のバンクを指定し、別の実験IDにします。旧データ・checkpointとは混在させません。

## 実行順序と完了の判定

表の上から1段ずつ実行します。すでに成功済みと検証できる段階は、実行記録に理由を書いてスキップできます。ファイルが存在するだけでは成功済みと判断しません。

| 順序 | 投入するPBS | 次へ進む前の確認 |
|---:|---|---|
| 1 | `prepare_dialogues.pbs` | `shared/dialogues.jsonl` が10,000件。IDが一意、密度ラベルと `continuous_listener_v1` 方針あり。挨拶・終話なし、長さの分布を確認。`source_config.json` と入力が一致 |
| 2 | `prepare_bank.pbs` | `check_bank.py` の検証が成功。既存バンクを使える場合は合成されず検証だけで終了 |
| 3 | `prepare_ai_placement.pbs` | AIの `dialogue/llm_dialogues/dialogues.jsonl` が10,000件。IDが共通対話と一致し、制約違反で中断していない |
| 4 | `render_real_v2.pbs` | real-v2の `tts/placement_bank/shard_000/training_set/synthetic_moshi_train.jsonl` が10,000件 |
| 5 | `condition_real_v2.pbs` | real-v2の `tts/placement_bank/shard_000_conditioned/training_set/` にmanifest。タグの欠落なし |
| 6 | `render_traditional_overlap.pbs` | 従来方式の `tts/merged/training_set/synthetic_moshi_train.jsonl` が10,000件 |
| 7 | `condition_traditional_overlap.pbs` | 従来方式の `tts/merged_conditioned/training_set/` にmanifest。全シャードでタグの欠落なし |
| 8 | `render_ai_placement.pbs` | AI方式の `tts/merged/training_set/synthetic_moshi_train.jsonl` が10,000件。バンク・KABURI経路を使っていない |
| 9 | `condition_ai_placement.pbs` | AI方式の `tts/merged_conditioned/training_set/` にmanifest。全シャードでタグの欠落なし |
| 10 | `assemble_paired_data.pbs` | `paired_summary.json` が成功。3条件の対話IDと順序が同じで、各 `paired/` にmanifest |
| 11 | `train_real_v2.pbs` | 自動再開チェーンと**自動投入されるreal-v2の評価**がすべて完了 |
| 12 | `train_traditional_overlap.pbs` | 自動再開チェーンと**自動投入される従来方式の評価**がすべて完了 |
| 13 | `train_ai_placement.pbs` | 自動再開チェーンと**自動投入されるAI方式の評価**がすべて完了 |
| 14 | `compare_results.pbs` | `comparison/results.md`、`results.csv`、`results.json` に3モデルの結果。試行ID・seedの一致検査が成功 |

件数は `COMPARE_N` を変えた場合、その値に読み替えます。対話IDの完全一致と各音声・sidecarの存在は、段階10の検査を必ず通します。

個別投入の例です。`stage` はその時点で実行可能な**1段階だけ**を指定します。この例を全段階のループに変更しないでください。

```bash
stage=prepare_dialogues
job_id=$(qsub -V -o "$PLAN_ROOT/jobs/${stage}_$(date +%Y%m%d_%H%M%S).log" \
  "scripts/2026-10-07/${stage}.pbs")
printf '%s\t%s\t%s\n' "$(date -Iseconds)" "$stage" "$job_id" \
  >> "$PLAN_ROOT/jobs/sequential_submissions.tsv"
qstat -f "$job_id"
```

監視は5〜10分程度の間隔を目安にし、秒単位で問い合わせ続けません。ジョブが一覧から消えただけでは次へ進みません。取得できる終了ステータス、保存したPBSログのエラー・最終メッセージ、表の生成物を確認します。状態を取得できない場合は不明として記録し、成功と推測して次を投入しません。

## 学習チェーンで間違えないこと

学習PBSは時間切れ前に次の学習PBSを自動投入し、正常に学習を終えると、その条件の評価PBSを自動投入します。**最初の学習ジョブの終了は、その条件の完了ではありません。**

- `jobs/<条件>_train_next.txt` と `jobs/<条件>_eval.txt` から後続IDを追い、待機中・実行中の後続がある間は同じ学習・評価を手動投入しない。
- 条件の完了には、最良checkpointの選択、exportされた `model.safetensors` と `moshi_lm_kwargs.json`、評価の成功が必要。`jobs/<条件>_model.txt` にexport先がある。
- 評価結果は `eval_runs/full_duplex/<実験ID>_<条件>/benchmark_results/summary.json`。`evaluation.status` が `complete` で、`per_case.jsonl` が揃っていることを確認する。`partial` を成功扱いしない。
- real-v2の評価が完了してから従来方式の学習を始め、従来方式の評価が完了してからAI方式の学習を始める。学習中に別条件の音声化や評価を並走させない。
- 評価は最終学習ジョブが自動投入するので、通常は `eval_<条件>.pbs` を別途投入しない。評価だけ失敗した場合に限り、実行中・待機中の評価がないことを確認して、その評価PBSだけを再投入する。

## 失敗・中断・引き継ぎ

失敗したら、その段階で止めてログと生成物を確認します。OOMやストレージ不足、同じエラーを自動で何度も再投入しません。学習チェーンの「checkpointなし」「進捗なし」「MAX_LINKS到達」も、理由を確認するまで止めます。

対話生成・AI配置・音声化は既存の途中再開を使います。入力・語彙・シャード数・clone参照・音声設定を変えて、同じ出力先で再開しません。学習は同じ実験ID・checkpointディレクトリから再開し、最新ではなく書き込みが落ち着いた使用可能なcheckpointを選ぶ既存経路を使います。

条件タグの付与は再実行不可です。途中失敗したconditioned出力に再度パディングを足さないでください。現在の条件付与PBSは、完成済みシャードだけを自動でスキップする実装ではありません。PBS全体を再投入する場合は、real-v2なら `tts/placement_bank/shard_000_conditioned/`、従来・AI方式なら `tts/conditioned/` と既存の `tts/merged_conditioned/` を、その条件のディレクトリ内の日時付き退避先へ保存してから実行します。元のTTS音声は保持し、TTSの再合成は行いません。退避による容量不足がないことも確認します。既存の音声・checkpointを削除してやり直すことを最初の対処にしません。

チェーンの後続投入を止める場合は `STOP_FILE` を使います。これは次の学習リンクの投入を止める仕組みで、実行中のジョブを即時終了させるものではありません。緊急に負荷を止める必要がある場合は、今回の実験の該当ジョブIDを確認し、そのジョブだけを対象にします。

実行時に `$PLAN_ROOT/jobs/AGENT_PROGRESS.md` を作り、別Agentに引き継げるように更新してください。更新する内容は次のとおりです。

- 実験ID、件数、コードのcommit、共通対話・バンク・clone参照のパスと設定。
- 完了した段階と検査結果、現在の段階とPBSジョブID、自動投入された後続ID、ログのパス。
- 学習の場合は条件名、最後に確認したcheckpoint step、export・評価の状態。
- 失敗や停止の理由、未完了の生成物、次に行う1つの作業、最終確認日時（日本時間）。

Agent交代時はこの記録とサーバー上のジョブ状態を照合します。記録にジョブIDがないことを理由に二重投入せず、すでに待機・実行しているジョブを先に探します。最終的に3モデルのexport先と比較表を報告し、未実行・失敗・途中のものを完了と記載しません。
