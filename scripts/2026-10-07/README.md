# real-v2との3条件の比較（2026-10-07作成、2026-10-08更新）

| 条件 / 出力モデル | 相槌の位置・個数 | 音声の作り方 |
|---|---|---|
| `real_v2` | 現行v2：句の切れ目を密度条件の確率で選ぶ | 現行の相槌バンク＋KABURI予測配置 |
| `traditional_overlap` | `real_v2`と同じ相槌テキスト・位置・個数 | 従来のwhole-utterance Qwen TTSで両者を合成し、句末付近で重畳。バンク・KABURIを使わない |
| `ai_placement` | 相槌AIが意味を見て位置・語・個数を選ぶ | `traditional_overlap`と同じ両側Qwen TTS＋重畳。バンク・KABURIを使わない |

`real_v2`→`traditional_overlap`では音声生成・重畳処理を、`traditional_overlap`→`ai_placement`では同じ音声生成方式のまま相槌の配置・個数選択を比較します。AI条件は個数と語も変わるため、位置だけの効果を測る実験ではありません。`real_v2`とAI条件の直接比較では、音声生成方式も変わります。

話者の発話、沈黙ターン、聞いているかの確認とその返答、対話ID、対話ごとの密度条件は共通です。AIの反応を受けて話者AIに続きを生成させることはせず、凍結した同じ対話に相槌を配置し直します。挨拶は全条件で外します。音声化の手法が異なるため、波形・実現した沈黙や対話の長さは一致しません。

## 会話を終わらせない聞き手

3条件とも、挨拶・名乗り・「おやすみなさい」などの終話を学習させません。話者AIにも総発話数・残り回数を知らせず、最後の発話でお礼や別れを言わせる旧指示を廃止しました。会話を続けているところで収録を切ります。直接の挨拶・終話が生成された場合は再要求し、改善しなければ失敗として扱います。語の引用や出来事としての言及は許容します。

収録ごとの話者発話数は2〜12回で無作為に選び、各発話も短文1文・中程度2〜3文・長め4〜6文の指示を混ぜます。音声を一定秒数に揃える設定ではありません。沈黙中は待ち、話が再開したら相槌を返す教師データを使います。密度0の条件と重複・連発の抑制は維持します。

`listener_protocol` に収録方針と長さの範囲を保存し、生成の再開では方針の一致を検査します。旧データや異なる方針を混ぜた入力の凍結は拒否します。学習の170秒はコンテキストの上限であり、会話を170秒で終了させる指示ではありません。長時間推論で反応が弱くならないかは、再学習後に別途確認が必要です。

## 相槌AIの設計

- 基本は一文を聞いて、句の切れ目を意味で選択します。短い文は隣接文とまとめ、文ごとに反応が連発することを抑えます。
- 密度0は相槌なし。それ以外では `ceil(密度×3)` 個まで（1〜3個、句数を超えない）。上限まで埋める義務はありません。単位の文末には1個置きます。
- 同じ位置への複数挿入、直近2回と同じ語、相槌間に24文字分未満しか話を挟まない配置を拒否します。沈黙や確認への応答の境界は別扱いです。
- 確認の質問には新たな相槌を足さず、元の返答を保持します。
- 語彙は現行v2と同一。制約違反は最大3回再要求し、それでも違反する場合は失敗します。ランダム配置やテンプレートには置き換えません。
- 文の内容は丸ごと見せる教師データ生成です。未来の別文・別ターンは提示しません。実運用の音声ストリーミングそのものを再現した推論評価ではなく、文字数による間隔制約も秒単位の間隔を保証しません。

AI側の全回答・制約違反と再要求は `ai_placement/dialogue/llm_dialogues/dialogues.trace.jsonl` に保存します。途中で止まっても成功済み対話は保持し、同じPBSの再投入で再開します。入力・語彙・生成設定のハッシュが変わる場合は再開を拒否します。

## 一括投入

別Agentが負荷を抑えて1段階ずつ実行する場合は、[順次実行の引き継ぎ方針](AGENT_EXECUTION_PLAN.md)を優先してください。以下の一括投入は条件間の並列実行を許す構成です。

サーバーのリポジトリ直下で実行します（この変更ではサーバーへの投入は行っていません）。

```bash
bash scripts/2026-10-07/submit_comparison.sh
```

既定は10,000対話です。2026-10-08更新後の既定実験IDは `aizuchi_compare_2026-10-08_continuous_10000` です。挨拶・終話のある旧対話、旧AIのKABURI音声、学習済みデータ、checkpointと混在しないよう、新しいIDで実行してください。まず小さく動作確認する場合：

```bash
COMPARE_N=20 COMPARE_ID=aizuchi_compare_2026-10-08_continuous_smoke \
  bash scripts/2026-10-07/submit_comparison.sh
```

旧v2対話は自動流用せず、継続する聞き手の新しい方針で共通対話を生成します。途中で止まった場合は同じ方針の生成済み分から再開します。新方針で生成済みの共通対話だけは明示的に流用できます：

```bash
COMPARE_N=20 COMPARE_ID=aizuchi_compare_2026-10-08_continuous_smoke \
COMPARE_SOURCE_INPUT="$PWD/data/runs/aizuchi_compare_2026-10-08_continuous_10000/shared/dialogues.jsonl" \
  bash scripts/2026-10-07/submit_comparison.sh
```

バンクの既定は `data/runs/diversity/real_aizuchi_bank_v2` です。なければ作成し、語彙全語の音声があることを確認します。同じボイス参照を使うため、3条件の `CLONE_OUT_DIR_MOSHI` を `data/clone_examples/99999`、`REF_RANK=1` に固定します。変更する場合は **別の実験ID** にし、同じ参照から作ったバンクも指定してください。

バンク照合は完全一致を優先し、書き起こし側に句読点がない場合は両側を同じように正規化して照合します。従来の片側だけの正規化によって、同じ語の音声が存在するのに別の語へフォールバックする問題も修正しました。既に作成済みの音声は、この修正だけでは書き換わりません。

```bash
COMPARE_ID=aizuchi_compare_2026-10-08_continuous_other_voice \
COMPARE_CLONE_DIR="$PWD/data/clone_examples/your_reference" \
COMPARE_BANK="$PWD/data/runs/diversity/your_matching_bank" \
  bash scripts/2026-10-07/submit_comparison.sh
```

主な共通指定は `COMPARE_N`、`COMPARE_ID`、`COMPARE_SOURCE_INPUT`、`COMPARE_BANK`、`COMPARE_CLONE_DIR`、`COMPARE_SEED`（既定0）、`COMPARE_EVAL_SEEDS`（既定0）です。TTSは4シャード、学習は2GPUが既定です。PBSのリソース指定を変更せず、これより大きいGPU数を指定しないでください。

## ジョブの構成

| PBS | 入力・役割 |
|---|---|
| `prepare_dialogues.pbs` | 挨拶・終話なしで長さを混ぜた共通対話を生成・凍結。新方針の入力だけ流用可 |
| `prepare_bank.pbs` | 共有v2バンクを検証、なければ生成 |
| `prepare_ai_placement.pbs` | 共通対話にAIが相槌を配置 |
| `render_real_v2.pbs` | 現行のバンク＋KABURIで音声化 |
| `render_traditional_overlap.pbs` | 従来の両側TTS＋重畳で音声化 |
| `render_ai_placement.pbs` | AI配置対話を従来方式と同じ両側TTS＋重畳で音声化 |
| `condition_<条件>.pbs`（3本） | 元の対話密度を無音のテキストタグとして付与 |
| `assemble_paired_data.pbs` | 3条件の対話IDを検査し、同じ並びの学習用manifestを作成 |
| `train_<条件>.pbs`（3本） | 共通f01相当のfull-FT。時間切れ時は同じPBSを自動再投入 |
| `eval_<条件>.pbs`（3本） | 最良checkpointのモデルを同じFull-Duplex-Bench-JAで評価 |
| `compare_results.pbs` | 3モデルを列としてCSV・Markdown・JSONにまとめる |

投入スクリプトは準備→音声化→条件タグ→対データ検査→学習を `afterok` の依存関係で繋ぎます。バンク準備に依存する音声化はreal-v2だけです。従来方式とAI方式はバンク・KABURIの準備を待たずに音声化でき、AI方式はAI配置準備にも依存します。3条件を揃える学習前の検査は全条件の完成を待ちます。

TTSの失敗を別の対話で埋めるspareは無効です。欠落IDや条件タグを付けられなかった対話があると、学習前の検査が失敗します。3条件で異なる成功例だけを使って学習が始まることはありません。3つのmanifestの順番と学習データ分割のseedを揃えます。ただし音声長による訓練チャンク数・具体的なstep数は条件で異なり得ます。

全モデルは同じ初期Moshi、学習率7e-6、weight decay0.1、batch1×2GPU×勾配蓄積8、最大12epoch、warmup1epoch、評価0.5epochごと、early stopping4回を使用します。コンテキストは170秒。最良評価損失ごとのcheckpoint保存と上位3個＋最新の保持は、直前に修正した経路を使います。

学習チェーンはwalltimeの1時間前に止まり、完全な学習状態から再開します。最大8ジョブ。`~/.miltoka/stop_fullft_chain` があれば後続を投入しません。正常終了時はearly stoppingも完了扱いとして最良checkpointをexportします。評価は最終学習ジョブが別PBSとして投入するので、最初の学習ジョブだけに依存して早く評価を始めることはありません。

## 出力と再実行

既定の出力は `data/runs/aizuchi_compare_2026-10-08_continuous_10000/` です。従来方式・AI方式の条件タグ付きデータは、各条件の `tts/merged_conditioned/training_set/` にまとまります。

- `shared/dialogues.jsonl`：凍結した共通対話。
- `<条件>/tts/`：各音声データ。`<条件>/paired/`：学習に使用するmanifest。
- `paired_summary.json`：3条件の対話数・音声時間。
- `jobs/submissions.tsv`：投入したジョブID。`jobs/<条件>_model.txt`：export済みモデルのパス。
- `comparison/results.md`、`results.csv`、`results.json`：3モデルの比較。

モデル本体は `experiments/_fullft_sweeps/<実験ID>_<条件>_f01/exported/best/model.safetensors` に並びます。

音声化は途中再開に対応します。条件タグの付与は既存と同じく再実行不可です。途中で止まった出力に再度パディングを足さないため、条件付与が失敗した際はその条件のconditioned出力を退避し、当該PBSを再投入してください。一括投入スクリプト全体を再度投入すると学習ジョブが重複し得るため、再開には個別PBSを使います。

```bash
qsub -V -v COMPARE_ID=aizuchi_compare_2026-10-08_continuous_10000,COMPARE_N=10000 \
  scripts/2026-10-07/train_ai_placement.pbs

qsub -V scripts/2026-10-07/compare_results.pbs
```

比較表は全モデルの評価が完了して同じ試行ID・seedが揃った場合だけ作ります。評価の部分失敗をゼロ値で埋めません。UTMOS等の既存音響評価、相槌数・頻度・JSD、内部書き起こしの語彙一致と同語反復を並べます。教師データの統計はモデル出力の評価と区別して載せます。独立ASRによる音声と内部テキストの一致検査やLLMによる意味の採点は、今回の自動ジョブには含めていません。

AI条件と従来方式は新規モデルとして学習します。既存の学習済みv2モデルをそのまま置くこともできますが、データ集合・学習設定を統一した比較には、この3条件一式で再学習する方式を使ってください。
