# 実対話リプレイ評価

実際のロールプレイ相談音声を入力に使い、そこで相談員が実際に返した応答を参照と
して Moshi を評価する。合成の [`full_duplex_evaluation.md`](full_duplex_evaluation.md)
とは**別トラック**で、スコアを同じ表に並べてはいけない（理由は後述）。

## 全体の流れ

```
1ch WAV
  └─ ステージ1  scripts/analyze_real_dialogue.py     ダイアライゼーション + 書き起こし
  └─ ステージ2  scripts/separate_real_dialogue.py    音源分離 + 分離トラックで再書き起こし
  └─ 人手       counselor_X.txt を置く / review TSV で間引く
  └─ 構築       eval/build_real_dialogue_dataset.py  FDB 形状のデータセット
  └─ 推論・評価 scripts/run_real_dialogue_eval.sh    推論 → 決定的評価 → 相槌一致度 → judge 入力
  └─ ローカル   eval/judge_openai.py                 LLM-as-a-judge（ローカル PC のみ）
```

## なぜ音源分離が要るか

1ch 録音では重畳区間に両者の声が同じ標本へ混ざっている。ステージ1 の
`speaker_A_solo.wav` は「その話者の区間以外を無音化」しただけなので、重畳区間の
相手の声は消えない。

これが致命的なのは、その混入音が**まさに相談員の相槌**だからである。モデルへの
入力に相談員の「ええ」「はい」が残っていると、相槌の打ち方を評価したいのに、
入力に正解の相槌が入っている状態になる。

ステージ2 は全区間を分離し、話者ごとの独立トラックを作る。分離器が返すチャネル
順はチャンクごとに不定（permutation problem）なので、ステージ1 のダイアライ
ゼーション結果を教師にして毎チャンク並べ直す。信頼度（2 通りの並びのスコア差）
が低いチャンクは直前の割り当てを引き継ぎ、`separation_report.json` に
`inherited_assignment` として記録する。

分離器の既定は `speechbrain/sepformer-whamr16k`（Apache-2.0、商用可）。日本語で
学習されたモデルではないが、話者分離は音響的手がかりが主で言語依存が小さい。
**採用前に `segments_separated/` を耳で確認すること。**

## 人手で行う 2 つの作業

### 1. 相談員側の指定

各対話ディレクトリ直下に、相談員側の話者ラベル名のマーカーファイルを置く。

```
data/real_dialogue/pilot/対話1/counselor_B.txt   ← 中身は空でよい
```

中身は読まない。存在することだけが意味を持つ。ユーザー側は残りの話者として
自動決定される。

マーカーが 0 個でも 2 個でもビルダーは停止する。取り違えると入力側と参照側が
丸ごと入れ替わり、**指標は普通に出るのに結果は全部無意味になる**ため、静かに
通してはいけない。

### 2. テストデータの間引き

ステージ2 が `<話者>_segments_review.tsv` を出す（UTF-8 BOM 付き TSV、Excel で
そのまま開ける）。

| 列 | 意味 |
|---|---|
| `keep` | `1` で採用、`0` で除外。ここだけ編集する |
| `index` | `segments_separated/` の WAV ファイル名の通し番号と一致 |
| `start_sec` / `end_sec` / `duration_sec` | 位置と長さ |
| `overlap_sec` | 相手と重なっていた秒数 |
| `is_aizuchi` / `aizuchi_labels` | 相槌判定と種類 |
| `text` | 分離トラックでの書き起こし |
| `wav` | 対応する切り出し WAV |

ユーザー側の `keep=0` にした行は、評価点の候補から外れる。

## 開ループであること（最重要の制約）

入力に入るのはユーザー側の実音声だけで、相談員側はモデルが置き換わる。実際の
相談員が何を言ったかはモデルに伝わらないので、文脈区間でモデルは**自分で相談員役
を喋りながら**進む。評価点に着く頃には、モデルの会話履歴は実際の対話と別物に
なっている。

帰結は 2 つ。

- **ユーザー側ストリームだけで定義できる指標は妥当**。TOR、応答レイテンシ、間の
  取り方、相槌頻度はユーザーの発話・沈黙に対する反応なので、そのまま読める。
- **重畳を前提とする指標は条件付き**。`user_backchannel` / `user_interruption` は
  「その瞬間モデルが実際に喋っていたか」に依存する。評価器が既に
  `overlap_achieved` を出しているので、**それが真の trial だけで集計する**こと。

また、`manifest.protocol.name` は v1.5 を名乗らない
（`Real dialogue replay (open-loop), FDB-JA metric definitions`）。
`run_full_duplex_bench.py` に `--require-v15-profile` を渡さず、
`evaluate_full_duplex_ja.py` に `--require-overlap` も渡さない運用にしてある。
合成 FDB-JA のスコアと構造的に混ざらないようにするため。

## 相槌一致度

`eval/evaluate_real_backchannel.py`。正解は「その場面で実際の相談員が打った
相槌」（時刻と種類つき）で、ビルダーが `metadata.backchannel_gt` に埋め込む。
予測側は評価器が `per_case.jsonl` に書く `speech_segments`（Silero VAD）と
`assistant_chunks`（時刻つきテキスト）から取る。

指標は 4 層。上ほど緩く、下ほど厳しい。どこで落ちたかを読めるようにするため。

| 層 | 指標 | 何を見るか |
|---|---|---|
| 1 | `rate_per_min` | 相槌の頻度だけ。タイミングも種類も見ない。人間との比 |
| 2 | `timing_f1` | 開始時刻が許容窓内で 1 対 1 対応した F1 |
| 3 | `typed_f1` | 上に加えて相槌の種類が重なることを要求した F1 |
| 4 | `type_accuracy` | timing で一致したペアのうち種類も一致した割合 |

許容窓（`--tolerance-sec`、既定 1.5 秒）を置くのは開ループだからである。同じ
場面でもモデルと人間の相槌位置は厳密には一致しないので、ミリ秒単位の一致を
要求すると測っているのは運になる。

`typed_f1` が落ちたとき、`type_accuracy` を見ればタイミングのせいか種類のせいか
が分かる。種類は `scripts/analyze_backchannels.py` の `AIZUCHI_PATTERNS`
（un / hai / ee / sou / naruhodo / hee / fun / aa / …）を使い、
`confusion_primary_label` に取り違えの内訳が出る。

マッチングは距離昇順の貪欲な 1 対 1。1 対 1 を守らないと、1 つの正解に複数の
予測を当てて再現率を水増しできてしまう。

### 相談員側の F1 について

正解が相談員自身なので、相談員の timing/typed F1 は定義上 1.0 になり意味を
持たない。人間側は `label_distribution.human` と `rate_per_min.human` を記述
統計として読むこと。F1 の実質的な上限を推定するには、同じ場面に対する別の
相談員の相槌が要る（現状のデータには無い）。

## LLM-as-a-judge と相談員のスコア

`eval/pack_real_dialogue_judge_input.py` は 1 ケースにつき 2 行を出す。

- モデルの応答（`system_kind: "model"`）
- 相談員の実際の応答（`system_kind: "human"`, `system_id: "human_counselor"`）

同じ文脈・同じ項目立てで両方を採点させることで、**相談員のスコアがそのケースの
実質的な上限**になる。モデル単体で 4.1 点と言われても高いのか低いのか判断でき
ないので、人間の点を同じ土俵で取っておく。

両者は構造的に同じ形（`assistant_text` / `assistant_timeline`）で入っており、
どちらが人間かを示す情報は本文側に無い。**judge のプロンプトに `system_id` を
渡さないこと。** 渡すと人間側に高い点が付く方向のバイアスが乗る。

pairwise ではなく絶対評価にしてあるのは、開ループのため両者が見ていた文脈が
厳密には同一ではなく、「どちらが良いか」の直接対決は前提が崩れるから。
「そのユーザー発話に対して妥当か」を各々独立に採点する形なら成立する。

judge はローカル PC でのみ実行する（既存の 2 フェーズ方針と同じ）。サーバ側の
実行は API を一切呼ばない。

## 実行

```bash
qsub -v 'WAVS=/path/dialogue1.wav,OUT_DIR=data/real_dialogue/pilot' scripts/run_real_dialogue_analysis.pbs
qsub -v 'WAVS=/path/dialogue1.wav,OUT_DIR=data/real_dialogue/pilot' scripts/run_real_dialogue_separation.pbs
```

人手の作業（`counselor_X.txt` と review TSV）を済ませてから:

```bash
MODEL_ID=lora_h01 MODEL_WEIGHT=/path/model.safetensors ANALYSIS_DIRS="data/real_dialogue/pilot/dialogue1" bash scripts/run_real_dialogue_eval.sh
```

出力:

- `eval_runs/real_dialogue/<run>/benchmark_results/summary.json` — タスク別の決定的指標
- `eval_runs/real_dialogue/<run>/benchmark_results/backchannel_agreement.json` — 相槌一致度
- `eval_runs/real_dialogue/<run>/real_judge_input.jsonl` — judge 入力（モデル + 相談員）

## 主な調整パラメータ

| 変数 | 既定 | 意味 |
|---|---|---|
| `REAL_CONTEXT_SEC` | `30` | 評価点の手前に付ける文脈の長さ。短いほど比較は公平、長いほど自然 |
| `REAL_BC_TOLERANCE_SEC` | `1.5` | 相槌一致とみなす開始時刻の許容差 |
| `--pause-min-sec` | `0.8` | ユーザーターン内のこれ以上の無音を pause とみなす |
| `--backchannel-min-turn-sec` | `6.0` | backchannel タスクに使うユーザー長発話の下限 |
| `--turn-merge-gap-sec` | `0.4` | 同一話者の連続セグメントを結合する間隔。相槌は結合しない |

`REAL_CONTEXT_SEC` は開ループのトレードオフを直接動かす唯一のつまみなので、
値を変えたら別 `RUN_ID` にして混ぜないこと。
