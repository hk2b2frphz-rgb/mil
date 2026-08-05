# 実対話リプレイ評価

実際のロールプレイ相談音声を入力に使い、そこで相談員が実際に返した応答を参照と
して Moshi を評価する。合成の [`full_duplex_evaluation.md`](full_duplex_evaluation.md)
とは**別トラック**で、スコアを同じ表に並べてはいけない（理由は後述）。

## 全体の流れ

```
1ch WAV
  └─ 前処理     scripts/analyze_real_dialogue.py     ダイアライゼーション + 書き起こし + test_wav
  └─ 人手       counselor_X.txt を置く / turns_review.tsv で間引く
  └─ 構築       eval/build_real_dialogue_dataset.py  FDB 形状のデータセット
  └─ 推論・評価 scripts/run_real_dialogue_eval.sh    推論 → 決定的評価 → 相槌一致度 → judge 入力
  └─ ローカル   eval/judge_openai.py                 LLM-as-a-judge（ローカル PC のみ）
```

## テスト入力（test_wav）

ダイアライゼーションは息継ぎごとに区間を切るので、そのままではテスト入力として
細かすぎる。`test_wav/` には**同一話者の連続発話をまとめ直した塊**を出す。

- 間隔が `--turn-gap-sec`（既定 1.0 秒）以下の同一話者セグメントを繋ぐ
- **間に相手の相槌が挟まっても繋ぐ**。相手の相槌でこちらのターンは切れない
- `--turn-max-sec`（既定 30 秒）を超えたらそこで切る
- `--turn-min-sec`（既定 0.5 秒）未満の塊は出さない

切り出し元は `speaker_<話者>_solo.wav`（その話者の区間以外を無音化したトラック）
なので、塊の途中に入る相手の相槌はここで無音になる。

### Whisper 定型文の除去

Whisper は無音・雑音区間で「ご視聴ありがとうございました」「次回予告」のような
動画定型文を吐く。実際に喋られた音ではなく ASR の作話なので、残すと塊の境界ごと
壊れる。`analyze_real_dialogue.py` の `HALLUCINATION_SUBSTRINGS` /
`HALLUCINATION_EXACT` に該当するものを `test_wav` から除外し、除外した内容は
ログに出す。`timeline.jsonl` には `hallucination` 印を付けて残すので、後から
判定の妥当性を確認できる。

「ありがとうございました」単体は**落とさない**。相談の締めで実際に使われるため、
落とすと本物の発話が消える。

## 重畳区間に相手の声が残る（重要な制約）

1ch 録音では重畳区間に両者の声が同じ標本へ混ざっている。solo トラックは「その
話者の区間以外を無音化」しただけなので、**重畳区間の相手の声は消えない**。

これが問題になるのは、その混入音が相談員の相槌でありうるからである。モデルへの
入力に相談員の「ええ」「はい」が残っていると、相槌の打ち方を評価したいのに、
入力に正解が入っている状態になる。

以前は音源分離（SepFormer）をステージ2 として挟んでいたが、廃止した。現在は
**重畳した塊を人手で落とす**運用で対処する。`test_wav` のファイル名に `_ov` が
付いているものが重畳を含む塊で、`turns_review.tsv` の `overlap_sec` 列にも秒数が
出る。相手の声が混ざって困るものは `keep=0` にすること。

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

`test_wav/turns_review.tsv`（UTF-8 BOM 付き TSV、Excel でそのまま開ける）。

| 列 | 意味 |
|---|---|
| `keep` | `1` で採用、`0` で除外。ここだけ編集する |
| `index` | `test_wav/<話者>/` の WAV ファイル名の通し番号と一致 |
| `speaker` | `A` / `B` |
| `start_sec` / `end_sec` / `duration_sec` | 位置と長さ |
| `n_segments` | まとめる前のセグメント数 |
| `overlap_sec` | 相手と重なっていた秒数（0 より大きいものは要確認） |
| `aizuchi_count` | 塊に含まれる相槌の数 |
| `text` | 書き起こし（塊内を連結したもの） |
| `wav` | 対応する切り出し WAV |

ユーザー側の `keep=0` にした行は、評価点の候補から外れる。

#### 編集を WAV に反映する

塊まとめは自動なので、境界がずれたり話者を取り違えたりする。TSV で直してから
作り直す。**TSV が正、WAV はその出力**という関係にしてある。

```bash
uv run python scripts/rebuild_test_wav.py data/real_dialogue/pilot/対話1
```

GPU も NeMo も Whisper も要らない（numpy と soundfile だけ）。切り出し元は分析
ディレクトリに既にある `speaker_<話者>_solo.wav` なので、元の WAV も要らない。

| やりたいこと | TSV での操作 |
|---|---|
| 間引く | `keep` を `0` にする |
| 境界を直す | `start_sec` / `end_sec` を書き換える |
| 話者を直す | `speaker` を `A` / `B` で書き換える |
| 塊を分ける | 行をコピーして、それぞれの `start_sec` / `end_sec` を狭める |
| 塊を繋ぐ | 片方の `end_sec` を伸ばし、もう片方を `keep=0` にする |
| 書き起こしを直す | `text` を書き換える（ファイル名に反映される） |

行は自由に足してよい。`index` は空でも重複していても構わない（その場合は空いて
いる番号が自動で振られる）。並び順も見ない。

`index` を尊重するのは、手元で聴いていた WAV のファイル名が作り直しで変わると
突き合わせられなくなるため。行を足した場合だけ新しい番号になる。

`--dry-run` を付けると WAV を書かずに採用件数だけ確認できる。`end_sec <=
start_sec` のような不正な行があると、**一件でも WAV を書く前に**行番号付きで
報告して停止する。既存の `test_wav/<話者>/*.wav` は消して作り直すが、
`turns_review.tsv` 自体は書き換えない。

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
qsub -v 'WAVS=/path/dialogue1.wav,OUT_DIR=data/real_dialogue/pilot' scripts/run_real_dialogue_pipeline.pbs
```

人手の作業（`counselor_X.txt` と `turns_review.tsv`）を済ませてから:

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
| `--turn-gap-sec` | `1.0` | test_wav の塊にまとめる間隔 |
| `--turn-max-sec` | `30.0` | 1 塊の上限。超えたら切る |
| `--turn-min-sec` | `0.5` | これ未満の塊は test_wav に出さない |
| `--min-segment-sec` | `0.02` | これ未満のダイアライゼーション断片は捨てる |
| `--pause-min-sec` | `0.8` | ユーザーターン内のこれ以上の無音を pause とみなす |
| `--backchannel-min-turn-sec` | `6.0` | backchannel タスクに使うユーザー長発話の下限 |
| `--turn-merge-gap-sec` | `0.4` | 同一話者の連続セグメントを結合する間隔。相槌は結合しない |

`REAL_CONTEXT_SEC` は開ループのトレードオフを直接動かす唯一のつまみなので、
値を変えたら別 `RUN_ID` にして混ぜないこと。
