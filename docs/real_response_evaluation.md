# 実データ応答評価

人手アノテーション由来のテストデータ（[real_dialogue_test_data.md](real_dialogue_test_data.md)）を
入力に、モデルの**応答率・応答速度・音声品質**を測る。合成 Full-Duplex-Bench-JA
とは**別トラック**で、スコアを同じ表に並べてはいけない。

## 指標

| 指標 | 定義 |
|---|---|
| 応答率 `response_rate` | 聞き取れる応答を返したケースの割合。無音のまま終わったものは失敗。テキストだけ出て音が出ていない場合も失敗 |
| 応答速度 `response_latency_sec` | User の発話終了から応答音声が鳴り始めるまでの秒数。**応答したケースだけ**で集計する |
| 音声品質 `utmos` | UTMOS（自動MOS、参照音声不要）。応答音声の区間だけを切り出して掛ける |

応答速度を応答したケースだけで集計するのは、応答しなかったものを 0 秒や無限大と
して混ぜると平均が意味を失うため。**応答率と必ず併せて読むこと。** 応答率が低い
モデルは、たまたま返した速い応答だけで平均が良く見える。

音声品質を応答区間に限るのは、先頭の無音を含めたまま掛けると値が無音側へ引っ
張られるため。

## gold（実際の相談員）

マニフェストに `model_id=gold` の行を入れると、**モデルを動かさずに実際の相談員
の応答を出力として置き**、他のモデルと同じコードで採点する。

```
gold||||||
```

gold は競争相手ではなく、**このデータで到達しうる水準**として読む。モデル単体で
「応答速度 0.9 秒」と言われても速いのか遅いのか判断できないので、人間の値を同じ
土俵で取っておく。

gold の音声は元の 1ch 録音から切り出すので、相談員の応答に User の声が重なって
いればそれも残る。「人間の声だから満点」という数字にはならない。

gold の応答率も 100% にはならない。ユーザーが話し終えても相談員が何も返さなかっ
た場面が実データには含まれるため。データセット構築時に件数が出る。

### 「相談員が応答した」の判定

`eval/build_real_test_dataset.py` の `gold_response()` が、アノテーションの時刻
だけで決める。起点は User 発話の終了時刻で、そこから `--gold-window-sec`（既定
30 秒）以内で**最初に始まる Staff 発話**が応答の頭。以降 `--gold-reply-gap-sec`
（既定 3 秒）以内で続く Staff 発話は同じ応答としてまとめる。

**次の User 発話では窓を閉じない。** 1ch のアノテーションでは User の 1 ターンが
複数行に割れていることが多く、以前は次の User 行が発話終了の直後に来るだけで窓
の幅がほぼ 0 になり、相談員が実際に応答していても「無応答」に落ちていた。ユーザー
が喋り続けたことは、相談員が応答しなかったことを意味しない。旧挙動は
`--gold-next-user-closes` で戻せる。

応答が始まる前に User が次の行を喋り出していたかは
`human_reference.user_continued_before_reply` に残るので、後から絞り込める。

判定を疑うときは、無応答とされたケースの周辺を並べる:

```bash
uv run python eval/diagnose_gold_no_response.py
```

## 実行

前提として、テストデータが作ってあること（`data/test_data/real_dialogue`）。

```bash
qsub -v MODELS_FILE=scripts/models_manifest.txt scripts/run_full_duplex_eval_batch.pbs
```

マニフェスト（`scripts/models_manifest.txt`）の例:

```
gold||||||
base||||||
lora_h01|/path/to/lora_h01/consolidated.safetensors|||||v1
full_f01|/path/to/full_f01/model.safetensors|||REAL_SEEDS=0,1,2||full_f01_seed012
```

1 モデルだけ回すなら PBS を経由しなくてもよい。

```bash
MODEL_ID=gold bash scripts/run_real_eval.sh
```

## 出力

```
eval_runs/full_duplex_batches/<batch>/
├── combined_summary.json          モデル横並びの表（gold が先頭）
├── batch.log
└── <output_name>/
    ├── benchmark_results/summary.json     指標の集計
    ├── benchmark_results/per_case.jsonl   ケース単位
    ├── real_judge_input.jsonl             LLM-as-a-judge 入力
    └── inference/                          応答音声
```

ジョブの最後に表が出る。

```
output_name                   応答率     遅延p50     遅延p90   UTMOS  status
---------------------------------------------------------------------
gold                        0.941     0.420     1.180    3.82  ok
v1                          0.887     0.910     2.340    3.41  ok
base                        0.612     1.740     4.020    3.05  ok
```

## LLM-as-a-judge

今まで通り `real_judge_input.jsonl` を出す。サーバ上では API を呼ばず、ローカル
PC で `eval/judge_openai.py` に掛ける（既存の 2 フェーズ方針のまま）。

1 ケースにつきモデルの応答と相談員の実応答の 2 行が出て、相談員のスコアがその
ケースの実質的な上限になる。`system_id` を judge のプロンプトに渡さないこと。
渡すと人間側に高い点が付く方向のバイアスが乗る。

`model_id=gold` の実行では相談員側の行は重複させない（自分自身がそれなので）。

## 主な調整

| 変数 | 既定 | 意味 |
|---|---|---|
| `REAL_SEEDS` | `0` | シード |
| `REAL_TAIL_SEC` | `8` | 応答を待つ長さ。これを過ぎたら応答なしと判定される |
| `REAL_CASES_PER_TASK` | 全件 | スモーク実行（先頭 N 件） |
| `REAL_CONTEXT_SEC` | `0` | User 発話の手前に付ける文脈 |
| `REAL_MOS_BACKEND` | `utmos` | `none` で音声品質を測らない |
| `REAL_MOS_DEVICE` | `cpu` | UTMOS のデバイス |
| `REAL_MAX_LATENCY_SEC` | 無制限 | これを超える応答を「応答なし」と扱う |
| `REBUILD_REAL_DATASET` | `0` | 共有データセットを作り直す |

`REAL_CONTEXT_SEC` は既定 0 のままにしておくこと。0 より大きくすると、1ch 録音
なので**入力に相談員の声が混ざる**。応答の打ち方を評価したいのに、入力に正解が
入っている状態になる。

## 合成 Full-Duplex-Bench-JA は

削除していない。行ごとに `extra_env` で `FDB_SYSTEM=moshi` / `FDB_SYSTEM=cascade`
を指定すれば従来の評価が走る。ただし既定ではなくなった。両者のスコアは別トラック
なので、同じ表で読まないこと。
