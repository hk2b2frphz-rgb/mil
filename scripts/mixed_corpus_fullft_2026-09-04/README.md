# 相づち＋本応答 混合コーパスでの full fine-tuning (2026-09-04)

**何を学習するか**: 相づちだけのコーパス（`aizuchi-only`）と、moshi が実際に応答する
コーパス（`multi-agent`）を混ぜた 1 つのデータセットで Moshi LM を full fine-tuning
する。狙いは、**相づちを打ち続けながら、必要なときには本応答を喋れる**モデル。

- 学習手法: 教師あり full fine-tuning（nu-dialogue/moshi-finetune、`parameters_to_finetune=all`）
- 学習データ: 相づち 10,000 対話 ＋ 本応答 10,000 対話（混合比は可変）
- 評価: Full-Duplex-Bench-JA（best checkpoint で自動実行）

外部 LLM の回答をテキストストリームに注入して読み上げさせる構成の**前提となる
ベースモデル**をここで作る。注入ロジック自体はこのパイプラインには含まない
（末尾の「含まれないもの」を参照）。

## なぜ混合コーパスが要るのか

現行の `aizuchi_normal_10000_v2` は `sanitize_aizuchi_only_turns`
（[generate_synthetic_moshi_training_data.py:3195](../generate_synthetic_moshi_training_data.py)）
が moshi 側の**語彙外発話を機械的に全部落とす**ので、学習後のモデルは
「はい。」「そうですか…。」しか発話経験がありません。

このモデルに外部 LLM の回答をテキストストリーム経由で注入しても、

- 長い実トークン列に対応する acoustic コードを持っていない → 声が荒れる
- 実トークンを連続で出す挙動を学んでいない → 常に PAD を出したがり、注入が消化されない

という形で破綻します。そこで **systemAI の本応答を含む multi-agent コーパスを混ぜ**、
「相づちを打ち続ける」挙動と「本応答を喋る」挙動を 1 つのモデルに持たせます。

**混合比がこのパイプラインの実験変数**です。少なすぎると注入が破綻したまま、
多すぎると相づちのタイミング特性が普通のターン制アシスタントに戻ります。本応答側も
10,000 対話生成しておき（既定で 1:1）、実際の比率は stage 3 の `MIX_TAKE_RESPONSE`
で下げながら振ります。データを作り直さずに比率だけ変えられるのが、この構成の狙いです。

## 全体像

| stage | ファイル | 何をするか | 出力 |
|---|---|---|---|
| 0a | `10_dialogue_aizuchi.pbs` | （任意）相づちコーパスを作り直す | `data/runs/aizuchi_normal_10000_v2/dialogue` |
| 0b | `20_tts_aizuchi.pbs` | （任意）その TTS | `data/runs/aizuchi_normal_10000_v2/tts/merged` |
| 1 | `11_dialogue_response.pbs` | **本応答**コーパスを生成（multi-agent） | `data/runs/response_10000_v1/dialogue` |
| 2 | `21_tts_response.pbs` | その TTS | `data/runs/response_10000_v1/tts/merged` |
| 3 | `30_mix_corpus.pbs` | 2 つを比率指定で混合 | `data/runs/mixed_normal_v1/mixed` |
| 4 | `40_train_eval.pbs` | full-FT → best ckpt export → Full-Duplex-Bench-JA | `eval_runs/full_duplex/<MODEL_ID>/` |

**stage 0a/0b は通常実行しません。** 相づち側は 2026-09-04 の成果物をそのまま
再利用し、stage 3 が既定でそこを見ています。0a/0b はその 2026-09-04 のジョブを
呼ぶだけの薄いラッパで、チューニング値は向こうに一元化してあります。

## 実行

stage 1 を投げるだけで 2 → 3 → 4 まで自動で連鎖します（各ジョブが末尾で次を qsub）。

```bash
qsub -V scripts/mixed_corpus_fullft_2026-09-04/11_dialogue_response.pbs
```

段ごとに手で回したいときは `CHAIN=0`:

```bash
qsub -V -v CHAIN=0 scripts/mixed_corpus_fullft_2026-09-04/11_dialogue_response.pbs
```

TTS が 24h に収まらなかった場合は、**同じファイルをもう一度投げれば再開**します
（完了した shard・対話・リクエストはスキップ）。stage 2 は merged マニフェストが
まだ無ければ自分自身を再投入し、揃った時点で stage 3 へ進みます。

## よく変える値

```bash
# 本応答コーパスのサイズ（既定 10000 = 相づち 10000 と同数）
qsub -V -v NUM_CASES=6000 scripts/mixed_corpus_fullft_2026-09-04/11_dialogue_response.pbs

# 混合比だけ振り直す（stage 3 から。データは作り直さない）
qsub -V -v MIX_TAKE_RESPONSE=2000,MIX_VERSION=v2 scripts/mixed_corpus_fullft_2026-09-04/30_mix_corpus.pbs
qsub -V -v MIX_TAKE_RESPONSE=0.3,MIX_VERSION=v3  scripts/mixed_corpus_fullft_2026-09-04/30_mix_corpus.pbs

# 韻律のみのアブレーション（テキスト方策を凍結し、depformer だけ動かす）
qsub -V -v NU_PARAMETERS_TO_FINETUNE=depformer scripts/mixed_corpus_fullft_2026-09-04/40_train_eval.pbs

# ベンチマークを回さず学習だけ
qsub -V -v SKIP_AUTO_EVAL=1 scripts/mixed_corpus_fullft_2026-09-04/40_train_eval.pbs
```

混合は音声をコピーせず絶対パスで参照するだけなので、**stage 3 は数秒で終わり、
比率のスイープが安い**のが利点です。TTS をやり直す必要はありません。

## 設計上、動かしてはいけないところ

**両コーパスの moshi 音声は同一クローン元** — `data/clone_examples/99999`。
stage 2 の `CLONE_OUT_DIR_MOSHI` はオプションではなく固定です。半分ずつ別の声に
なると、同じ話者に対して acoustic codebook の目標が二峰性になり、full-FT で
声が不安定になります（v1 TTS がこれを踏んで両チャンネル CustomVoice に落ちた）。

**対話長は 170s 未満に収める。** `moshi-finetune` は `duration_sec` を超える対話を
等分割し、2 個目以降のチャンクは冒頭挨拶なしで始まります。モデルはそれを通話の
先頭と区別できず、挨拶を飛ばして相づちから始めるよう学習します（v1 aizuchi モデルの
失敗そのもの）。stage 1 が `MULTI_AGENT_MIN_PAIRS/MAX_PAIRS` を 3/5 に据え置いて
いるのはこのためです。`mix_summary.json` に p95 / max / 170s 超過率が出るので、
stage 4 の冒頭ログで必ず確認してください。

## 学習側の現状値（stage 4）

- `--parameters_to_finetune`: **`all`**（launcher の既定。tempformer + depformer 全部）
- 損失重み: `semantic=100.0` / `acoustic=1.0` / `text_padding=0.5`（すべて
  `finetune.py` の既定値。launcher はフラグを渡していない）
  - semantic 1 本 : acoustic 7 本なので、**音声損失の約 93% が内容側、韻律・声質は約 6.5%**
  - 韻律を主目的にする実験では `--acoustic_loss_weight` を上げる必要があり、これは
    `scripts/run_nu_fullft_experiment.sh` の `LAUNCH_CMD` への追記が要ります
    （env では未公開）
- pattern `f01`: `lr=7e-6` / weight_decay 0.1 / 実効 batch 8 / 12 epochs /
  warmup 1 epoch / early stopping patience 4 evals / `duration_sec=170`
- `--model_user_stream` 有効（ユーザ側ストリームにも損失がかかる）

## このパイプラインに含まれないもの

推論時のテキストストリーム注入そのもの（`lm_gen.step` のパッチと、
LISTEN / PENDING / SPEAK の状態機械）は**未実装**です。ここで作るのは、その注入を
成立させるための**モデル**まで。

順序として、注入ロジックを書く前に stage 4 の結果で
**「注入なしで本応答が自然に喋れるか」**を先に確認してください。ここが通らないまま
注入を実装すると、音が荒れた原因が注入ロジックなのかコーパスなのか切り分けられなく
なります。

## 参照

- 混合スクリプト: [scripts/mix_training_manifests.py](../mix_training_manifests.py)
- 相づち側の設計判断: [scripts/2026-09-04/aizuchi_normal_dialogue_10000.pbs](../2026-09-04/aizuchi_normal_dialogue_10000.pbs)
- ハイパラの根拠: [experiments/fullft_base_config/HYPERPARAMS.md](../../experiments/fullft_base_config/HYPERPARAMS.md)
- sweep pattern 定義: [scripts/run_fullft_sweep_pair.sh](../run_fullft_sweep_pair.sh)
