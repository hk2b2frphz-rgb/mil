# 学習のジョブ分割（チェーン実行）

2400 steps で 24h walltime に余裕をもって収まる。それを超える学習は 1 ジョブで
完走できないので、[`scripts/run_train_chain.pbs`](../scripts/run_train_chain.pbs)
でジョブを繋ぐ。

```bash
qsub -v 'EXP_NAME=lora_base_config,SRC_RUN_DIR=data/runs/<RUN_ID>,TOTAL_STEPS=7200' scripts/run_train_chain.pbs
```

これで 3 ジョブ（0→2400、2400→4800、4800→7200）が走る。**投入するのは最初の
1 本だけ**で、各ジョブが自分で次を qsub する。

## 設計

### max_steps はチェーン全体の総数

`max_steps` は全ジョブで同じ値（= `TOTAL_STEPS`）にする。学習率スケジュールが
`max_steps` から作られるためで、ジョブごとに変えると各ジョブの先頭で warmup が
やり直しになり、チェーン全体で 1 本のスケジュールにならない。

各ジョブが担当する区間は `resume_step` / `stop_at_step` で表す。

| ジョブ | resume_from | resume_step | stop_at_step | max_steps |
|---|---|---|---|---|
| 1 | （なし） | 0 | 2400 | 7200 |
| 2 | job1 の ckpt | 2400 | 4800 | 7200 |
| 3 | job2 の ckpt | 4800 | 7200 | 7200 |

`stop_at_step` は絶対 step 番号（区間の長さではない）。

### 実装の位置

- **アダプタの読み込み** — `finetune/wrapped_model.py` の `get_fsdp_model` 内、
  `initialize_lora_parameters` の直後。**FSDP でラップされる前**である必要が
  ある。ラップ後は state_dict のキーにシャードの接頭辞が付き、`strict=False`
  の下で何にもマッチしないまま素通りしてしまうため。
- **step とスケジュールの復元** — `train.py` の `TrainState` 構築直後。
  `max_steps` を `stop_at_step` まで狭めるのは**スケジューラ構築より後**。
  スケジューラは構築時に warmup 長と全体長を取り込むので、この順序なら
  スケジュールはチェーン全体を表したまま、ループの終了条件だけが変わる。
  `is_last_step` が `stop_at_step` で立つので、引き継ぎ点のチェックポイントは
  必ず保存される。

いずれも [`scripts/patch_kyutai_moshi_finetune.py`](../scripts/patch_kyutai_moshi_finetune.py)
が実験起動のたびに冪等に当てる。上流が該当ブロックを変えた場合はパッチが例外を
投げて止まる（黙って諦めない）。

## 制約: optimizer 状態は引き継がれない

上流の `checkpointing.py` はモデルの重みしか保存しない（optimizer 状態を
書き出す経路が無い）。したがって**重みのみの resume** であり、引き継ぎのたびに
AdamW のモーメント推定がゼロから再開する。

`warmup_constant` を使っていれば引き継ぎ時点で学習率は一定域に入っており、
モーメントも数十 step で再蓄積されるので、コストは**短い過渡**であって
スケジュールの断絶ではない。とはいえ継ぎ目であることに変わりはないので、
**短いジョブを何本も繋ぐより、長いジョブを少数繋ぐ方がよい**。

## 早期終了との関係

1 エポック未満しか回らない大規模データでは同じサンプルを二度見ないので過学習が
起きず、early stopping はまず発火しない。実質的に `max_steps` だけが停止条件に
なる。

一方、early stopping が発火した場合は `stop_at_step` に到達しないまま学習が
終わる。チェーンはそれを検出して**次のジョブを投げずに正常終了**する。eval loss
が頭打ちになった後に walltime を燃やしても意味がないため。ログには実際に存在する
チェックポイントの一覧が出る。

## 出力の場所

ジョブごとに独立した run ディレクトリを使う。

```
experiments/<EXP_NAME>/checkpoints/<CHAIN_ID>_job01/
experiments/<EXP_NAME>/checkpoints/<CHAIN_ID>_job02/
```

`RUN_TS` を陽に与えて名前を決定的にしているので、各ジョブは前のジョブの
チェックポイントを glob せずに直接指せる。

best checkpoint の選択と merge/export は各ジョブの末尾で走るので、**最終ジョブの
run ディレクトリ**にある `merged/` が最終成果物になる。ただし選択対象はその
ジョブが書いたチェックポイントだけなので、チェーン全体から選び直したい場合は
`scripts/select_best_checkpoint.py` を全 run ディレクトリに対して手で回す。

## 主なパラメータ

| 変数 | 既定 | 意味 |
|---|---|---|
| `TOTAL_STEPS` | （必須） | チェーン全体の総 step 数 |
| `STEPS_PER_JOB` | `2400` | 1 ジョブあたりの step 数。walltime に収まる範囲で大きく |
| `CHAIN_ID` | `chain_<jobid>` | run ディレクトリの接頭辞 |
| `NPROC` | `1` | GPU 数 |

`CHAIN_INDEX` と `RESUME_FROM` は各ジョブが次のジョブへ渡す内部変数。チェーンを
途中から再開したいとき以外は手で設定しない。

その他の `HP_*` は最初のジョブに渡せば、`qsub -V` で以降の全ジョブへ伝播する。

## 単発で resume したい場合

チェーンを使わず手で続きを回すこともできる。

```bash
qsub -v 'EXP_NAME=lora_base_config,HP_MAX_STEPS=7200,HP_RESUME_STEP=2400,HP_STOP_AT_STEP=4800,HP_RESUME_FROM=/path/to/lora.safetensors' scripts/run_experiment.pbs
```

`HP_RESUME_STEP` を指定して `HP_RESUME_FROM` を忘れると、warmup を飛ばした状態で
初期化直後のアダプタを学習してしまうので、その組み合わせはエラーで停止する。
LoRA の rank が引き継ぎ元と違う場合も、形状不一致として停止する。
