# 2026-09-02 の分離 PBS

GPU 種別を混ぜないため、次の順で投入します。

```bash
tts_job=$(qsub -V screw_poc/scripts/2026-09-02/data_dialogue_tts.pbs)
qsub -V -W "depend=afterok:${tts_job}" screw_poc/scripts/2026-09-02/train_and_evaluate.pbs
```

- `data_dialogue_tts.pbs`: V100 4 GPU。対話データ作成、左チャンネル（システム）を Kokoro、右チャンネル（ユーザー）を Qwen3-TTS で合成し、`screw_poc/artifacts/2026-09-02_mixed_tts/merged/` に出力します。
- `train_and_evaluate.pbs`: A100 1 GPU。LoRA 学習、未使用の直接質問30件と領域外質問18件（計48件）の Qwen3 音声入力による推論、採点を順に実行します。評価部分は `evaluate_only.pbs` を呼び出すだけなので、実装は 1 か所だけです。
- `evaluate_only.pbs`: A100 1 GPU。学習はせず、評価だけを実行します。

評価の出力先は `screw_poc/artifacts/2026-09-02_heldout_<job-id>/score.json` です。件数は `EVAL_LIMIT`、TTS出力先は `OUT_ROOT`、学習側の入力は `TTS_RUN_DIR` で上書きできます。

## 評価だけをやり直す / 続きから流す

`evaluate_only.pbs` は再開できます。TTS 入力が揃っていれば作り直さず、推論も transcript がまだ無い問いだけを流します。walltime で落ちても同じ `EVAL_OUT` で投げ直せば続きから進みます。

```bash
# 学習済みの最新モデル（merged 配下で一番新しい consolidated.safetensors）を評価
qsub -V screw_poc/scripts/2026-09-02/evaluate_only.pbs

# 途中で落ちた train_and_evaluate.pbs の評価を、その出力先で続きから流す
qsub -V -v EVAL_OUT=$PWD/screw_poc/artifacts/2026-09-02_heldout_<job-id> \
    screw_poc/scripts/2026-09-02/evaluate_only.pbs

# 任意のモデルを評価する（出力先も分けておくと比較しやすい）
qsub -V -v MODEL_WEIGHT=/path/to/consolidated.safetensors,EVAL_OUT=$PWD/screw_poc/artifacts/2026-09-02_heldout_other \
    screw_poc/scripts/2026-09-02/evaluate_only.pbs
```

`MODEL_WEIGHT` 未指定時は `screw_poc/experiments/lora_base_config/merged/` 配下で最も新しい `consolidated.safetensors` を使います。`FORCE_INPUTS=1` で TTS 入力の作り直し、`FORCE_INFERENCE=1` で全問の推論やり直しです。
