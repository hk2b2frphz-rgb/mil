# 2026-09-02 の分離 PBS

GPU 種別を混ぜないため、次の順で投入します。

```bash
tts_job=$(qsub -V screw_poc/scripts/2026-09-02/data_dialogue_tts.pbs)
qsub -V -W "depend=afterok:${tts_job}" screw_poc/scripts/2026-09-02/train_and_evaluate.pbs
```

- `data_dialogue_tts.pbs`: V100 4 GPU。対話データ作成、左チャンネル（システム）を Kokoro、右チャンネル（ユーザー）を Qwen3-TTS で合成し、`screw_poc/artifacts/2026-09-02_mixed_tts/merged/` に出力します。
- `train_and_evaluate.pbs`: A100 1 GPU。LoRA 学習、未使用の直接質問30件と領域外質問18件（計48件）の Qwen3 音声入力による推論、採点を順に実行します。

評価の出力先は `screw_poc/artifacts/2026-09-02_heldout_<job-id>/score.json` です。件数は `EVAL_LIMIT`、TTS出力先は `OUT_ROOT`、学習側の入力は `TTS_RUN_DIR` で上書きできます。
