# Qwen3.6-27BをV100 32GB × 4枚で使う

対話生成用のV100構成を追加しました。既存のA100用ジョブとは別のPython環境を使い、
同じ対話生成処理・400背景・相談目的などの5軸を利用します。
クラスタ実機でのビルド・生成は未検証です。まず20件で確認してください。

## 初回セットアップと少量生成

クラスタのリポジトリ直下で実行します。プロキシが必要なら、従来どおり`PROXY_URL`を設定します。

```bash
setup_job=$(qsub -V scripts/setup_vllm_qwen_v100_env.pbs)
NUM_CASES=20 qsub -V -W "depend=afterok:$setup_job" scripts/run_dialogues_qwen_v100_4gpu.pbs
```

セットアップ済みなら次の1行だけで実行できます。

```bash
NUM_CASES=20 qsub -V scripts/run_dialogues_qwen_v100_4gpu.pbs
```

生成ジョブもセットアップの確認を行います。環境がなければそこでビルドしますが、
初回は1GPUのセットアップジョブを先に使うと、ビルド中に4GPUを占有せずに済みます。
既定の生成数は3000件です。少量生成を確認後、`NUM_CASES=3000`で投入してください。

## 実行条件

| 項目 | 設定 |
|---|---|
| PBS | `xvn_s` / `select=1:res=middle2`（4GPU） |
| モデル | `Qwen/Qwen3.6-27B` |
| 精度・分散 | FP16・TP=4、量子化なし |
| 実行環境 | `.venv-vllm-qwen-v100` |
| vLLM | 0.21.0、CUDA 12.6でsm70向けソースビルド |
| PyTorch | 2.11.0+cu126（torchaudio・torchvisionもcu126で固定） |
| Transformers | 4.57.1。Qwen設定はvLLM側のレジストリで読む |
| Attention | `TRITON_ATTN`、テキストのみ、chunked prefill有効 |
| 起動時設定 | CUDA graphs無効、会話長16K、同時実行8、prefillバッチ1024トークン |
| Thinking | 無効 |

V100 16GB × 4枚はこの構成の対象外です。各GPUが30GiB以上あり、V100が4枚見えていることを検査します。
PyTorchのCUDA版・sm70対応・vLLMの版・拡張モジュールも検査し、起動後は日本語JSONを生成します。
JSONが不正、途中切れ、Thinkingが混入する場合は、本番の対話生成を開始せず失敗します。
モデルを別のものやテンプレートへ自動で置き換えません。

既存のA100用ジョブがインストールするcu129配布版で、V100用環境を上書きしないようにしています。
TTS用の`.venv-vllm-omni`にもインストールしません。

## 出力と変更できる項目

20件の既定出力は`data/runs/qwen_dialogues_v100_4gpu_20_v1/llm_dialogues/dialogues.jsonl`です。
ジョブログは`experiments/pbs_logs/`、ビルドログは`vllm_qwen_v100_build.log`です。
生成数ごとに出力先が分かれ、完了済みデータへの上書きは既存ジョブと同じく拒否します。
新しい生成ラウンドでは`DIALOGUES_VERSION=v2`などを指定してください。

`MAX_MODEL_LEN`、`MULTI_AGENT_CONCURRENCY`、`VLLM_MAX_NUM_SEQS`、
`VLLM_MAX_NUM_BATCHED_TOKENS`を調整できます。最初は既定値を使ってください。
`QWEN_V100_CUDA_MODULE`、`QWEN_V100_CUDA_VISIBLE_DEVICES`でクラスタ固有の設定を変更できます。
環境の場所を変える場合は`VLLM_QWEN_V100_ENV_DIR`を使い、A100・TTS環境とは別の場所を指定します。

対話生成モードや`INTERACTION_TRACK`などは既存ジョブと同じ指定ができます。
終了希望のデータを生成する場合は`INTERACTION_TRACK=end-request`と
`DIALOGUE_GENERATION_MODE=multi-agent`を指定し、継続相槌データとは別の`BATCH_ID`にしてください。
3条件比較の一括投入は従来のA100構成のままです。

TTSへ渡す例（20件の既定出力）：

```bash
SOURCE_BATCH_ID=qwen_dialogues_v100_4gpu_20_v1 qsub -V scripts/run_qwen_tts_whole_utterance_3000_4gpu.pbs
```

## 検証

ローカルではGPU環境の異常検知、日本語JSONの検査、4GPU設定の引き渡し、
既存環境の再利用、検査失敗時に生成しないことをモックで確認しています。
実機のCUDAカーネル・GPU間通信・日本語対話品質・速度は、クラスタの少量生成で確認が必要です。

参考：[vLLM 0.21.0のCUDAビルド条件](https://github.com/vllm-project/vllm/blob/v0.21.0/CMakeLists.txt)、
[V100でのQwen3.6-27B実測報告](https://github.com/KumphanartDansiri/v100-vllm-2026/blob/main/models/qwen3_6_27b.md)。
