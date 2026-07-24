# moshimoshi-J

The Japanese Full-Duplex-Bench v1/v1.5 evaluation follows upstream commit
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e` with only documented English-to-Japanese adaptations. See [docs/full_duplex_evaluation.md](docs/full_duplex_evaluation.md).
Evaluation input audio uses Qwen3-TTS, with automatic `pyopenjtalk` fallback when Qwen3-TTS is unavailable.

日本語分野C窓口向けに Moshi (llm-jp/llm-jp-moshi-v1) を LoRA で
ドメイン適応するためのパイプライン。合成データの生成から fine-tune まで一気通貫で回せる。

## パイプライン概要

```
1. use_cases.jsonl          ← 軸の組み合わせで多様な対話ケースを生成
       ↓ build_use_cases.py
2. dialogues.jsonl          ← Gemma 4 が感情・沈黙付き対話スクリプトを生成
       ↓ generate_synthetic_moshi_training_data.py
3. ステレオ WAV + manifest  ← Qwen3-TTS で音声合成 (moshi=左ch, user=右ch)
       ↓ generate_qwen3_tts_data.py
4. Moshi LoRA fine-tune     ← moshi-finetune で学習
```

## セットアップ

```bash
# moshi-finetune を兄弟ディレクトリに clone
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune

# 依存の sync
uv sync                           # Moshi + Qwen3-TTS
uv sync --project gemma_runtime   # Gemma 4
```

**要件**: Python 3.11+, NVIDIA GPU (A100 80GB 推奨), CUDA 対応 PyTorch

PBS計算ノードで外部モデルを取得するためにプロキシが必要な環境では、
送信元の`HTTP_PROXY`を暗黙利用せず明示的に渡す。

```bash
qsub -v PROXY_URL=http://<proxy-host>:<port>,MODEL_ID=base \
  scripts/run_full_duplex_eval.pbs
```

`socket.getaddrinfo: Name or service not known`が出る場合は、指定した
プロキシホストが計算ノードから名前解決できるか確認する。

実験DAGの自己連結 smoke も同じ方法で渡せる。最初の `qsub` と、各計算
ノードから投入される次段の `qsub -V` に引き継がれ、各段の
`scripts/setup_proxy.sh` が `HTTP_PROXY` / `HTTPS_PROXY` を設定する。

```bash
PROXY_URL=http://<proxy-host>:<port> \
  bash scripts/smoke_experiment_dag.sh pbs
```

## 使い方

### フルパイプライン (データ生成 → 学習)

```bash
# 100 対話 (~1h) を生成して学習まで一気に回す
bash scripts/run_pipeline.sh

# 対話数やステップを環境変数で調整
NUM_CASES=250 bash scripts/run_pipeline.sh
STEPS=use_cases,dialogues bash scripts/run_pipeline.sh   # 音声化前まで
```

### 実験ベース (推奨)

ハイパラを変えて比較する場合は `experiments/` 配下で管理する。

```bash
# 既存データで実験を起動
bash scripts/run_experiment.sh lora_base_config ./data/runs/2026-06-02_130539

# データ生成 + 実験をまとめて実行
bash scripts/generate_and_run.sh lora_base_config 250
```

### 既存の学習データを指定して実行

生成済みの学習データを使う場合は、`SRC_RUN_DIR` に
`training_set/synthetic_moshi_train.jsonl` を含む run ディレクトリを指定する。

```bash
# 例:
#   ./data/runs/3h_dataset/training_set/synthetic_moshi_train.jsonl

export SRC_RUN_DIR=./data/runs/3h_dataset

# full fine-tuning sweep. Default is f01 only; override patterns for tuning.
qsub scripts/fullft_sweep.pbs

# 必要なら sweep pattern を上書きする。
qsub -v SRC_RUN_DIR=./data/runs/3h_dataset,SWEEP_PATTERNS=f01,f04 scripts/fullft_sweep.pbs
```

PBS を使わずにシェルから直接試す場合:

```bash
SRC_RUN_DIR=./data/runs/3h_dataset \
SWEEP_PATTERNS="f01 f02" \
bash scripts/run_fullft_sweep_pair.sh
```

同じ `SRC_RUN_DIR` は LoRA sweep と単発PBSにも使える。

```bash
qsub -v SRC_RUN_DIR=./data/runs/3h_dataset,SWEEP_PATTERNS=h01,h02 scripts/sweep_lora.pbs
qsub scripts/run_experiment.pbs
```

詳細は [experiments/README.md](experiments/README.md) 参照。

### Whole-utterance TTS (推奨)

話者ごとに全発話を連結して1回のTTSで合成し、MMS_FA Forced Alignment で
セグメント境界を復元する。韻律が一貫し、相槌もユーザー発話を止めずに
自然にオーバーラップする。

```bash
# smoke (1GPU, 3件)
qsub -V scripts/run_qwen_tts_whole_utterance_smoke.pbs

# 全件 (4GPU並列)
qsub -V scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs

# スタイルプリセットを固定したい場合
STYLE_PRESET=counseling_anxious qsub -V scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs
```

バックエンドは smoke/1000 が Qwen3-TTS、**3000/10000 は既定で Kokoro**
（`hexgrad/Kokoro-82M`。Qwen3-TTS ではこの規模は walltime 内に終わらないため）。
1000 スクリプトも `TTS_BACKEND=kokoro qsub -V scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs`
で、3000 と同じ Kokoro バックエンドを選択できる（既定は互換性のため Qwen3-TTS）。
Kokoro は emotion/style instruct 非対応（`STYLE_PRESET` は無視される）。
moshi=`jf_alpha` 固定、user は日本語4話者（jf_gongitsune/jf_nezumi/jf_tebukuro/jm_kumo）を
対話ごとにローテーション（`KOKORO_VOICE_MOSHI` / `KOKORO_USER_VOICES` で変更、
`TTS_BACKEND=qwen3` で旧挙動）。Kokoro の依存は隔離 uv 環境でジョブ内に自動構築される。

#### Qwen3-TTS 1.7B / vLLM-Omni（V100 32GB × 4）

Qwen3-TTSの音質とstyle instructを維持して10,000対話を生成する場合は、
vLLM-Omni専用ジョブを使う。各V100に1.7Bモデルを1つずつ複製し、各シャードで
複数対話の発話を長さ順にまとめてbatch 16で生成する。tensor parallelは使わない。

```bash
# 初回のみ。V100が見える計算ノード上でPython 3.12環境を作成する。
# PROXY_URL は setup_proxy.sh により pip と git の両方へ反映される。
PROXY_URL=http://<proxy-host>:<port> \
  bash scripts/setup_vllm_omni_v100_env.sh

# CMake configure + CUDA コンパイルが長いので、インタラクティブではなく
# 24h の PBS ジョブ（xvn_s / res=small = V100 1枚）で回す場合はこちら。
# ログインノードから tail -f <repo>/vllm_build.log で進捗を追える。
# 再投入は安全: 完成済みなら即終了、途中なら venv と vLLM checkout を
# 使い回してビルドは差分から続く（毎回フル再コンパイルにはならない）。
# 完全に作り直したいときだけ FRESH=1 を付ける。
PROXY_URL=http://<proxy-host>:<port> \
  qsub -V scripts/setup_vllm_omni_v100_env.pbs

# まず100対話で速度・VRAM・音質を確認
NUM_DIALOGUES=100 FRESH=1 qsub -V scripts/run_qwen_tts_vllm_10000_4gpu.pbs

# 10,000対話本番。同じBATCH_IDで再投入すると途中から再開する
qsub -V scripts/run_qwen_tts_vllm_10000_4gpu.pbs
```

既定は `TTS_BATCH_SIZE=16`、`DIALOGUE_BATCH_SIZE=16`、`float16`。
V100はbfloat16非対応なので変更しない。OOM時は両方を8に下げる。専用環境は
`cuda12.6_cudnn9.7.1_nccl2.24.3` module と PyTorch 2.11の `cu126` wheelを使う。
cu128/cu129 wheelはV100のsm70 kernelを含まないため使わない。セットアップは
PyTorchのsm70対応とCUDA実行を先に検査し、そのPyTorchに対してvLLM 0.21.0を
`TORCH_CUDA_ARCH_LIST=7.0` でソースビルドする。初回ビルドには時間がかかる。
ビルドは ninja 進捗を流しつつ `vllm_build.log`（共有FS）へ tee するので、計算
ノードに入れなくても、ログインノードから `tail -f vllm_build.log` で進捗を追える。
`[N/M] Building CUDA object ...` が進めば健全、`Fetching`/configure で停止したまま
なら FetchContent のネットワーク固着を疑う。`MAX_JOBS` は既定でノードのコア数
（`nproc`）に連動する。メモリ不足なら明示的に下げる。ノードのシステム CMake が
古い（3.26 未満、例: 3.22.1）とビルドが落ちるため、モダンな CMake と ninja を
venv に入れ、venv の `bin` を PATH 先頭に置いてそちらを使う。
実行ジョブも同じCUDA moduleを自動loadする。変更が必要なら
`VLLM_CUDA_MODULE`、`VLLM_VERSION`、`VLLM_OMNI_VERSION` を明示する。
出力形式、MMS_FA alignment、resume、4シャードのマージは従来ジョブと同じ。

#### 100対話の新規速度テスト

本番前の速度・VRAM・出力整合性確認は、既存の対話バッチから先頭100件を
4 GPU に分割して行う。`BATCH_ID` に当日の日付と時刻を含め、既存の結果へ
resume せず毎回新しい出力ディレクトリを作る。

```bash
cd <repo>

# 100件以上の dialogues.jsonl を持つ既存バッチ名に置き換える
export SOURCE_BATCH_ID=qwen_dialogues_10000_v1
test -s "data/runs/$SOURCE_BATCH_ID/llm_dialogues/dialogues.jsonl"
wc -l "data/runs/$SOURCE_BATCH_ID/llm_dialogues/dialogues.jsonl"

# 例: qwen_tts_vllm_smoke_100_20260723_153045 のような新規出力を作る
export BATCH_ID="qwen_tts_vllm_smoke_100_$(date +%Y%m%d_%H%M%S)"
export NUM_DIALOGUES=100
export FRESH=1
export PROXY_URL=http://<proxy-host>:<port>
JOB_ID="$(qsub -V scripts/run_qwen_tts_vllm_10000_4gpu.pbs)"
echo "submitted: $JOB_ID  batch: $BATCH_ID"
```

完了後、`merged_rows: 100` が出れば成功である。ジョブ全体のログは
`experiments/pbs_logs/${BATCH_ID}_*.log`、GPUごとの詳細は
`data/runs/${BATCH_ID}/logs/shard_*.log` に残る。

```bash
grep -E 'merged_rows|merged_manifest|finished_at' \
  "experiments/pbs_logs/${BATCH_ID}_"*.log
```

速度はGPUごとの `shard_*.log` にvLLM batch単位の `audio_x` / `rtf` と
シャード合計を、ジョブ全体ログに4 GPU合計の `performance:` を出す。

```bash
grep -E 'batch complete|performance summary|^performance:' \
  data/runs/${BATCH_ID}/logs/shard_*.log \
  "experiments/pbs_logs/${BATCH_ID}_"*.log
```

出力は `data/runs/<printed BATCH_ID>/`。ジョブログの `out_root` をマージ・学習に使う。
`BATCH_ID` の既定は smoke/1000 ジョブは**タイムスタンプ付き**（毎回新規 run）、
3000/10000 ジョブは**固定名**（再投入すると同じ dir に resume し、完了済みシャードは
スキップ）。3000/10000 で**旧データを残したまま作り直す**場合は `FRESH=1` を付けて
新しいタイムスタンプ付き run に出力する。walltime 内に終わらない場合は同じ
`BATCH_ID` で再投入すれば途中から再開する（チェーン投入例は各 .pbs のヘッダ参照）。

#### CTC alignment 失敗と対話数の担保

長い連結発話は TTS の生成長上限で音声が実際のテキストより短くなることが
あり、その場合 MMS_FA の forced alignment が
`target_length is too long for CTC` で失敗することがある。対策として:

- `generate_qwen3_tts_data.py` は話者ごとの連結テキストを
  `--whole-utterance-max-chars`（既定 150 文字）ごとに分割して合成・alignment
  するため、1回のTTS呼び出しが長くなりすぎて音声が打ち切られる事態を避ける。
  それでも alignment に失敗した対話は**既定で破棄**される（`--fa-fallback skip`。
  比例配分の推定境界で切った破損音声を学習データに混ぜないため。旧挙動は
  `--fa-fallback proportional` でデバッグ用に残る）。破棄分は `--success-target`
  併用で予備対話から自動補充され、末尾ログに FA 破棄件数が集計される。
  詳細は [knowledge/decisions/0003](knowledge/decisions/0003_word_level_alignments.md)。
- 対話単位の合成・書き出しは例外を捕捉するようになり、1件が失敗しても
  シャード全体は止まらず次の対話へ進む（失敗件数はログの
  `対話合成完了: 成功 N 件, 失敗 N 件` に出る）。
- 4GPU本番スクリプトはシャードごとに `SPARE_RATIO`（既定 0.15 = 15%）分の
  予備対話を追加で割り当て、`--success-target` で本来の割当数に達した時点で
  打ち切る。失敗が出ても予備対話で埋め合わせるため、`NUM_DIALOGUES` が
  そのまま維持されやすい。予備込みでも目標に届かない場合はそのシャードの
  プロセスがエラー終了するので、`shard_*.log` で原因を確認する。

#### シャード一部失敗時のマージ

上記の対策後もシャードが完全に停止した場合は、成功した対話分だけで
マージできる。

```bash
BATCH_ID=<printed_BATCH_ID>
uv run python scripts/merge_training_shards.py \
  --batch-dir data/runs/$BATCH_ID \
  --out-dir data/runs/$BATCH_ID/merged \
  --expected-shards 4 \
  --allow-partial
```

`merge_summary.json` に各シャードの件数と `missing_wavs` 数が記録される。

#### マージ後の学習

```bash
# LoRA (h01 パターン)
SRC_RUN_DIR=./data/runs/$BATCH_ID/merged \
SWEEP_PATTERNS=h01 qsub -V scripts/sweep_lora.pbs

# Full FT (f01 パターン)
SRC_RUN_DIR=./data/runs/$BATCH_ID/merged \
SWEEP_PATTERNS=f01 qsub -V scripts/fullft_sweep.pbs
```

### 実験やり直しランブック（FA修正後の再生成 → 30h/100h 比較）

alignments 単語化 + FA失敗破棄（`e540182`,
[decision 0003](knowledge/decisions/0003_word_level_alignments.md)）を反映した
データを作り直し、スケール比較をやり直す手順。**対話テキスト（dialogues.jsonl）は
FAバグと無関係なので再利用**し、TTS 段だけ再レンダリングする。

```bash
# 0. 対話テキストを再生成する（相槌AIプロンプト改良後の推奨。既存流用なら
#    「なるほど」を grep して v1.2 世代であることを確認してから使う）
qsub -V scripts/run_dialogues_qwen_1000.pbs
qsub -V scripts/run_dialogues_qwen_3000.pbs
#    出力は固定名 data/runs/qwen_dialogues_{1000,3000}_v1/（DIALOGUES_VERSION 既定 v1）。
#    後段の TTS ジョブは同じ名前を既定で読むため、受け渡しの指定は不要。
#    完成済み v1 が既にある場合はジョブが冒頭でエラー停止する（上書き防止）。
#    次の世代を作るときは両ジョブに DIALOGUES_VERSION=v2 を渡す。

# 1. 30h相当（1000対話）を再レンダリング。BATCH_ID は毎回タイムスタンプ付きなので
#    旧データと衝突しない
qsub -V scripts/run_qwen_tts_whole_utterance_1000_4gpu.pbs

# 2. ログ末尾で merged_manifest と FA破棄件数を確認。
#    破棄率が 10% を超えるようなら学習に進む前に原因を調べる

# 3. LoRA h01 を1本（テキスト崩壊が直ったかの検証）
SRC_RUN_DIR=./data/runs/<BATCH_ID>/merged \
SWEEP_PATTERNS=h01 qsub -V scripts/sweep_lora.pbs

# 4. 判定: eval loss 曲線 / merge_lora.pbs でマージ後の推論テキストが文として
#    成立しているか / analyze_backchannels.py の相槌分類。
#    旧 h01（壊れたデータ）と比較し、直っていなければ 100h に進まない

# 5. 100h相当（3000対話）。BATCH_ID が固定名なので FRESH=1 必須
#    （付けないと旧データの dir に resume され「完了済み」スキップになる）
FRESH=1 qsub -V scripts/run_qwen_tts_whole_utterance_3000_4gpu.pbs

# 6. 同じ h01 で学習 → 「きれいな30h vs きれいな100h」で初めてスケール効果を
#    正しく比較できる（旧30h vs 旧100h はノイズ床で頭打ちのため比較として無効）
SRC_RUN_DIR=./data/runs/<BATCH_ID_3000>/merged \
SWEEP_PATTERNS=h01 qsub -V scripts/sweep_lora.pbs
```

結果（テキスト崩壊の解消・相槌/自然性の変化）は
[knowledge/decisions/0003](knowledge/decisions/0003_word_level_alignments.md) の
TODO に追記する。

### 個別ステップ

```bash
# 1. use case 生成
uv run python scripts/build_use_cases.py --out-path data/v1/use_cases.jsonl --num 100

# 2. Gemma で対話生成
uv run python scripts/generate_synthetic_moshi_training_data.py \
  --out-dir data/v1/gemma_dialogues \
  --use-cases-jsonl data/v1/use_cases.jsonl \
  --num-dialogues 100 --mode dialogues-only \
  --gemma-backend transformers-subprocess --allow-template-fallback

# 3. Qwen3-TTS で音声合成
uv run python scripts/generate_qwen3_tts_data.py \
  --out-dir data/v1/training_set \
  --dialogues-jsonl data/v1/gemma_dialogues/dialogues.jsonl
```

### Full-duplex 学習データ

固定の評価14ケースとは別に、Gemmaで内容を生成し、Qwen3-TTSで実際の
同時発話を左右チャンネルへ配置する。

```bash
NUM_CASES=140 RUN_ID=full_duplex_v1 \
bash scripts/run_full_duplex_training_data.sh

# PBS / V100
qsub scripts/run_full_duplex_training_data.pbs
```

詳細は [docs/full_duplex_training_data.md](docs/full_duplex_training_data.md) 参照。

#### 大規模 (~100h) 生成 (8x V100)

10 シャードの PBS 配列ジョブ。各シャードが use_cases → dialogues → enrich →
audio を 1 枚の V100 で自己完結して実行する（途中失敗時はそのインデックスだけ
再投入すればよい）。傾聴70%/タスク30%、自然な相づち・話速・応答間、感情の
平滑化を含む。詳細は [docs/full_duplex_training_data.md](docs/full_duplex_training_data.md)。

```bash
# 0. 話の種バンクを一度だけ生成（内容の多様化用。省略可だが推奨）
qsub scripts/build_content_seeds.pbs
#    -> data/content_seeds/seeds.jsonl

# 1. パイロット 1 シャード（実スループットと平均対話長を必ず先に確認）
qsub -J 0-0 scripts/run_full_duplex_training_data_100h.pbs

# 2. 本実行（10 シャード。8 枚なら 8 並列でスケジューラが順次消化）
qsub scripts/run_full_duplex_training_data_100h.pbs
#    スケジューラが同時実行数の上限指定に対応していれば:
#    qsub -J 0-9%8 scripts/run_full_duplex_training_data_100h.pbs
#    もっと欲しい場合は配列を拡大: -J 0-95 で ~1000h

# 3. 全シャード完了後、1 つの manifest へ統合（CPU のみ・ログインノードで可）
BATCH_ID=<printed_BATCH_ID>
uv run python scripts/merge_training_shards.py \
  --batch-dir data/runs/$BATCH_ID \
  --out-dir   data/runs/$BATCH_ID/merged \
  --expected-shards 10
#    -> data/runs/$BATCH_ID/merged/training_set/synthetic_moshi_train.jsonl

# 4. 統合データセットで学習
qsub -v SRC_RUN_DIR=data/runs/$BATCH_ID/merged scripts/fullft_sweep.pbs
```

主な調整用環境変数（`qsub -v KEY=VAL,...` で渡す）:

| 変数 | 既定 | 説明 |
|---|---|---|
| `BATCH_ID` | auto (`fd_100h_<jobid>`) | output under `data/runs/<BATCH_ID>/shard_*` |
| `DIALOGUES_PER_SHARD` | `250` | 1 シャードの対話数（下げると失敗時の損失減・Gemma再ロード増） |
| `LISTENING_RATIO` | `0.7` | 自由対話（雑談・傾聴）の割合 |
| `GAP_SEC` | `0.2` | ターン交替の間（応答速度） |
| `CONTENT_SEEDS` | `data/content_seeds/seeds.jsonl` | 話の種バンク（無ければ自動でスキップ） |
| `MIN_SHARD_FRACTION_PCT` | `80` | シャード成功とみなす最小生成率 |

総量は配列サイズ × `DIALOGUES_PER_SHARD`。100h からずらすには `-J 0-N` か
`DIALOGUES_PER_SHARD` を変更する。

## 実対話データ（1chロールプレイ）の分析とテストデータ作成

1ch のロールプレイ録音（応答者/利用者の2話者）を話者タイムライン化し、
**耳で確認できる WAV 群**と統計を出す。ダイアライゼーションは
`nvidia/diar_streaming_sortformer_4spk-v2`（NeMo・非ゲート・CC-BY-4.0、
HF の同意/トークン不要）、書き起こしは faster-whisper。

```bash
# 0. テスト分割の凍結（データに触る前に一度だけ）
#    統計・相槌バンク・学習に使う前に、セッション単位で test 分を切り出して封印する。
#    目安: test 15–20分 / 開発用 残り。分割リストを git 管理して固定し、
#    test セッションは評価（実データ参照値・聴取評価アンカー）専用にする。

# 1. 分析を実行（PBS）
qsub -v WAVS=/path/to/session1.wav scripts/run_real_dialogue_analysis.pbs
# 複数ファイルやglob:
#   qsub -v 'WAVS=/path/a.wav,/path/b.wav' scripts/run_real_dialogue_analysis.pbs
#   qsub -v 'WAVS=/path/sessions/*.wav'    scripts/run_real_dialogue_analysis.pbs
# 手元やログインノードで直接も可（CPUの場合 whisper は自動で small になる）:
#   uv run python scripts/analyze_real_dialogue.py <wav> --out-dir data/real_dialogue/pilot
# 書き起こしを待たず切り出しだけ素早く聴くなら EXTRA_ARGS="--skip-asr"
```

出力（入力 WAV ごとに `<OUT_DIR>/<wav名>/`）:

| ファイル | 用途 |
|---|---|
| `speaker_A_solo.wav` / `speaker_B_solo.wav` | 相手区間を無音化。**相手の声が（重畳区間以外で）混ざっていなければダイアライゼーション合格** |
| `stereo_diarized.wav` | L=話者A / R=話者B（重畳区間は分離前なので両chに残る） |
| `segments/A/`, `segments/B/` | 発話ごとの WAV。ファイル名 = `通番_時刻_長さ[_ov][_aizuchi]_テキスト先頭` |
| `stats.json` / `timeline.jsonl` | 相槌頻度(/分)・語彙別内訳・応答gap分布・重畳長分布 |

話者は総発話時間の多い順に A / B。Sortformer が3話者以上を誤検出した場合は
上位2名以外の断片を警告付きで自動破棄する（`--num-speakers` で変更可）。

用途:

- **テストデータ**: 封印したセッションを Full-Duplex 評価の実データ参照値・
  聴取評価のアンカーに使う（学習・統計校正・バンクには使わない）
- **統計校正**: `stats.json` の実測値で `enrich_dialogue_timing.py` の相槌頻度
  （`SEC_PER_AIZUCHI` / `AIZUCHI_PROB`）や `GAP_SEC` を勘の値から置き換える
- **相槌バンク素材**: `segments/` の `_aizuchi` かつ `_ov` なし（非重畳）クリップ
- **学習混合（ステージ2・未着手）**: 重畳区間の音源分離による完全 2ch 化。
  ダイアライゼーション品質を耳で確認してから着手する

## 実験一覧

`experiments/` 配下は sweep の**ベース設定テンプレート**を置く場所。実際の
sweep結果は `experiments/_sweeps/` / `experiments/_fullft_sweeps/`（git ignore）
に生成される。詳細は [experiments/README.md](experiments/README.md)。

| 実験 | データ量 | 狙い |
|---|---|---|
| `lora_base_config` | ~100 対話 (~1h) | LoRA rank=32 ベースライン。`sweep_lora.pbs`/`lora_lr_sweep.pbs` の `BASE_EXP` |
| `fullft_base_config` | ~100 対話 (~1h) | フル fine-tuning で LoRA との比較。`fullft_sweep.pbs` の `BASE_EXP` |

## 学習済みモデルの利用

### LoRA チェックポイントのマージ

LoRA 実験のチェックポイントをベースモデルにマージして、推論に使える単一ファイルを生成する。

```bash
# マージ
uv run --project ../moshi-finetune python scripts/merge_lora.py \
  --lora-ckpt experiments/lora_base_config/checkpoints/<ts>/checkpoint_000500/consolidated/lora.safetensors \
  --out merged_model/consolidated.safetensors

# PBS
qsub -v \
LORA_CKPT=/path/to/checkpoint/consolidated/lora.safetensors,\
OUT_WEIGHT=/path/to/merged/consolidated.safetensors \
scripts/merge_lora.pbs

# マージ済みモデルで推論
uv run python response_recorder.py \
  --moshi-weight merged_model/consolidated.safetensors \
  --inputs prompts/hello.wav --out-dir results/lora_merged/
```

`--scaling` は checkpoint の `config.json` から自動取得される。

### フル FT チェックポイント

nu-dialogue runner はチェックポイントを `accelerator.save_state()` で保存するため、
`step_<N>/pytorch_model/zero_pp_rank_*.pt` という **DeepSpeed ZeRO シャード**で
出力される (`consolidated/...` のような単一ファイルは作られない)。推論に使うには
2 段階で単一 weight に変換する。

```
step_<N>/  (ZeRO shard)
   │ ① zero_to_fp32.py    … shard を統合 → fp32 単一ファイル (MoshiForFinetuning 形式)
   ▼
<ft_dir>/model.safetensors + moshi_lm_kwargs.json
   │ ② clean_moshi.py     … 元の LMModel 形式へ変換
   ▼
<clean_dir>/model.safetensors   (素の Moshi 形式 = response_recorder が読める)
```

チェックポイントの保存先は
`experiments/_fullft_sweeps/<RUN_ID>_<pattern>/checkpoints/nu_<ts>/step_<N>/`
(sweep 経由の場合)。`<NU_MODEL_DIR>` は sweep が初期化したモデル置き場
(デフォルト `../moshi-finetune-nu-dialogue/init_models/llm-jp-moshi-v1-both_streams-float32`)。

#### 一発エクスポート (推奨)

①② をまとめて実行する。`--remove_modules_for_user_stream` の要否は
checkpoint の `moshi_lm_kwargs.json` (`dep_q`) から自動判定する。

```bash
uv run python scripts/export_fullft_checkpoint.py \
  --step-dir <run>/checkpoints/nu_<ts>/step_120 \
  --out-dir  <run>/exported/step_120_clean

# PBS
qsub -v \
STEP_DIR=/path/to/checkpoints/nu_<ts>/step_120,\
OUT_DIR=/path/to/exported/step_120_clean \
scripts/export_fullft_checkpoint.pbs

# 推論
uv run python response_recorder.py \
  --moshi-weight <run>/exported/step_120_clean/model.safetensors \
  --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/f01_step120/
```

`--nu-repo` / `--moshi-lm-kwargs` で場所を上書き可。自動判定を上書きするなら
`--remove-user-stream` / `--no-remove-user-stream`。中間生成物を残すなら
`--keep-intermediate`。

#### 手動で 2 段実行する場合

`clean_moshi.py` は repo root の `models` パッケージを import するため、
`tools/` から直接実行すると `ModuleNotFoundError: No module named 'models'`
になる。`PYTHONPATH=.` を付けて nu repo root を通すこと。

```bash
# ① ZeRO シャード -> 単一 fp32 weight
#    第1引数は step_<N> (pytorch_model/ の親)。--tag pytorch_model を明示する。
#    出力先は事前に作らないこと (makedirs が exist_ok 無しのため既存だとエラー)。
cd ../moshi-finetune-nu-dialogue
uv run python tools/zero_to_fp32.py \
  <run>/checkpoints/nu_<ts>/step_120 \
  <run>/exported/step_120_ft \
  --tag pytorch_model \
  --safe_serialization \
  --moshi_lm_kwargs_path <NU_MODEL_DIR>/moshi_lm_kwargs.json

# ② 素の Moshi 形式へ変換
#    学習が --model_user_stream (full-FT sweep のデフォルト) の場合は
#    --remove_modules_for_user_stream を付けて dep_q=16 -> 8 に戻す。
PYTHONPATH=. uv run python tools/clean_moshi.py \
  --moshi_ft_dir <run>/exported/step_120_ft \
  --save_dir     <run>/exported/step_120_clean \
  --model_dtype bfloat16 \
  --remove_modules_for_user_stream

# ③ 推論
cd ../miltoka
uv run python response_recorder.py \
  --moshi-weight <run>/exported/step_120_clean/model.safetensors \
  --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/f01_step120/
```

どの `step_<N>` を選ぶかは、`run_nu_<ts>.log` または MLflow の `eval.loss` が
最小になった step を使う (full-FT baseline は 1 epoch ごとに checkpoint を保存)。

手早く生成だけ確認するなら、① の出力を nu repo の `generate.py` が
`MoshiForFinetuning.from_pretrained` で直接読めるので ② を省略できる
(ただし response_recorder の before/after 比較とは別形式)。

## Response Recorder

Moshi に固定音声を入力して応答を録音する実験ツール。
ドメイン適応の前後比較に使う。

```bash
uv run python response_recorder.py \
  --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/

# テキストプロンプトも可
echo "こんにちは" | uv run python response_recorder.py --out-dir results/stdin/
```

`--hf-repo` でモデル指定、`--silence-sec` で応答待ち時間を調整。
詳細は `python response_recorder.py --help` を参照。

## ディレクトリ構成

```
scripts/
  run_pipeline.sh                  # E2E パイプライン
  run_experiment.sh                # 実験単体起動
  generate_and_run.sh              # データ生成 + 実験起動
  build_use_cases.py               # use case 生成
  generate_synthetic_moshi_training_data.py  # Gemma 対話生成
  generate_qwen3_tts_data.py       # Qwen3-TTS 音声合成
  merge_lora.py                    # LoRA adapter をベースモデルにマージ
experiments/
  lora_base_config/                # LoRA sweep のベース config.yaml + HYPERPARAMS.md
  fullft_base_config/               # full-FT sweep のベース config.yaml + HYPERPARAMS.md
configs/
  moshi_lora_jp_loneliness.yaml    # パイプライン用 FT config
data/runs/                         # 実行ごとの出力 (git ignore)
```

## Full-FT / LoRA repository split

- Full fine-tuning: `nu-dialogue/moshi-finetune`, default checkout
  `../moshi-finetune-nu-dialogue`.
- LoRA fine-tuning: `kyutai-labs/moshi-finetune`, default checkout
  `../moshi-finetune`.

`scripts/run_fullft_sweep_pair.sh` uses the nu-dialogue repo and will clone it
if `../moshi-finetune-nu-dialogue` is missing. `scripts/run_sweep_pair.sh` and
`scripts/run_experiment.sh` keep using the Kyutai repo for LoRA.

Existing generated data can be reused by pointing `SRC_RUN_DIR` at the run
directory that contains `training_set/synthetic_moshi_train.jsonl`:

```bash
SRC_RUN_DIR=/path/to/data/runs/3h_dataset \
SWEEP_PATTERNS="f01" \
NPROC=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_fullft_sweep_pair.sh
```

PBS full-FT tuning uses `scripts/fullft_sweep.pbs` with `res=middle`,
`NPROC=2`, and `CUDA_VISIBLE_DEVICES=0,1` by default. LoRA tuning uses
`scripts/sweep_lora.pbs` with `res=small`. Details are in
`experiments/fullft_3h_sweep_10_patterns.md`.

For full fine-tuning, MLflow receives machine-readable stdout metrics from the
nu-dialogue runner. The important curves are `train.loss`, `train.loss.text`,
`train.loss.audio`, `eval.loss`, `eval.loss.text`, `eval.loss.audio`,
`train.accuracy.*`, `eval.accuracy.*`, and `learning_rate.*`.

## MOSS-TTSD training-data backend

`scripts/generate_qwen3_tts_data.py` also supports
`--tts-backend moss-ttsd`. It uses MOSS-TTSD once per utterance with a single
`[S1]` speaker and a role-specific voice-cloning reference. Stereo assembly is
unchanged: left is `moshi` (応答者), right is `user` (利用者).

MOSS jobs use two environments because `qwen-tts`/Moshi require Transformers 4
while MOSS-TTSD-v1.0 requires Transformers 5. First, the shared project
environment generates Qwen3-TTS references under `$OUT_ROOT/moss_refs/`
(`refs.json` plus four WAVs): Serena for `moshi`, Ono_Anna for `user`, Dylan for
`other`, and Ryan for `background`. The shared transcript is
`こんにちは、今日はよろしくお願いします。ゆっくりお話しできればと思います。`.
Existing WAVs are reused.

The audio stage then runs `generate_qwen3_tts_data.py` with
`uv run --isolated --no-project`, Transformers 5.0.0, and the cluster's exact
cu121 torch/torchaudio versions. It consumes only the dialogue JSONL and
`moss_refs/refs.json`; `qwen-tts` is not installed or imported in that step.

The audio-only three-dialogue quick test requires no manual reference audio:

```bash
qsub -V scripts/run_moss_ttsd_quicktest.pbs
```

The 50-dialogue Gemma pilot also uses these defaults and runs on one V100 in
float16:

```bash
qsub -V scripts/run_moss_ttsd_pilot.pbs
```

Output is written to
`data/runs/<timestamped BATCH_ID>/training_set/{data_stereo,synthetic_moshi_train.jsonl}`.
Override `BATCH_ID` or `NUM_CASES` before submission if needed.

Run `uv sync` for the MOSS runtime dependencies. The cluster must also provide
FFmpeg for `torchcodec`. FlashAttention 2 is optional and must be installed
manually against the cluster's CUDA/PyTorch stack; the V100 pilot uses the
default attention implementation because FlashAttention 2 requires newer GPU
compute capability.

## MOSS-TTSD NATURALNESS AUDITION mode

The dialogue pilot sends each complete script to MOSS-TTSD as one native
multi-speaker request. MOSS decides turn-taking, overlap, and backchannel
timing, and writes one mono mixed WAV plus a JSON sidecar per dialogue:

```bash
qsub -V scripts/run_moss_ttsd_dialogue_pilot.pbs
```

Outputs are written under
`data/runs/<timestamped BATCH_ID>/dialogue_audio/`. This mode is for
listening-only naturalness auditions. Its mono mixes and model-selected timing
are not training data and do not replace the existing per-utterance stereo
pipeline.

## TTS smoke listening phase

Before large-scale data generation, submit one two-A100 taste-listening job:

```bash
qsub -V scripts/run_tts_comparison.pbs
```

The job generates shared Qwen3 reference clips once, then renders one smoke WAV
per backend:

- `cosyvoice3`: Apache-2.0 repository code and
  `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, shared-reference zero-shot voices for
  consultant/user taste checks.
- `qwen3`: Apache-2.0 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, fixed preset
  voices for agent/user taste checks.
- `kokoro`: Apache-2.0 library/model, fixed Japanese voices through
  `misaki[ja]`/`pyopenjtalk`. Some Japanese voices include attribution
  metadata; review `VOICES.md` for deployment.

This is intentionally not an emotion-text A/B test. Every backend renders the
same dialogue once, with no per-turn emotion instruction. Every WAV remains
training-compatible stereo: left is `moshi`, right is `user`.

Default output:

```text
data/runs/tts_comparison_<timestamp>[_jobid]/
  shared_refs/
    refs.json
    moshi_Serena.wav
    user_Ono_Anna.wav
    other_Dylan.wav
    background_Ryan.wav
  listening/
    INDEX.html
    INDEX.md
    <dialogue_id>/
      cosyvoice3.wav
      cosyvoice3.json
      qwen3.wav
      qwen3.json
      kokoro.wav
      kokoro.json
```

Download `listening/` and open `INDEX.html`, or use `INDEX.md`. Each JSON
sidecar records the turns, voice mode, sample rate, and duration.

Cluster-specific requirement: `scripts/run_tts_comparison.pbs` defaults to
`#PBS -q xvn_s` and `#PBS -l select=1:res=small`. Before submission, confirm
that this is still the site's small GPU select string; replace that directive if
the scheduler requires an explicit GPU count.
