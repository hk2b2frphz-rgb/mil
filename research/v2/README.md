# v2 — GRPO interactivity alignment（追加FT）

現在の `scripts/grpo/` および関連ランチャーと同一内容（本フォルダは archive
用の複製）。データ生成（v0〜v1.2）とは独立した軸: v1.x で作った合成データで
LoRA/full-FT した Moshi チェックポイントに対して、さらに GRPO で
「間の取り方・相づち・割り込みへの反応」を強化する追加学習を行う。

Kyutai の Multi-Faceted Interactivity Alignment 論文 (arXiv 2606.11167) を
セグメントベース学習で再現する:

1. 軸ごとのセグメントデータセット(`D_pause`, `D_turn`, `D_bc`, `D_int`)を読み込む
2. 各エポックでセグメントをサンプルし、必要なら文脈音声を前置する
3. user側音声をMoshiに与えて G 個のロールアウトを生成
4. 生成音声にVADをかけ、軸ごとの報酬を計算
5. GRPOでポリシー更新（テキストトークンのみ）

## 構成

- `grpo/config.py`, `grpo/segment_extractor.py`, `grpo/rewards.py`,
  `grpo/llm_judge.py`, `grpo/train_grpo.py` — 学習ループ本体
- `run_extract_grpo_segments.pbs` — 軸ごとのセグメント抽出ジョブ
- `run_grpo.sh` / `run_grpo.pbs` — 学習起動スクリプト

## 実行について

`train_grpo.py` は `from scripts.grpo.config import ...` のように
絶対importで自分のパッケージを参照しているため、この archive フォルダに
置いたコピーをそのまま `research/v2/grpo/train_grpo.py` として実行することは
できない（コピーは差分参照用）。実際に動かす場合は現行の `scripts/grpo/` を使う:

```bash
accelerate launch --num_processes 2 --use_deepspeed \
    --deepspeed_config_file configs/deepspeed_grpo_zero3.json \
    scripts/grpo/train_grpo.py --config configs/grpo_loneliness.yaml
```
