# GRPO (Group Relative Policy Optimization)

状態: 下書き / 更新日: 2026-07-01

## 一言で
1プロンプトから複数出力をサンプリングし、**グループ内の相対報酬**でアドバンテージを
決めるRL整合手法。PPOの価値関数(critic)を不要にし、メモリと実装を軽くする。

## なぜ本プロジェクトで重要か
本ブランチ `grpo-interactivity-alignment` の中核手法。全二重対話の
**インタラクティビティ（相槌・割り込み・沈黙の適切さ）**を報酬にして、SFT後の
Moshiを破綻させずに整えるのに使う。criticを持たないぶん、音声モデルの重い学習で
メモリ的に有利。

## 仕組み / 要点
- グループ {o_1..o_G} を同一プロンプトから生成し報酬 {r_i} を得る。
- アドバンテージ `A_i = (r_i − mean(r)) / std(r)`（グループ内正規化）。
- 目的: PPO類似のクリップ付き方策勾配 ＋ 参照ポリシーへのKL正則化。
- critic無し → メモリ削減・不安定要因減。報酬設計の質が結果を左右する。

## 本repoでの現れ方
- 学習ジョブ: [scripts/run_grpo.pbs](../../scripts/run_grpo.pbs)（2GPU/`res=middle`, 48h）
- 報酬候補: Full-Duplex-Bench-JAの指標（[eval/evaluate_full_duplex_ja.py](../../eval/evaluate_full_duplex_ja.py)）の流用可否は要検討。

## 注意（本プロジェクト特有）
- 「良い掛け合い」の報酬定義が未確定＝設計の勘所。ハック（無言連発で安全に稼ぐ等）を
  避ける報酬整形が要る。
- サンプル群Gのサイズ×音声生成コストのトレードオフ。

## 出典・さらに読む
- 論文: [DeepSeekMath/GRPO](../papers/shao2024_deepseekmath_grpo.md)（arXiv:2402.03300）, PPO(2017), DeepSeek-R1(2025)
