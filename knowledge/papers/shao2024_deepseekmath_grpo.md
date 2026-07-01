# DeepSeekMath: Pushing the Limits of Mathematical Reasoning（Shao et al., 2024）

status: draft / updated: 2026-07-01
- arXiv: 2402.03300
- PDF: `references/03_grpo_rl_alignment/shao2024_deepseekmath_grpo.pdf`（git外）
- 役割: base（GRPOの初出。本ブランチ `grpo-interactivity-alignment` の根拠）

## 3行要約
- 数学推論に特化したLLMを、継続事前学習＋RLで強化した研究。
- RL段階で **GRPO (Group Relative Policy Optimization)** を提案。1プロンプトに
  対し複数サンプルを生成し、**グループ内の相対的な優劣**でアドバンテージを推定する。
- PPOで必要な**価値関数(critic)を廃し**、メモリ/計算を削減しつつ安定して整合。

## GRPOの押さえどころ
- サンプル群 {o_1..o_G} の報酬 {r_i} を取り、グループ内で正規化した
  `A_i = (r_i - mean(r)) / std(r)` を各トークンのアドバンテージに使う。
- 価値ネットワーク不要 → RLHF比でメモリ軽い（大規模モデルで効く）。
- KL正則化で参照ポリシーから離れすぎないようにする。

## 本repoとの繋がり
- 実装/利用箇所: [scripts/run_grpo.pbs](../../scripts/run_grpo.pbs)（GRPO学習ジョブ、`res=middle`/2GPU, walltime 48h）。
- 応用の狙い: 数学の正誤報酬の代わりに、**インタラクティビティ（相槌・ターン
  テイキング・沈黙の適切さ）を報酬**として全二重対話を整える想定。
- 差分/未確定: 報酬関数の定義（何を「良い掛け合い」とするか）が本研究固有の設計点。
  → [paper_map.md](../paper_map.md) の「音声対話へのRL適用」は論文未特定。

## 疑問・未消化・TODO
- [ ] 報酬設計: Full-Duplex-Bench-JAの指標を報酬に流用できるか／別途学習報酬が要るか。
- [ ] サンプル群Gのサイズと音声生成コストのトレードオフ。
- [ ] KL係数・参照ポリシー（SFT後モデル）の選び方。

## 関連
- 概念: [GRPO](../concepts/grpo.md)
- 論文: PPO(2017), DeepSeek-R1(2025), DPO(2023), InstructGPT(2022) — いずれも `paper_map.md`
