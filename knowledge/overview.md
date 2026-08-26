# 研究全体像

状態: 下書き / 更新日: 2026-07-01

## 一言で
日本語の**分野C窓口**向けに、全二重音声対話モデル **Moshi**
(`llm-jp/llm-jp-moshi-v1`) をドメイン適応し、**インタラクティビティ（相槌・
ターンテイキング等の対話の掛け合い）**を GRPO で整えるパイプライン。合成データ
生成から fine-tune、評価まで一気通貫。

## リサーチクエスチョン（例。実態に合わせて更新）
- RQ1: 合成対話＋合成音声だけで、日本語の全二重対話にドメイン適応できるか。
- RQ2: GRPO で「相槌/割り込み/沈黙」のインタラクティビティを、破綻なく強化できるか。
- RQ3: end-to-end（Moshi）は cascade（ASR→LLM→TTS）や SpeechLLM に対しどの観点で優位か。

## パイプライン地図（実装対応）
```
1. use_cases.jsonl      軸の組合せで対話ケース生成    scripts/build_use_cases.py
       ↓
2. dialogues.jsonl      Gemma が感情・沈黙付き対話     scripts/generate_synthetic_moshi_training_data.py
       ↓                                              scripts/gemma_dialogue_worker.py
3. ステレオWAV+manifest Qwen3-TTS合成(L=moshi/R=user)  scripts/generate_qwen3_tts_data.py
       ↓  (whole-utterance: MMS_FA強制アライメント)
4a. LoRA fine-tune       kyutai moshi-finetune          scripts/run_experiment.sh
4b. Full fine-tune       nu-dialogue moshi-finetune     scripts/run_nu_fullft_experiment.sh
       ↓                 (DeepSpeed ZeRO-3, A100×2)     configs/deepspeed_zero3_*.json
5. GRPO 整合             インタラクティビティ強化        scripts/run_grpo.pbs
       ↓
6. 評価                  Full-Duplex-Bench-JA           eval/evaluate_full_duplex_ja.py
   ベースライン比較      cascade / SpeechLLM            eval/local_baseline_common.py
```

## 研究の柱と代表論文（詳細は [paper_map.md](paper_map.md)）
- 全二重対話基盤: Moshi, Full-Duplex-Bench, dGSLM
- 整合手法: GRPO(DeepSeekMath), PPO, RLHF(InstructGPT), DPO
- 合成データ: Qwen3-TTS/VALL-E系, MMS強制アライメント
- 学習基盤: LoRA, DeepSpeed ZeRO
- 比較系: Whisper(cascade ASR), Qwen2-Audio(SpeechLLM)
- 応用: 分野C支援 / 共感対話(EmpatheticDialogues)

## 関連する運用ドキュメント（how-to側）
- [docs/full_duplex_evaluation.md](../docs/full_duplex_evaluation.md)
- [docs/full_duplex_training_data.md](../docs/full_duplex_training_data.md)
- [docs/local_baselines.md](../docs/local_baselines.md)
- [docs/evaluation_plan.md](../docs/evaluation_plan.md)

> `knowledge/` は「なぜ・背景理論・振り返り」、`docs/` は「どう動かすか」を担当。
