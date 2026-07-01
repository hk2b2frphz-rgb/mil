# 論文 ↔ 実装 対応表

この資料集の中核。各論文が **本repoのどこで実装/利用/参照されているか** を1行で示す。
PDFは `references/<cat>/<file>.pdf`（git外）。詳細メモがあれば `papers/<name>.md` へリンク。

凡例:
- **役割**: `base`=基盤採用 / `impl`=実装に直接対応 / `baseline`=比較用実装 / `background`=理論的背景（コード直結せず）
- ✅=実装済みで対応箇所あり / 📄=背景参照のみ

| 論文 (年) | 役割 | 本repoでの繋がり（コード/設定） | メモ |
|---|---|---|---|
| **Moshi** (2024) | base | ベースモデル `llm-jp/llm-jp-moshi-v1`。全パイプライン。学習: [run_nu_fullft_experiment.sh](../scripts/run_nu_fullft_experiment.sh), [run_experiment.sh](../scripts/run_experiment.sh) | 全二重・ストリーミング音声対話の土台 |
| **Full-Duplex-Bench** (2025) | impl ✅ | 評価: [eval/evaluate_full_duplex_ja.py](../eval/evaluate_full_duplex_ja.py), [eval/build_full_duplex_ja_dataset.py](../eval/build_full_duplex_ja_dataset.py), [docs/full_duplex_evaluation.md](../docs/full_duplex_evaluation.md), [run_full_duplex_eval.pbs](../scripts/run_full_duplex_eval.pbs) | 日本語版として追随（upstream commit 3e799c4） |
| **dGSLM** (2022) | background 📄 | 2話者音声対話（left=moshi, right=user の2ch設計）の先行研究 | ターンテイキング/オーバーラップの考え方 |
| **EnCodec** (2022) | background 📄 | Mimiコーデックの背景。音声トークン化 `tools.tokenize_audio`（nu repo） | 残差ベクトル量子化 |
| **Qwen2-Audio** (2024) | baseline ✅ | SpeechLLMベースライン [scripts/speechllm_worker.py](../scripts/speechllm_worker.py)（Qwen2-Audio-7B-Instruct） | cascadeと並ぶ比較系 |
| **SpeechGPT** (2023) | background 📄 | 音声↔テキストLLMの設計思想の参考 | — |
| **AudioLM** (2022) | background 📄 | 音声言語モデルの生成アプローチ | — |
| **DeepSeekMath / GRPO** (2024) | base ✅ | **GRPOの初出**。本ブランチの中核 [run_grpo.pbs](../scripts/run_grpo.pbs) | → [concepts/grpo.md](concepts/grpo.md), [papers/shao2024_deepseekmath_grpo.md](papers/shao2024_deepseekmath_grpo.md) |
| **DeepSeek-R1** (2025) | background 📄 | GRPOによるRL学習の大規模実践 | 報酬設計・安定化の参考 |
| **PPO** (2017) | background 📄 | GRPO/RLHFの基盤アルゴリズム | — |
| **InstructGPT / RLHF** (2022) | background 📄 | 選好整合の標準リファレンス | — |
| **DPO** (2023) | background 📄 | RLフリー代替。手法選択の比較材料 | GRPO採用理由の対比に使う |
| **VALL-E** (2023) | background 📄 | コーデックLM型TTSの代表。合成TTS(Qwen3-TTS/MOSS-TTSD)の背景 | [generate_qwen3_tts_data.py](../scripts/generate_qwen3_tts_data.py) |
| **Whisper** (2022) | baseline ✅ | cascade ASR = faster-whisper。[eval/local_baseline_common.py](../eval/local_baseline_common.py) `LocalASR` | pyproject に `faster-whisper` 追加済み |
| **MMS** (2023) | impl ✅ | whole-utteranceモードのMMS_FA強制アライメント。[generate_qwen3_tts_data.py](../scripts/generate_qwen3_tts_data.py) `ForcedAligner` | "target_length is too long for CTC" の対象 |
| **LoRA** (2021) | impl ✅ | LoRAドメイン適応 [run_experiment.sh](../scripts/run_experiment.sh) + kyutai moshi-finetune | full-FTと二本立て |
| **ZeRO** (2019) | impl ✅ | full-FTのDeepSpeed ZeRO-3/offload [configs/deepspeed_zero3_fp16_warmlr_act_ckpt.json](../configs/deepspeed_zero3_fp16_warmlr_act_ckpt.json) | A100×2 OOM対策の理論的背景 |
| **QLoRA** (2023) | background 📄 | 量子化+LoRA。将来のメモリ削減候補 | 現状未使用 |
| **Mixed Precision** (2017) | background 📄 | fp16学習の背景（deepspeed configの`fp16`） | — |
| **LLM-jp** (2024) | base 📄 | ベースモデルllm-jp-moshi-v1の出自 | 日本語LLMの基盤 |
| **EmpatheticDialogues** (2019) | background 📄 | 応用ドメイン（孤独・孤立支援）の共感対話の参考 | 対話生成([build_use_cases.py](../scripts/build_use_cases.py)等)の観点 |

## 未取得だが対応が強い（要選定）
| テーマ | 本repoでの繋がり | 状態 |
|---|---|---|
| 音声対話へのRL適用（ターンテイキング報酬） | GRPO報酬設計 [run_grpo.pbs](../scripts/run_grpo.pbs) | 論文未特定 |
| Backchannel/相槌の予測・生成 | `--auto-overlap-aizuchi`, aizuchi overlap ([generate_qwen3_tts_data.py](../scripts/generate_qwen3_tts_data.py)) | 論文未特定 |
| ターンテイキング計算モデル | 全二重の交替・割り込み評価 | 論文未特定 |
| Qwen3-TTS / MOSS-TTSD | 合成データTTSバックエンド | arXiv無・モデルカード |
| CTC (Graves 2006) | 強制アライメントの失敗条件 | arXiv外 |

> 「論文未特定」は指示があれば WebSearch で正確な文献を特定して `references/` に追加し、
> ここへ1行追記する。
