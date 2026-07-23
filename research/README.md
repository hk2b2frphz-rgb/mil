# research/ — 対話生成・TTS・追加FT のバージョン系譜

この配下は、分野C窓口向け合成データ生成パイプラインが辿ってきた
実装のスナップショットを、実際に使われた順に整理したものである。
`scripts/` 直下は現在アクティブな開発の場所であり続け（v1.2 相当の内容と一致）、
ここは「その時点で実際に動いていたコード」を後から参照・再現できるようにする
研究アーカイブ。各フォルダは自己完結しており、他フォルダへの依存はない
（同じファイルが複数フォルダに重複コピーされているのは意図的）。

## バージョン一覧

| バージョン | 内容 | 対応コミット/状態 |
|---|---|---|
| [v0](v0/) | 対話生成(単発LLM)＋ Kyutai/Moshi 直接TTSレンダリング | `b0b30b0`(Qwen3-TTS導入直前) |
| [v1.0](v1.0/) | マルチエージェント対話生成 ＋ Qwen3-TTS（ターンごと合成、whole-utterance以前） | `4234fdf`(whole-utterance導入直前) |
| [v1.1](v1.1/) | 同上 ＋ Qwen3-TTS whole-utterance合成（MMS_FA強制アライメント） | HEAD 直前コミット `2e07755` |
| [v1.2](v1.2/) | 固定冒頭挨拶の声色固定・感情ミラーリング・相づち再設計 | 現在の `scripts/` と同一内容 |
| [v2](v2/) | GRPO によるインタラクティビティ・アラインメント追加FT | 現在の `scripts/grpo/` と同一内容 |

サブバージョン `v1.1.0` / `v1.1.1` / `v1.1.2` はコードは同一で、
`run_qwen_tts_whole_utterance_{1000,3000,10000}_4gpu.pbs` の対話数(1000/3000/10000)
だけが異なる本番ジョブを指す。`v1.2` も同じジョブ構成を引き継いでいる。

## 系譜の詳細

```
v0   (単発LLM対話生成 + Kyutai/Moshi 直TTS)
  ↓  Qwen3-TTS 導入、マルチエージェント対話生成(userAI/moshiAI/judgeAI/aizuchiAI)へ移行
v1.0 (マルチエージェント対話生成 + Qwen3-TTS per-turn合成)
  ↓  話者ごとの発話を連結して1回のTTSで合成 + MMS_FA で境界復元（韻律の一貫性向上）
v1.1 (同上 + whole-utterance合成)  … v1.1.0=1000対話 / v1.1.1=3000対話 / v1.1.2=10000対話
  ↓  「なるほど」連発の解消、感情ミラーリング、冒頭挨拶の声色固定
v1.2 (今回の改善)
  ↓  データ生成とは独立軸: GRPO で人間らしい応答性（間・相づち・割り込み）を追加FT
v2   (GRPO interactivity alignment)
```

## 除外したもの

MOSS-TTSD バックエンド・TTS比較ハーネス(`tts_comparison_backends.py` 等)・
gpt-oss-120b 代替対話生成(`run_dialogues_gptoss_2a100.pbs`)は、本線の系譜には
含めていない。これらは README/docs に手順が残る現存オプションであり
「未採用」ではないため `scripts/` 直下に残置し、このアーカイブには複製していない。
一方、どこからも参照されていなかった `run_vllm_smoketest*.pbs` /
`vllm_smoke_infer.py` / `verify_moshi_backchannels.pbs` /
`run_dialogues_gptoss_1000.pbs` は孤立コードとして削除した。

`eval/`（Full-Duplex-Bench-JA 評価）、`knowledge/`（論文・決定ログ）、
および LoRA/full-FT・MLflow同期・PBSクラスタ設定などの学習/評価基盤は、
特定バージョンに紐づかない共通基盤として `scripts/`・`eval/`・`knowledge/`
直下に残したまま各バージョンから参照する。

## 使い方についての注意

各フォルダのコードは、その時点の `pyproject.toml`/依存関係を前提に動いていた
スナップショットである。現在の環境で v0/v1.0/v1.1 をそのまま再実行できる保証は
なく、主目的は「何がどう変わったかを読んで追える」ことである。実際に動かす場合は
現行の `scripts/`（v1.2 + v2 相当）を使うこと。
