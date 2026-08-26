# knowledge/ — 論文・実装・知識の資料集

このリポジトリの研究を**後から振り返れる**ようにするための、git管理されるナレッジ
ベース。論文（`references/` のPDFはgit外）と、この repo の実装コード、そして概念的な
知識を相互リンクで結ぶ。

> `references/`（PDF実体）は `.gitignore` 済みで共有しない。ここ `knowledge/` は
> **テキスト（.md）のみをgitで共有**する。PDFはローカルパスで参照するだけ。

## 構成

```
knowledge/
  README.md          ← このファイル（index / 使い方 / 更新ルール）
  overview.md        ← 研究全体像・リサーチクエスチョン・パイプライン地図
  paper_map.md       ★ 論文 ↔ 実装 の対応表（この資料集の中核）
  glossary.md        ← 用語集（aizuchi, whole-utterance, backchannel 等）
  papers/            ← 論文ごとの読書メモ（要点＋本repoとの繋がり）
    _TEMPLATE.md
    <著者年_短縮名>.md
  concepts/          ← 概念・知識の解説メモ（1概念1ファイル）
    _TEMPLATE.md
    <concept>.md
  implementation/    ← 実装からの逆引き（このモジュール ← どの論文/概念）
    _TEMPLATE.md
    <area>.md
  decisions/         ← 意思決定ログ（なぜその設定にしたか。ADR風）
    _TEMPLATE.md
    NNNN_<title>.md
```

### 4つのビューで同じ知識を多面的に引ける
1. **paper_map.md** — 論文起点。「この論文はどこで実装/使用されているか」
2. **papers/** — 論文ごとの深掘り。要点と、本repoでの使われ方・差分。
3. **implementation/** — コード起点の逆引き。「このモジュールの背景理論は何か」
4. **decisions/** — 時系列の判断記録。「なぜ lr=1e-5 にしたか」等（振り返りの主役）

## 更新ルール（軽量に保つ）
- **実装を変えて、その背景に論文/概念があるとき**は、`decisions/` に1本足すか
  既存メモにWhy追記。コミットと同じ粒度で書くと後で辿りやすい。
- 新しい論文を読んだら `papers/_TEMPLATE.md` をコピーして1ファイル。読了後、
  `paper_map.md` の表に1行追加（実装との対応 or 「背景のみ」を明記）。
- 概念の説明が2回目に必要になったら `concepts/` に切り出して両所からリンク。
- リンクは相対パスで（例: `[GRPO](concepts/grpo.md)`, `[run_grpo.pbs](../scripts/run_grpo.pbs)`）。
- **状態タグ**を各メモ冒頭に置く: `状態: 未着手 | 下書き | 確認済み`、`更新日: YYYY-MM-DD`。

## ナビゲーション
- まず [overview.md](overview.md) で全体像 → [paper_map.md](paper_map.md) で論文↔実装 →
  個別は `papers/` / `implementation/` / `decisions/` へ。
- 用語で迷ったら [glossary.md](glossary.md)。
