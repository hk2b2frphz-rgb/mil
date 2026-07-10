# 0003 — ICASSP 2027: 聞き返し(clarification)研究トラックの新設

状態: 確認済み / 更新日: 2026-07-10

## 決定

ICASSP 2027 (締切 2026-09-16) に向け、`ICASSP2027/` 配下に独立した実験
トラックを新設する。リサーチクエスチョンは「E2Eフルデュプレックス音声対話
モデルは、スロット該当区間が音響的に劣化したとき、不確実性に気づいて
聞き返せるか」。intent classification ではなく **slot filling** を対象に
選んだ。

## Why

- 空白領域: 聞き返し研究はカスケード型(ASR信頼度依存)のみ
  (arXiv:2605.25404 等)。E2Eフルデュプレックスの評価軸としては
  Full-Duplex-Bench v1/v1.5/v2 のいずれにも存在しない。
  ICASSP 2026 HumDial Challenge の直後で時流も良い。
- slot filling を選んだ理由: (1) スロット値は発話内の**局所区間**なので
  「その区間だけ劣化させる」操作が定義でき、聞き返しの必要性を物理的に
  制御できる。intent は発話全体に分散し局所劣化と相性が悪い。
  (2) 誤りコスト(誤った時刻で予約)が明確で hallucinated confirmation
  という危険な失敗様式を定義できる。(3) MASSIVE ja-JP にスロット注釈が
  完備。
- 既存基盤の再利用: TTS(Qwen3)+強制アライメント(MMS_FA)+Moshi
  ストリーミング推論(response_recorder)+LoRA sweep+FDB-JA をすべて
  流用し、新規実装は劣化DSP・閉ループドライバ・指標・データ生成に限定。

## How to apply

- 設計: `ICASSP2027/docs/design.md` / 先行研究: `docs/related_work.md`
- 実行: `ICASSP2027/README.md` の p0→p7 (全てPBS、V100/A100以下)
- ブランチ: `icassp2027-clarification`
