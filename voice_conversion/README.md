# V100向け・参照音声による非ストリーミングVC

採用モデルは **OpenVoice V2（コード・公式重みともMIT）** です。2026-10-16の要件更新に合わせ、GPLのSeed-VC用スクリプトは取り下げました。ピッチ固定は行わず、元音声の発話内容・話し方をできるだけ保持する声質変換として利用します。

日本語は公式の対応言語に含まれます。ただし、日本語のVC品質で他モデルを上回ることを確認したわけではありません。日本語対応TTSの評価と、日本語音声を入力したVCの評価は区別してください。アクセント・間・話者類似度の実音声検証は未完了です。

Apache-2.0のCosyVoiceも候補ですが、最新TTSの日本語性能をそのままVC性能とは扱いません。今回の実行経路はOpenVoiceのconverterだけです。公式モデルのライセンスがMITであることと、依存ライブラリを含めて全てMIT/Apacheだけであることは別です。後者の網羅的な監査、および学習データの権利監査を完了したという意味ではありません。

OpenVoice V2のToneColorConverterだけを利用します。元音声の発話内容・話し方をできるだけ保ち、参照話者の声質に変換します。追加学習、文字起こし、テキスト入力、MeloTTSは不要です。元話者・目標話者の埋め込みをそれぞれ音声から抽出します。

## セットアップ（Linux / V100計算機）

前提: git、uv、NVIDIAドライバ（CUDA 11.8対応）。プロジェクトのルートから:

```bash
bash voice_conversion/setup.sh
```

独立したPython 3.9環境にPyTorch 2.1.2 / CUDA 11.8を入れます。V100はsm_70なので、起動時にGPUとPyTorch wheelの対応を検査します。FP32で実行し、BF16・FlashAttentionは使用しません。既存のプロジェクト環境は変更しません。初回はインターネット接続が必要です。必要なら計算機のプロキシを設定してから実行してください。

公式実装をvendorに取得し、公式Hugging Faceリポジトリからconverterの重みを取得します。取得したコードと重みのコミットをそれぞれopenvoice-revision.txt / model-revision.jsonに記録します。初回の取得対象はその時点のmainです。再現実験ではこれらのリビジョンを保管し、vendorとmodelsを固定して利用してください。

## 変換

```bash
voice_conversion/.venv/bin/python voice_conversion/convert.py \
  --source source.wav \
  --reference target_speaker.wav \
  --output voice_conversion/outputs/converted.wav
```

参照音声は単一話者・雑音やBGMのない5〜15秒程度をまず推奨します（運用上の目安）。同じ台詞を読む必要はありません。元音声が短い／雑音が多い場合、同じ元話者の別のきれいな録音を `--source-reference source_speaker.wav` で指定できます。モデルのサンプリングレートに自動変換し、モノラルWAVを出力します。

入力ファイル全体を読み込んで一括変換する非ストリーミング処理です。まず5〜15秒の元音声で確認してください。長時間音声はメモリ使用が増えるため、発話ごとに分けて実行してください。複数話者の元音声は話者ごとに分ける必要があります。

## 商用利用と選定理由

- 公式コード: https://github.com/myshell-ai/OpenVoice （MIT）
- 公式重み: https://huggingface.co/myshell-ai/OpenVoiceV2 （MIT、商用利用可の明記あり）
- 確認日: 2026-10-16。MITの著作権・ライセンス表示を再配布時に保持してください。vendor内のLICENSEも保持してください。
- 音声入力はローカル処理です。Whisper/VADの追加重みは取得せず、extract_seを直接呼び出します。補助watermarkモデルもロードしません。
- Seed-VCも候補ですが、この構成はMITで商用利用条件が明確なOpenVoice V2に絞っています。最新モデルとの品質比較を行った結果ではありません。

## 検証状況・制約

準備環境はWindowsで、V100・CUDA実行環境はありません。Python構文とCLIを確認する範囲に留まり、依存パッケージのインストール、重み取得、V100上の推論、音質・処理速度・最大VRAM使用量は未検証です。ネットワーク制限によりこの環境で公式コードと重みの取得は完了していません。

`setup.sh` 最後の `--check` は実機でGPU対応とモデルロードを検査します。実音声の変換品質を保証する検査ではありません。抑揚・ピッチ・発音・音質が完全に不変になる保証もありません。入力と出力を試聴して確認してください。
