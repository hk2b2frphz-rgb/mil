# Full-duplexへの「乗り方」：世界・日本の戦略地図

調査日: 2026-08-28  
対象: 公式発表、公式製品ドキュメント、公開論文。ここで扱うのは**基盤モデルのランキングではなく、各組織がfull-duplexを何のための能力として位置付けているか**である。

## 要旨

本レポートでは、**AIが応答を話している間にもユーザーの発話を受け取り、聞き続けて会話を制御できる枠組み**をFull-duplexと呼ぶ。各社は必ずしも専用モデルを公開する必要はない。実際には、次の五つの異なる乗り方がある。

| 乗り方 | 狙い | 主な組織 |
|---|---|---|
| **会話そのものを作り替える** | 同時聴取・同時発話をモデルの基本能力にする。 | Kyutai、Meta、ByteDance、NVIDIA、LLM-jp／名古屋大 |
| **大規模プロダクトのUIにする** | 音声・映像・端末を常時対話の入口にする。 | Google、Meta、ByteDance、Apple |
| **企業向け音声エージェント基盤にする** | 割込み、エコー除去、電話、ツール実行を部品化する。 | OpenAI、Amazon、Microsoft、Hume |
| **業務の並行処理にする** | 会話中に意図を先読みして業務処理を開始し、応対時間を減らす。 | NTT、AI inside、Hitachi、LINEヤフー、サイバーエージェント、富士通 |
| **日本語のデータ・評価・再現性を作る** | 日本語の相槌、間、電話音声、敬語を扱える研究・評価基盤を整える。 | LLM-jp／NII、名古屋大、NTT、各社研究組織 |

この定義では、ユーザーがAIの応答中に話せて、システムがその音声を受けて処理を切り替えるbarge-inもFull-duplexに含める。ただし、次の二層は区別して記す。

| 区分 | 意味 | 例 |
|---|---|---|
| **運用上のFull-duplex** | AIの応答中にも入力を聞き、割込み・停止・応答切替を行える。 | GPT-Realtime、Gemini Live、Nova Sonic、Azure Voice Live、Hume EVI |
| **狭義のnative full-duplex** | 入出力音声を同時にモデル化し、重なり、相槌、沈黙、タイミング自体を生成・制御することを明示する。 | GPT-Live、Moshi、Seeduplex、PersonaPlex、Meta Synchronous LLM |

前者も本レポートではFull-duplexである。後者はその中でも、実際の会話の重なりや短い相槌まで扱おうとする、より踏み込んだ実装である。

## 1. 世界の大手は、full-duplexを何に変えようとしているか

### OpenAI：音声エージェントを作るための共通インターフェース

OpenAIは、Full-duplexを開発者がリアルタイム音声エージェントへ組み込むためのAPI機能に分解している。Realtime APIはspeech-to-speech、server/semantic VAD、応答中断を提供し、GPT-Realtime系列はリアルタイムの音声入出力を持つ。[Realtime APIリファレンス](https://platform.openai.com/docs/api-reference/realtime?lang=javascript) [GPT-Realtime](https://developers.openai.com/api/docs/models/gpt-realtime)

ここは名称を分ける必要がある。

| 名称 | 位置付け | 本レポートでの扱い |
|---|---|---|
| **GPT-Live** | OpenAIのFull-duplex音声会話そのものを担う、利用者向けの会話体験・モデル。 | **OpenAIにおけるnative Full-duplexの中核**。モデル公開やAPI提供とは別に扱う。 |
| **GPT-Realtime** | 開発者が音声入出力、VAD、割込み、ツール実行を組み込むためのリアルタイム音声モデル。 | **運用上のFull-duplex**。アプリや電話エージェントの基盤。 |
| **Realtime API** | GPT-Realtime等をWebRTC／WebSocket／SIPで接続するセッション・通信面。 | モデルではなく、Full-duplex体験を実装する接続・制御面。 |
| **GPT-Live-Transcribe** | ライブ音声から低遅延の文字起こしを返す音声入力専用モデル。 | 音声を話し返さないため、単体ではFull-duplex音声対話ではない。 |

GPT-LiveとGPT-Realtimeを同一視してはいけない。前者は「聞きながら話せる」Full-duplex会話を利用者へ届ける側、後者はそれを含むリアルタイム音声アプリを開発者が構成する側である。したがって、**OpenAI内でFull-duplex体験そのものを担う唯一の位置はGPT-Live**として扱う。

なお、GPT-Realtimeの音声入出力・リアルタイム接続、GPT-Live-Transcribeの入力専用仕様は[OpenAI公式モデル情報](https://developers.openai.com/api/docs/models/all)で確認できる。一方、今回参照可能だった公式APIカタログにはGPT-Liveの公開モデルカード／API仕様は掲載されていないため、GPT-Liveの詳細な仕様・提供条件は公開一次ソースを受領後に追記する。

### Google：マルチモーダルな常時対話を、検索・端末・映像に埋め込む

GoogleはGemini 2.5のnative audioで、音声を直接推論・生成する低遅延対話、背景会話の無視、感情・多言語・映像入力を打ち出している。これはGemini Live（利用者向けの会話体験）、Gemini Live API（開発者向けのリアルタイムセッション）、Search Liveのような既存の巨大な接点へ載せる構図である。[Gemini 2.5 native audio](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-2-5-native-audio/) [Live APIの発表](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-updates-io-2025/)

Gemini Liveは、AIが話している最中にユーザーが話せる。Live APIの公式実装ガイドも、モデルが応答中にユーザーが発話すると `interrupted` が送られ、クライアント側の再生バッファを直ちに捨てるよう定めている。[Gemini Live APIの割込み処理](https://ai.google.dev/gemini-api/docs/live-api/best-practices)

よって、**Gemini Liveはこのレポートの運用上のFull-duplexに入る**。native audioを用いる点で音声統合は強いが、AI自身の発話を継続したままユーザーと重ねて話すことを標準挙動として公開保証しているわけではないため、狭義のnative full-duplexとは別に扱う。Googleの価値は、カメラ、画面共有、検索、ツールを共有したまま声で作業できることであり、**Full-duplexは検索窓やアプリ画面を置き換えるマルチモーダルUIの一部**として乗っている。

### Meta：人間同士に近い会話を、研究とソーシャル製品の両方で試す

Metaは研究で最も明確にfull-duplexを定義している。Synchronous LLMsは、時間感覚を持たないLLMに実時間の同期を与え、オーバーラップ、相槌、動的なターン交替を扱う研究である。[Synchronous LLMs](https://ai.meta.com/research/publications/beyond-turn-based-interfaces-synchronous-llms-as-full-duplex-dialogue-agents/)

同時にMeta AIアプリにはfull-duplex speech技術によるvoice demoを載せた。これはテキストの読み上げではなく、会話音声を直接生成する将来の体験として位置付けられている。[Meta AI app](https://about.fb.com/news/2025/04/introducing-meta-ai-app-new-way-access-ai-assistant/)

**Metaの乗り方は、会話の自然さをソーシャル接点、AIグラス、個人向けアシスタントの差別化にすること**である。研究の公開性と、数十億人規模の会話接点を同時に持つ点が強い。

### Amazon：電話・コンタクトセンターの「割込み可能な業務対話」

Amazon Nova Sonicは双方向イベントストリーミングのS2Sとして提供され、ユーザーが話し始めるとAIの応答を止め、文脈を保ったまま新しい入力を処理するbarge-inを明示している。[Nova Sonic](https://docs.aws.amazon.com/nova/latest/userguide/speech.html) [barge-in仕様](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-barge-in.html)

これは会話を重ねることより、業務対話で「待たせない・話を遮らない・古い音声を再生しない」ことを優先する設計である。**Amazonの乗り方は、full-duplex的な操作感を、AWS上で安全に運用できる企業用音声基盤へ落とすこと**である。

### Microsoft：音声品質・エコー・認証まで含む企業導入の部品

Azure Voice Live APIはWebRTC/WebSocket、semantic VAD、サーバー側エコーキャンセル、barge-in、Azure AIの各種モデル・アバターとの統合を提供する。[Azure Voice Live API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-how-to)

Microsoftの焦点は、モデルそのものよりも、企業が音声対話を導入した際に問題となるエコー、騒音、認証、電話・Webアプリ接続を管理可能にすることにある。**full-duplexはAzureの会話サービス群を自然にするための運用レイヤー**として進んでいる。

### Apple：対話能力より、端末・個人文脈・プライバシーの統合を優先

AppleはSiri AIで、端末をまたぐ会話履歴、個人文脈、より表情豊かな音声を発表している。[AppleのSiri AI発表](https://www.apple.com/newsroom/2026/06/apple-introduces-siri-ai-a-profoundly-more-capable-and-personal-assistant/)

しかし、今回確認した公式資料にはfull-duplexの同時聴取・同時発話を明示する説明はない。現時点のAppleは、会話の重なりよりもオンデバイス処理、個人データの保護、OS横断の文脈継続を優先していると見るのが妥当である。**この領域への不参加ではなく、公開上の差別化軸が別にある状態**である。

## 2. 中国勢は、消費者規模・オープンエコシステム・end-to-end化で分かれる

### ByteDance：full-duplexを大量利用されるプロダクトの基本動作へ

ByteDanceのSeeduplexは、話しながら聞くnative full-duplex、背景話者・雑音の抑制、意味情報も用いた終話判断を掲げ、Doubaoアプリへの全面展開を発表した。さらにSeedRealtimeでは、音声・映像・テキストを継続的に処理するaudio-visual full-duplexへ広げている。[Seeduplex](https://seed.bytedance.com/en/blog/introducing-seed-full-duplex-speech-llm-attentive-listening-robust-interference-suppression-enabling-more-natural-interaction) [SeedRealtime](https://seed.bytedance.com/en/blog/seedrealtime-audio-visual-full-duplex-llm-released-toward-omni-modal-natural-interaction)

**ByteDanceの乗り方は、full-duplexを「特別なデモ」ではなく、巨大な消費者アプリでの標準的な会話リズムにすること**である。公開された導入規模・性能値は同社発表であり独立検証とは分ける必要があるが、プロダクト実装まで一気通貫である点は際立つ。

### Alibaba：モデルを開き、端末・クラウド・開発者へ広げる

Qwen2.5-Omniは、音声・画像・映像を受け、リアルタイムにテキストと自然音声を返すend-to-endなomniモデルとして、オープンソースとクラウド双方へ展開された。[Qwen2.5-Omniの発表](https://www.alibabagroup.com/en-US/document-1843362291857227776) [実装リポジトリ](https://github.com/QwenLM/Qwen2.5-Omni)

ここでの戦略は、full-duplexを明示して体験を囲い込むことではなく、**音声・映像を扱えるモデルを開発者や端末向けに広く使わせ、エコシステムを作ること**である。ストリーミングS2Sはfull-duplexへの技術的な足場だが、公開資料だけでnative full-duplexの会話制御を主張するのは避けるべきである。

### StepFun、Baiduなど：音声基盤は出るが、full-duplexの立場はまだ読み切れない

StepFunのStep-Audio2は理解・認識・対話・音声生成を一体化したend-to-end音声会話モデルを公開している。[Step-Audio2](https://github.com/stepfun-ai/Step-Audio) ただし、full-duplexの会話制御を主戦略として掲げる公開資料は確認できない。

Baiduなども音声・マルチモーダルの投資先ではあるが、今回確認できた一次情報では、ByteDanceほど明示的なfull-duplexの研究・製品方針は特定できなかった。これは技術がないという意味ではなく、**このパラダイムを対外的な主戦略にしているかは公開情報からは判定できない**という意味である。

## 3. Kyutai以降のオープン勢は、会話の「制御可能性」を広げる

### Kyutai：会話の時間構造をオープンな研究対象にした

KyutaiのMoshiは、full-duplex音声対話を公開研究・実装の土台にした。Moshiの複数ストリームの考え方は、Hibiki-Zero、Unmute、MoshiRAGにも展開されている。[Kyutai](https://kyutai.org/) [Moshi論文](https://arxiv.org/abs/2410.00037)

Kyutaiの本質的な貢献は、音声対話を「ASR→LLM→TTSを速くつなぐ問題」だけではなく、**二者の音声の時間構造を学習・評価する問題**にしたことである。これが日本語化、NVIDIAの派生、full-duplex評価研究の出発点になっている。

### NVIDIA：自然な会話を、役割・声・顧客対応へ使える形にする

PersonaPlexはMoshi系のfull-duplexを維持しつつ、テキストによる役割指定と音声による声質条件付けを加え、コード・重みを公開している。顧客対応などでのタスク遵守と会話タイミングの両立を狙う。[PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/)

**NVIDIAの乗り方は、会話の自然さを「調整可能で導入できるエージェント」に変換すること**である。これは基盤モデルの発表競争よりも、企業が使えるペルソナ、評価、配布形態を整える動きといえる。

### HumeとSesame：会話の自然さを別の入口から攻める

Hume EVIは感情・韻律を理解して応答するS2S APIであり、常時割込み可能な音声エージェントとして提供される。[Hume EVI](https://dev.hume.ai/docs/speech-to-speech-evi/overview) 重点は共感・感情表現である。

SesameはConversational Speech Modelで会話的な韻律を作る一方、同社自身が会話構造はまだ扱えず、最終目標はturn-taking、pause、pacingを学ぶfull-duplexだと説明している。[Sesameの技術説明](https://sesameaivoice.com/technology)

両社は、**音声が人間らしいことと、会話の時間構造が人間らしいことは別問題**だと示す。前者を先に商品化しても、後者は次の大きな技術課題として残る。

## 4. 日本は「誰がどのレイヤーを担うか」で見る

### LLM-jp／NII・名古屋大学：日本語full-duplexを検証可能にする

LLM-jpはMoshiを日本語対話データで追加学習したLLM-jp-Moshi-v1を公開している。[LLM-jp-Moshi](https://llm-jp.github.io/llm-jp-moshi/) 名古屋大学のJ-Moshiは、日本語で相手の話を聞きながら話すfull-duplexを目標に、J-CHAT、CSJ、CallHome-Japanese等を活用した。[J-Moshi](https://www.nagoya-u.ac.jp/researchinfo/result/2025/02/-ai-j-moshi.html)

この両者の役割は、巨大消費者サービスをただちに作ることではない。**日本語で何を相槌、割込み、沈黙、自然な重なりとみなすかを、データ・モデル・評価として再現可能にすること**である。研究機関・企業が共通に使える土台という意味で、長期的に重要である。

### NTT：通信とコールセンタにおける自然な応対へ

NTTは、人のように素早く応答し、相槌を打ち、自然な抑揚で話すfull-duplex型音声対話AIを研究し、2025年のR&Dフォーラムでコールセンタ自動応対を想定した展示を行った。[NTT技術ジャーナル](https://journal.ntt.co.jp/article/39223)

NTTの乗り方は、会話モデルそのものの公開よりも、通信品質、音響、電話、コンタクトセンタという現場に成立させることである。**日本で最も実用展開に近い「会話リズム × 通信サービス」の担い手**と考えられる。

### AI inside：話している間に業務を始める

AI insideは、対話と業務実行を同時処理する日本語full-duplex音声対話モデルをGENIAC成果として発表した。発話途中から意図を捉えて処理を始める点を売りにし、業務完了時間96%短縮を同社実証値として示している。[AI insideの発表](https://inside.ai/news/2026/0408_full-duplex)

ここでfull-duplexは、会話を人間らしく見せるためだけの技術ではない。**業務の待ち時間をなくす実行制御**として使われている。予約・事務・問い合わせでは、この価値のほうが雑談の自然さより直接的に事業成果へつながる。

### Hitachi：現場・コンタクトセンタのマルチモーダルAIの一部

日立は研究領域として、Full Duplex音声対話、Speech-to-Speech、音声／音響基盤モデルを明記し、フロントワーカーとコンタクトセンタ支援を応用先に置く。[日立の研究領域](https://www.hitachi.co.jp/rd/careers/lab/ai/03.html)

日立は公開モデルの存在よりも、現場の騒音、複数話者、業務知識、信頼性を含むマルチモーダルAIの一機能としてfull-duplexを位置付けていると読める。今回の調査では詳細なモデル仕様・製品発表は確認できなかったが、**現場業務の信頼性と結び付ける方向性**は明確である。

### 富士通：ターンテイキングの蓄積を持つが、生成AI full-duplexの公開方針は未確認

富士通はサービスロボットで、発話の切れ目の検出、相槌、話者交替時の無音を減らす対話タイミング制御を以前から研究している。[富士通の対話タイミング制御](https://www.fujitsu.com/jp/documents/about/resources/publications/magazine/backnumber/vol68-5/paper07.pdf)

これはfull-duplexの前提となる人間中心の会話設計である。ただし、今回確認した公開資料では、生成AIによるnative full-duplexモデルや大規模な導入方針を特定できなかった。富士通は現状、**会話の間・ロボット・業務システムの知見を持つが、full-duplexを対外的な主戦略としてはまだ打ち出していない**と整理するのが正確である。

### LINEヤフー：サービス接点へ載せるための低遅延S2S

LINEヤフーのSpeech and Acoustic AI部は、ASR/TTSとSpeech LLMによるS2Sの事前検証、リアルタイム音声対話の技術検証を公開している。[LINEヤフーのTech-Verse報告](https://techblog.lycorp.co.jp/ja/20250903a)

同社の強みは、多言語コミュニケーション、検索、予約などユーザー接点が広いことにある。公開資料はnative full-duplexを主張しないが、**既存の生活者向けサービスへ音声エージェントを滑らかに組み込むための低遅延S2S**という方向にある。full-duplexは、その次に積み上がる会話制御レイヤーと考えられる。

### サイバーエージェント：声の自然さ、応対品質、実サービスの最適化

サイバーエージェントは完全自動対話研究センターを設け、Voicebot、キャラクター、コールセンタでの音声対話を研究・導入する。LLM音声対話の高速化、対話品質の評価にも取り組む。[完全自動対話研究センター](https://www.cyberagent.co.jp/news/detail/id%3D26776) [音声対話の高速化](https://developers.cyberagent.co.jp/blog/archives/44592/)

同社はnative full-duplexを公開主張してはいない。しかし、音声対話を広告・コンテンツ・接客のUXへ展開し、評価と高速化まで自前で持つ。**サイバーエージェントの乗り方は、full-duplexを将来の音声エージェント品質の延長として、実データと実サービスで磨くこと**にある。

## 5. この競争で本当に争われるもの

### 「賢い返答」より「いつ、どのくらい話すか」

Moshiは並列音声ストリームによる対話を示し、Full-Duplex-Benchは、沈黙、相槌、割込み、応答開始の評価を定式化した。dGSLMも発話者別の音声チャネルを生成対象にした。[Moshi](https://arxiv.org/abs/2410.00037) [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) [dGSLM](https://arxiv.org/abs/2203.16502)

競争の中心はLLMの知識量ではなく、次の五つである。

1. 沈黙を「考え中」「発話終了」「感情的な間」のどれとして扱うか。
2. ユーザーの相槌を割込みと誤認せず、AI自身は短い相槌にとどまれるか。
3. 周囲の会話、テレビ、案内放送、AI自身の再生音を目的話者から分離できるか。
4. 会話を止めずに検索・予約・記録・安全確認などの業務をどこまで並行実行できるか。
5. 誤割込み、聞き逃し、過剰な相槌を、言語・業務・騒音条件ごとに評価できるか。

## 6. 日本の機会と不足

日本は、研究（LLM-jp／名古屋大）、通信・CX（NTT）、業務実行（AI inside）、現場AI（日立）、大規模サービス接点（LINEヤフー、サイバーエージェント、富士通）を持つ。足りないのは、一社がすべてを保有することではない。以下の接続である。

- **共通評価**：Full-Duplex-Bench-JAに、電話・騒音・複数話者・敬語・相談・安全性を含む実会話評価を足す。
- **データ**：相槌や発話の重なりをノイズとして消さず、話者分離したまま学習できる日本語データを作る。
- **運用基盤**：エコーキャンセル、目的話者推定、録音同意、個人情報の扱い、応対ログの監査を会話モデルと一体にする。
- **用途の選別**：雑談の人間らしさだけを追わず、コールセンタ、予約、現場支援、傾聴支援のように「早く割り込める／待てる」ことの価値を測る。

この意味で、日本が取るべき位置は「世界最大の音声モデルを作ること」ではない。**日本語と業務現場で、full-duplexが安全で測定可能な価値を生む条件を定義し、オープン基盤と実サービスをつなぐこと**にある。

## 調査上の注意

- 「公開資料で確認できない」は、取り組みが存在しないという断定ではない。非公開研究、顧客限定提供、日本語のみの発表は残り得る。
- 各社の性能値・導入規模・業務改善値は、特記がない限り各社発表であり、独立比較ではない。
- 本レポートは、モデルを出したかではなく、公開されたプロダクト、研究、API、データ、評価、業務実装から戦略の方向を読むことを目的とする。
