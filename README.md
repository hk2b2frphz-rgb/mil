# moshimoshi-J

日本語の傾聴対話向けに Moshi (`llm-jp/llm-jp-moshi-v1`) をドメイン適応する
パイプライン。合成対話の生成から学習・評価まで。

```
対話生成 -> TTS(ステレオWAV) -> 学習 -> 評価
```

## セットアップ

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune
uv sync
uv sync --project gemma_runtime
```

Python 3.11+ / NVIDIA GPU。計算ノードからの外部取得にプロキシが要る環境では
`PROXY_URL` を明示的に渡す（`qsub -V` で引き継がれる）。

```bash
qsub -v PROXY_URL=http://<proxy-host>:<port>,MODEL_ID=base scripts/run_full_duplex_eval.pbs
```

## 1. 学習データ生成

対話を生成し（A100・vLLM で Qwen3.6-27B）、話者ごとに全発話を連結して1回で
合成、MMS_FA で境界を復元する。

```bash
qsub -V scripts/run_dialogues_qwen_3000.pbs
qsub -V scripts/run_qwen_tts_whole_utterance_3000_4gpu.pbs
```

対話数は `1000` / `3000` / `10000` の3系統。動作確認は `_smoke` を使う。
TTS のバックエンドは 1000 が Qwen3-TTS、3000/10000 は Kokoro（Qwen3-TTS では
walltime 内に終わらないため）。

### KABURI-TTS バックエンド

上の経路は発話ごとに合成して並べるため、チャンネル間の重なりがほとんど残らない
（[KABURI-TTS 論文](https://arxiv.org/abs/2609.07200) の Irodori 系ベースライン
と同じ条件で、overlap 0.013 / 交替 11.6 回per分）。相槌のかぶりを含む学習データ
が要る場合は、2話者を左右チャンネルへ同時にレンダリングする KABURI-TTS を使う。

```bash
bash scripts/setup_kaburi_env.sh            # 初回のみ（../kaburi-tts を用意）
qsub -V scripts/2026-09-04/aizuchi_normal_kaburi_tts_3000.pbs
```

セットアップで詰まりやすい2点は `scripts/setup_kaburi_env.sh` と
`scripts/kaburi_uv_env.sh` の冒頭に対処を書いてある。

- `invalid peer certificate: UnknownIssuer` / `download-r2.pytorch.org` に繋がらない
  → TLS 傍受プロキシ。`SSL_CERT_FILE` か `KABURI_TORCH_FROM_PYPI=1`
- `libnvrtc.so.13: cannot open shared object file`
  → torchaudio 2.9+ が load/save を torchcodec 経由にしており、その wheel の CUDA が
  torch と食い違っている。`scripts/fix_kaburi_audio_io.sh` が自動修復する。setup と
  レンダリングジョブの preflight の両方から呼ばれるので、**別途実行する必要は無い**
  （健全なら何もしない）。単体確認は `bash scripts/run_kaburi_audio_check.sh`

本番前に3対話だけ流して所要時間と重なりを測る smoke が両方式に用意してある。
`scripts/report_tts_smoke.py` が対話あたりの秒数・RTF・3000本換算の見積もりと、
論文 Table III と同じ定義の overlap / silence / switches-per-min を出す。

```bash
qsub -V scripts/2026-09-04/aizuchi_normal_kaburi_smoke.pbs   # KABURI
qsub -V scripts/2026-09-04/aizuchi_normal_qwen_smoke.pbs     # 既存のQwen3経路（比較用）
```

インタラクティブノードでは同じものを直接叩ける（PBS版はこれにキューを付けただけ）。
`both` は両方流してから並べた表を出す。

```bash
bash scripts/run_tts_smoke.sh kaburi 3
bash scripts/run_tts_smoke.sh both 3
```

タイミングの作り方は3種類あり、音響モデルは共通で「どの発話をいつ始めるか」だけが違う。
`RASTER_MODE`（本番）/ smoke の引数で選ぶ。

| モード | 実体 | 学習データに使えるか |
| --- | --- | --- |
| `pred`（既定） | realizer + gap model（上流の最新版, `kaburi_tts/raster/`） | 使える |
| `stat` | 学習コーパスの統計配置（`kaburi_tts/placement/`、論文の統計配置条件） | 使える |
| `paper` | 論文版の単一 timing predictor（`kaburi_tts/predictor/`） | **使えない**（発話単位のタイミングを返さないので alignments が書けない。聴き比べと重なり計測専用で、`ALLOW_MISSING_ALIGNMENTS=1` が要る） |

```bash
bash scripts/run_tts_smoke.sh kaburi-all 3   # 3モードを順に流して並べる
RASTER_MODE=stat qsub -V scripts/2026-09-04/aizuchi_normal_kaburi_tts_3000.pbs
```

相槌の頻度プリセットは `AIZUCHI_PRESET` で切り替える。コーパスはプリセットごとに
別ルート・別バージョン（normal=v6 / eager=v7 / flood=v6。eager が v7 なのは v6 の
eager が normal-v6 より薄く出てしまい、プリセットを引き上げ直したため）。

```bash
AIZUCHI_PRESET=eager bash scripts/run_tts_smoke.sh kaburi-pred 3
```

配置（間・かぶり）は KABURI の gap model が決めるので `LEAD_IN_SEC` / `GAP_SEC` /
`--auto-overlap-aizuchi` は無い。発話境界は入力した音素ラスタから取るため MMS_FA も
通さない。KABURI のキャンバスは30秒固定なので、対話は30秒以内のチャンクに切って
合成し連結する（境界で韻律が途切れる）。codec が bfloat16 で復号するため A100
（`xan_s`）で回す。出力の形は Qwen3/Kokoro 版と同じで、`merged/` 以降は共通。

### Qwen3-TTS の多様性プローブ

相槌を大量に作り置きして、欲しい間・韻律のものを検索で引く――という案を
試すなら、先に「同じ参照音声で作った相槌が、そもそも散らばるのか」を見ておく。
参照 1 本で相槌 20 種 x 10 本 = 200 本を合成し、散らばりを 2 つに分けて測る。

| | 何を見るか |
| --- | --- |
| text 内 | 同じ文字列を投げ直したときのばらつき。サンプリング由来 |
| text 間 | 「うーん」「うーーん」「うぅん」のように書き方を変えたときのへだたり |

```bash
qsub -V scripts/2026-09-14/qwen3_tts_diversity_probe.pbs
bash scripts/run_qwen3_tts_diversity.sh          # インタラクティブノード
TEMPERATURE=1.2 bash scripts/run_qwen3_tts_diversity.sh
```

出力は `data/runs/diversity/<batch>/report.md`。1 音ごとに尺・F0・拍・有声区間を
取り、group ごとの表、隣の group との隔たり `d`、全体の分離比を出す。

- group 内 cv が 0 近辺 → 同じ入力をいくら投げても貯まらない
- `d` < 1 → 隣と分布が重なる。1 本引いて狙った長さは出ない（引いて測って選ぶ）
- 分離比だけだと「平均が近い」のか「group 内が広い」のか区別できないので `d` を見る
- `cv` と `cv*` が食い違う → 広がったのではなく尾を引いているだけ。`cv*` は
  中央値の 2.5 倍を外れた尺を抜いた cv で、`外れ` 列がその本数。低温での
  繰り返しや無音の垂れ流しはここに出るので、「散った」と「壊れた」を
  取り違えずに済む

既定の 20 種は実録音（116 件）で上位を占めた うん / うーん / そっか / うんうん を
中心に伸ばし方と表記を振ったもの。いずれも今の合成語彙には無い語なので、
そもそも出せるのかの確認も兼ねている。`TEXTS_FILE` / `TEXTS` で差し替えられる。

#### 同じ text のまま温度で散らすか

表記を変えれば散って当然なので、バンクを厚く積むときに効くのは「同じ入力で
どこまで散るか」の方。text を固定して温度だけ振る:

```bash
qsub -V scripts/2026-09-14/qwen3_tts_temperature_sweep.pbs
TEXTS=<word> REPEATS=50 TEMPERATURE_SWEEP=0.7,1.0,1.3 bash scripts/run_qwen3_tts_diversity.sh
```

温度ごとに group が分かれ（`うん @T1.3`）、`## temperature を振ったとき` の表に
並ぶ。`REPEATS` は温度あたりの本数で、エンジンは 1 回しか積まない。既定の
sampling params も報告に載る。`--temperature` などは production のバックエンドに
手を入れず、このプローブの中だけで stage-0 に差し込む。

#### 測り直し

測り方を変えたときは合成し直さない。作り直すと別の音になって前の結果と
比べられないため。GPU も vLLM も要らない:

```bash
uv run python scripts/qwen3_tts_diversity_probe.py --out-dir <dir> --reanalyze
```

F0 は自己相関の候補を複数残し、発話全体の代表 F0 から離れた候補に罰則を付けて
選ぶ（オクターブ飛びの抑制）。それでも group 内の F0 sd が平均の 15% を超えたら
`要確認` を立て、理由まで出す。

| 理由 | 中身 | 次の手 |
| --- | --- | --- |
| `octave` | 計測の取り違えが残っている | トラッカー側を直す |
| `accent` | 促音などで有声が 2 島に割れ、どちらが長いかで中央値が振れている | `F0 頭`（先頭の島）の列を読む |
| `unstable` | どちらでもない | 聴く |

`F0 頭` は同じ語なら島の並び順が変わらないので、draw をまたいで比べられる。
`跳ね(st)` は島から島への半音差＝アクセントの大きさで、これ自体が検索の
マッチング条件に使える量でもある。

### バンクだけで聞き手を作る（端から端まで）

user 側は TTS、聞き手側はバンクからの検索、タイミングは KABURI の予測。
提案そのものを一度に通す。

```bash
qsub -V scripts/2026-09-15/aizuchi_bank_stereo_test.pbs
bash scripts/run_bank_stereo_test.sh 3        # インタラクティブノード
```

4 段。最初の 1 段があるのは、**いまのコーパスには うん が 1 つも無い**ため。
実録音 116 件の 76% を占めた 4 語（うん / うーん / そっか / うんうん）が
どれも対話生成の語彙に無いので、バンクを試す相手のコーパスは作るしかない。

| | 何をするか | どこで |
| --- | --- | --- |
| 1 | 聞き手の相槌を実測の頻度から引き直す＋user だけの写しを出す | CPU・一瞬 |
| 2 | Qwen3-TTS で **user 側だけ**合成 | V100 |
| 3 | 対話テキスト全体から KABURI の配置（音響モデル無し） | CPU |
| 4 | 無音の聞き手チャンネルにバンクの音を差し込む | CPU |

**聞き手の音は一度も合成しない。**バンクから来るので要らないうえ、合成すると
捨てる音が user 側のタイムラインを押してしまう（重ならなかった聞き手ターンは
それ自体が時間を占める）。

`FULL_RENDER=1` で対話まるごと合成して聞き手チャンネルを後から差し替える経路に
戻せる。無駄な合成とタイムラインのずれを払う代わりに、捨てるはずの聞き手の音が
**同じ語・同じ位置で「合成した場合」の A/B ベースライン**として残る。

差し替えは**語を合わせる**。対話が うん と そっか を打ち分けているのに
どちらにも うん を差したら打ち分けが消えるので、同じ語のクリップから引き、
無ければ尺で代用してその件数を報告する。

頻度の定義は `scripts/2026-09-15/real_backchannel_dist.tsv`（重みは実測の生の
件数）。1 語だけで試すなら `BACKCHANNEL_TEXT=うん`。バンクは書き換えが入れる語を
持っている必要がある — 20 表記のプローブは既定の 7 語を全部持っているが、
温度スイープは うん しか持っていない。

### 相槌バンクの差し込み（KABURI × Qwen3）

配置は KABURI、音はバンクという分担を実際に聴いてみるための実験。経路は 2 つ
あり、**KABURI に何をさせるか**が違う。

| | KABURI | 音源 | GPU |
| --- | --- | --- | --- |
| 配置のみ | いつ相槌を打つか（realizer + gap model） | 既存のレンダリング済みコーパス | **不要** |
| 全合成 | いつ + 相手の発話の音も | KABURI の出力 | A100 |

**配置のみ（既定で試すならこちら）。** KABURI の音は 1 サンプルも使わないので、
音響モデルを積む必要が無い。既存の Qwen3/Kokoro コーパスの user チャンネルを
そのまま使い、聞き手チャンネルだけを差し替える。

```bash
bash scripts/run_kaburi_placement_bank.sh 3
qsub -V scripts/2026-09-15/aizuchi_normal_placement_bank.pbs
```

`SOURCE_DIR`（user 側の音声を借りる既存コーパス）は既定で、そのプリセットが
すでに合成されている先を順に探す: `data/runs/<corpus>/tts/merged/training_set`
→ 同 `shard_*/training_set` → 最新の Qwen3 smoke。解決した先は必ず表示する。

対話の対応は**ファイル名ではなく dialogue id** で取る。シャードに割ると
`sample_00001` が振り直されるので、名前で合わせると別の対話が対になる。

移すのは**位置**。相槌ごとに「どの user 発話の、どこで」打たれたかを KABURI の
時間軸で読み、既存側の同じ順番の発話に当てはめる。2 通りに分けて持つ。

| | 意味 | 持ち方 |
| --- | --- | --- |
| `inside` | 相手の発話の最中に入った（＝かぶり） | 発話長に対する**割合** |
| `after` | 言い終わってから入った | 終わりからの**秒数** |

終わりからの差だけで測るとかぶりを表せない。相槌は言い終わるのを待つものでは
なく、実録音でも「うん」の 71% は相手の発話に重なっていて、終わりから測った差は
p10 で −17 秒だった。「終わりの何秒前」ではなく「発話のどのあたり」が本体なので、
`inside` は割合で持つ。既存側の発話が KABURI 側より長くても短くても、同じ
「あたり」に入る。

user の音声は伸縮も移動もしない。相槌の対応は**出てくる順**で取り、本数が合わない
対話は黙って通さずスキップする。実行後に「相手の発話中 N / 言い終えた後 M」が
出るので、**KABURI がそもそも重ねて置いているのか**がそこで分かる。

`--placement-only` は `generate_kaburi_tts_data.py` のフラグで、acoustic model を
ロードしないまま配置だけ JSON に出す。5GB のチェックポイントも A100 も要らない。

```bash
uv run python scripts/generate_kaburi_tts_data.py --placement-only \
  --dialogues-jsonl <jsonl> --out-dir <dir> --ref-pack <pack> --device cpu --mode pred
```

**全合成。** KABURI で普通にレンダリングしたあと、出来上がったステレオ WAV の
相槌区間だけを差し替える。合成し直さないので配置も相手の発話も 1 サンプル動かず、
差し替え前後をそのまま A/B できる。

```bash
qsub -V scripts/2026-09-15/aizuchi_normal_kaburi_bank.pbs
bash scripts/run_kaburi_bank_dialogues.sh 3        # インタラクティブノード
```

差し替え先は **KABURI が空けた区間の長さに近い順**に並べ、上位 `MATCH_TOP_K`
本から 1 本引く（毎回いちばん近いものを取ると同じ音ばかりが並ぶため）。区間より
長い音はそのまま相手に食い込ませる（切らない）。相槌のかぶりはむしろそれが
自然なので、区間は「始まる位置」であって「収める箱」ではない。

`BANK_DIR` は既定で `data/runs/diversity/` の最新。1 語だけのバンクで構わない
（温度スイープの出力が向く）。実録音 116 件では「うん」だけで 28 件、上位 4 語で
76% を占めていて、いま合成に使っている語彙とほぼ重ならない。

**置き換える相槌は語彙で決める**（`scripts/2026-09-15/continuer_vocab.txt`）。
相槌には 2 種類あり、うん で代用できるのは片方だけ。

| | 例 | うんで代用 |
| --- | --- | --- |
| 継続 | はい / ええ / うん、うん | できる |
| 評価 | そうなんですね / なるほど / 大丈夫ですよ | **できない** |

評価の方を うん にすると、相手の話への反応が消えて会話が成立しなくなる。
文字数で切ると「なるほど」（4 文字）が継続側に混ざるので、長さではなく語で
判定する。実行後に「差し替えた語」と「置き換えなかった短い発話」の内訳が
出るので、語彙を広げるかどうかはそれを見て決める。

差し替えだけ CPU で何度でもやり直せる。

```bash
uv run python scripts/splice_aizuchi_bank.py \
  --training-dir <dir> --bank-dir <bank> --out-dir <dir>_k1 --match-top-k 1
```

何をどこに差したかは `bank_swaps.jsonl`。`metadata.aizuchi_bank` にも残る。
音と `alignments` は差し替え後に揃えてあるが、`metadata.dialogue.turns` は
「何を言わせるつもりだったか」の記録として元のまま残してある。

### 実録音の語彙で 500 本（4 ジョブ）

対話生成から full-FT まで、実録音で数えた相槌語彙で通す一式。
`scripts/2026-09-15/` に 4 本。順に投げる。

```bash
qsub -V scripts/2026-09-15/real_aizuchi_dialogue_500.pbs   # 1. 対話 500 本   A100
qsub -V scripts/2026-09-15/real_aizuchi_bank_10000.pbs     # 2. 応答バンク    V100 x4
qsub -V scripts/2026-09-15/real_aizuchi_stereo_500.pbs     # 3. ステレオ合成  V100
qsub -V scripts/2026-09-15/real_aizuchi_fullft_500.pbs     # 4. full-FT       A100
```

1 と 2 は独立なので同時に投げてよい。3 は両方が要る。

**1 の相槌の入れ方が従来と違う。**句の切れ目ごとの確率で抽選して上限で抑える
のをやめ、**どこで打つかをモデルに決めさせる**（`AIZUCHI_ONLY_PLACEMENT=llm`）。
挿入数の上限は無く、**発話の末尾も候補**（言い切った所で黙る聞き手は、聞いて
いないのと同じに聞こえる）。語彙は
`scripts/2026-09-15/real_backchannel_dist.tsv`。実録音 116 件の上位 4 語は
従来の語彙にひとつも無いので、従来のコーパスではそもそも扱えない。

**語彙は結局、従来の書き起こし語彙（`AIZUCHI_ONLY_VOCAB`: はい。／ええ。／
そうなんですね。／あぁ…。など）に戻した。** 実測（`real_backchannel_dist.tsv`）
から作った語彙は「そっか」系中心で、聴くと「そっか」が浮いた（例文が
そっか系に偏っていたことも一因）。従来語彙に戻すと、今度は裸の「うん。」
だけが浮いたので、そこだけ抜いてある（`aizuchi_vocab_no_bare_un.tsv`。
「うん、うん。」「はい、はい。」は従来通り残す）。

途中で作った実データ由来の語彙・例文一式（`listening_vocab.tsv`、
`real_aizuchi_examples.md`）はファイルとして残してあるが、既定では使わない。
そちらの方向を再検討するときのために置いてある。

この過程で見つけたバグ: プロンプトの指示文が「そっか」を直接書いていて、
語彙を差し替えても文字列として残ってしまっていた（今の語彙には「そっか」が
無いのに「そっかを使ってください」と指示していた）。特定の語を名指しせず
「語彙にある一番短い語」のように一般化する形に直した。

**さらに位置決めそのものを `--aizuchi-only-placement density` に変えた。**
モデルに位置も語も決めさせる `llm` は、そっかへの偏り・thinking の予算切れ
バグなど、今日の問題のほとんどの発生源だった。`density` は:

- 位置は句の切れ目ごとの確率抽選（従来の `rule` と同じ考え方）。ただし
  **発話の末尾は密度に関わらず必ず打つ**（従来は末尾も他の位置と同じ確率
  抽選だったので、reserved では末尾すら無反応があり得た）
- 語は従来の `rule` と同じく LLM に選ばせる（一度はランダム抽選にしたが、
  「相槌の種類選択は文脈を見て選ばせていたはず」という指摘で差し戻した）。
  位置決めのような判断は不要でも、直前の発話内容に応じて受け止め語を優先
  するといった選び方はランダムでは再現できない
- 頻度は `AIZUCHI_DENSITY`（0〜1 の連続値）。0 は相槌 0 件、0.25/0.5/0.75 が
  reserved/normal/eager 相当、1 が flood（打ちすぎ側の上限。推論時の既定には
  しないが、学習データ生成には使ってよい）。アンカーの間は線形補間する

`rule`・`llm` はどちらもコードとしては残っており、`--aizuchi-only-placement`
で選び直せる。

`AIZUCHI_NO_REPEAT_WINDOW=0`。既定の 3（直近 3 回に使った語は禁止）だと
最頻形が打てなくなる。実録音では「うん」だけで 4 回に 1 回だった。

**相槌頻度をモデルに条件付けする。** 密度は元々データ生成時だけのパラメータで、
学習時にはモデルへ一切伝わらず、生成された対話に埋め込まれた結果（相槌の
間隔・頻度）からしか学習できなかった。頻度をモデルへの制御可能な入力にする
ために、以下を追加した:

- `Dialogue.aizuchi_frequency_label` フィールド。placement=density なら実際に
  使った値（例: `"density=0.75"`）、rule ならプリセット名（`"eager"` 等）、
  llm なら頻度を制御していないことを表す `"llm"`。`dialogues.jsonl` に
  シリアライズされ、TTS 側（`generate_qwen3_tts_data.py` /
  `generate_kaburi_tts_data.py` のどちらも）が sidecar JSON の
  `metadata.dialogue.aizuchi_frequency_label` に引き継ぐ。間に挟まる
  `rewrite_aizuchi_vocab.py` と `splice_aizuchi_bank.py` は元の dict /
  metadata をそのまま残すので、対話生成から学習データまで欠けずに届く
- `--aizuchi-density-mixed`（`AIZUCHI_DENSITY_MIXED=1`）。`--aizuchi-density` の
  固定値を無視し、対話ごとに 0〜1 の一様乱数を引く
  （`--aizuchi-only-frequency mixed` の density 版）。1 ジョブの中で 0〜1
  全域をカバーする対話が作れるので、単一の値しか無い固定密度ジョブより
  条件付けの学習に向く
- `scripts/inject_inner_thoughts.py --from-aizuchi-density`。既存の「読み上げ
  ない内心をテキストストリームにだけ差し込む」仕組み（音声は一切変更しない）
  を流用し、`aizuchi_frequency_label` から `<相槌:density=0.75>` のようなタグを
  対話の冒頭挨拶の直前に 1 つだけ埋め込む。emotional_state 版
  （`--from-emotional-state`）と違い対話全体で値が変わらないのはむしろ狙い
  通りで、学習用途に使ってよい

  **注意:** タグは挨拶の直前の無音に収める必要があるが、この対話生成
  パイプラインが使う KABURI レンダラーは無音ターンを明示的なタイミングとして
  扱わず（無音ターンは落として自前のタイミングモデルで間を作る）、生成側
  からは挨拶の前に無音がどれだけあるか制御できない。無音が足りない対話では
  タグが置けずに skip され、生成時に付けたはずの頻度がモデルに伝わらないまま
  学習データに混ざってしまう（条件付け精度に直接効く）
- `--pad-lead-in-sec`。上の問題を「偶然の無音を当てにする」のではなく
  「確実な予約済みスロットにする」ことで解決する。音声の先頭に固定秒数の
  無音を実際に足し、`alignments_utterance` の全タイムスタンプをその分だけ
  後ろへずらしてからタグを挿す。`--from-aizuchi-density` と併用することを
  推奨する。挿入後は `WARNING: ... density_tag_dropped` が出ないこと
  （＝全対話にタグが乗ったこと）を確認する。それでも出る場合は
  `--pad-lead-in-sec` の値がタグの長さに対して短すぎる

- 使い方は `scripts/2026-09-15/real_aizuchi_fullft_500.pbs` のコメント参照。
  step 1（対話生成）を `AIZUCHI_DENSITY_MIXED=1` で回してから
  `inject_inner_thoughts.py --from-aizuchi-density --pad-lead-in-sec 3.0` を
  step 3 の出力に対して実行し、その `--out-dir` を fullft の `SRC_RUN_DIR`
  に渡す（`aizuchi_normal_fullft_10000.pbs` の inner-thought 版と同じ
  パターン）

推論側で同じタグをモデルのテキストストリームの先頭に与えて頻度を選ばせる
実装は、学習コード自体（`../moshi-finetune-nu-dialogue`）が別リポジトリに
あり本リポジトリの管轄外。ここではタグの形式と生成側の契約を揃えるところ
まで。

**one-shot 例は実データから**（`AIZUCHI_ONLY_EXAMPLE=1`、既定で有効）。実際の
タイムスタンプ付き書き起こしを見ると、裸の一語（「うん」「うん」と2回連続）は
ごく普通に出てくる。一方「そっかー。何年生かな?」のように相槌と質問が同じ
ターンに融合している箇所もあり、このモードが再現したいのは相槌だけなので、
質問部分は落として例に使っている。プロンプトの「裸の一語を並べないでください」
という指示も、この発見を受けて撤回した（誤りだった）。

沈黙と「聞いていますか?」は **有効**（normal v6 では切っていた）。聞き手モデルが
このデータからしか学べないことが 2 つある — 沈黙を相づちで埋めないこと、
問いかけには相づちではなく返事をすること。確認は沈黙の後にしか起きないので、
沈黙を 0 にすると確認も消える。返事も口語に揃えてある
（`listening_probe_replies.txt`。既定のままだと聞き手が急に敬語へ戻る）。

**思考を入れて位置を決めさせることもできる**（`AIZUCHI_ENABLE_THINKING=1`）。
このパイプラインは他の全エージェントで逆方向 -- `/no_think` と低い
`reasoning_effort` -- に振ってある。ほとんどの呼び出しはトークン予算が小さく、
reasoning モデルが分析チャンネルに予算を使い切って本文が空になるため。相槌の
位置決めだけは、発話の重み・進み具合を見て判断するという性質上、他より
reasoning に向いている可能性がある。有効にすると `AIZUCHI_THINKING_MAX_TOKENS`
（既定 1600）まで予算を上げる。既定は無効。

予算内に考え終わらないこともある。`</think>` が閉じないまま切れると JSON が
一切届かない。固定の予算をいくら上げても発話ごとに考える長さは変わるので
「今度は足りる」保証にならず、代わりに**失敗を検知して thinking なしで
自動再試行**する。「モデルが reactions=[] と正しく判断した」場合（JSON は
ちゃんと届いている）と「予算切れで JSON が無い」場合は区別してあるので、
反応無しが正しい判断のときに無駄な再試行はしない。

2 は 24 形 × 120 本 × 4 シャード = 11,520 本。`REPEATS` はシャードあたりで、
どのシャードも全語彙を引くので、シャードが落ちても失うのは深さだけ。温度は
既定のまま（上げても広がらず、下げると壊れた draw が尾を引くだけだった）。

3 は **聞き手側を一度も合成しない**。user 側だけ合成し、KABURI の配置に
バンクの音を差し込む。`KEEP_TEXT=1` なのは 1 ですでに実録音の語彙で生成して
あるため（語を引き直させない）。

## 2. 学習

ハイパラは `experiments/<name>/config.yaml` で管理する。

```bash
bash scripts/run_experiment.sh lora_base_config ./data/runs/<RUN_ID>
```

walltime を越える step 数はチェーンで分割する。LR スケジュールを連続させる
ため `max_steps` は全ジョブで `TOTAL_STEPS` 固定。

```bash
qsub -v 'EXP_NAME=lora_base_config,SRC_RUN_DIR=data/runs/<RUN_ID>,TOTAL_STEPS=7200' \
  scripts/run_train_chain.pbs
```

h01 のようなスイープパターンをそのままチェーンで回す場合（`TOTAL_STEPS` は
パターン自身の step 数が既定になる）:

```bash
qsub -v 'EXP_NAME=lora_base_config,SRC_RUN_DIR=data/runs/<RUN_ID>,SWEEP_PATTERN=h01,STEPS_PER_JOB=1000' \
  scripts/run_train_chain.pbs
```

パターンの定義は `scripts/sweep_patterns.sh`。スイープとチェーンで共有する。

チェーンのロジックだけを先に検証する:

```bash
bash scripts/check_train_chain.sh
```

### スイープ

複数パターンを順に回す。**そのまま投げれば walltime を跨げる。** 残り1時間に
なった時点で学習を止め、最後のチェックポイントから続きを次のジョブに引き継ぐ
（同じデータセットを使い回し、完了済みパターンはスキップする）。

```bash
qsub -V scripts/sweep_lora.pbs
qsub -V scripts/fullft_sweep.pbs
```

進捗は `experiments/pbs_logs/<RUN_ID>_sweep_state.tsv`。引き継ぎを止めるなら
`SWEEP_CHAIN=0`、走っているチェーンを終わらせるなら
`touch ~/.miltoka/stop_sweep_chain`。

止める余裕は `TIMEBOX_LEAD_SEC`（既定 3600 秒）。チェックポイントは
`HP_CKPT_FREQ` step ごとに書かれているので、引き継ぎ1回あたり失うのは
最大でその step 数。

full-FT も同じように跨げる。ただし再開の作りが違う: LoRA はアダプタを
読み直すだけだが、full-FT は ZeRO チェックポイントを
`export_fullft_checkpoint.py --intermediate-only` で MoshiForFinetuning 形式に
戻し、それを次ジョブの `NU_MODEL_DIR` に渡す。重みだけが渡り optimizer state は
渡らないので、残り step 数を引いたうえで warmup は初回のみ行う。

full-FT のチェックポイント間隔は `HP_CKPT_FREQ`。1回の walltime 内に最低1つは
書かれる値にしておくこと。1つも無いと再開点が作れず、そのジョブは失敗する。

## 3. 評価

Full-Duplex-Bench-JA。計算ノード側は外部 API を呼ばない。

```bash
qsub -v MODEL_ID=base scripts/run_full_duplex_eval.pbs
qsub -v MODEL_ID=lora_h01,MODEL_WEIGHT=/path/model.safetensors,MODEL_CONFIG=/path/moshi_lm_kwargs.json \
  scripts/run_full_duplex_eval.pbs
```

LoRA は評価前にマージする。

```bash
qsub -v LORA_CKPT=/path/to/checkpoint/consolidated/lora.safetensors scripts/merge_lora.pbs
```

## ディレクトリ

| | |
| --- | --- |
| `scripts/` | 生成・学習・評価の実行スクリプトと PBS ジョブ |
| `eval/` | Full-Duplex-Bench-JA の評価本体 |
| `configs/` | DeepSpeed / TTS / GRPO の設定 |
| `experiments/` | 実験ごとのハイパラ |
| `eval_sets/` | 評価シナリオと正解データ |
| `gemma_runtime/` | 対話生成・カスケード用 Gemma の隔離 venv |
| `agent_hpc/` | A100 上のコーディングモデルをローカルから使う一式 |
