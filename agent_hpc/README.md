# agent_hpc — A100 ノードを「ローカルのエージェントの頭脳」にする

ローカル PC の Claude Code / Codex CLI から、HPC の A100 上で動く
Qwen3.6-27B を叩くための一式です。

```
[A100 計算ノード]                          [ローカル PC]
  vLLM  127.0.0.1:VLLM_PORT                 claude
    ^ localhost                               |
  LiteLLM proxy  0.0.0.0:PROXY_PORT           v
    ^                                    127.0.0.1:8787
    |                SSH トンネル             |
    +--------------- (自分で張る) ------------+
```

- **vLLM** は OpenAI 互換 API しか話せず、Claude Code は Anthropic 互換 API を要求します。
  そこで同じノード上に **LiteLLM プロキシ**を立てて Anthropic ⇄ OpenAI を変換します。
  ローカルからはこのプロキシの 1 ポートだけ見えれば十分です。
- vLLM は `127.0.0.1` 固定、外に出るのはプロキシのポートのみ。
  そのプロキシにも**ジョブごとのランダム API キー**を掛けてあります。

## 1. HPC 側でサーバを起動

```bash
qsub -V agent_hpc/pbs/run_qwen_agent_server.pbs
```

ジョブは walltime（既定 24h）いっぱい生き続けます。止めるときは `qdel <jobid>`。

### walltime を越えて生かす（チェーン）

既定では walltime（24h）で止まります。跨いで生かすなら `CHAIN=1` を付けます。

```bash
CHAIN=1 qsub -V agent_hpc/pbs/run_qwen_agent_server.pbs
```

walltime の20分前に次のジョブを投入するので、キューに並んだ状態で現ジョブが
落ち、モデルのロード時間ぶんだけの空白で引き継がれます。既定で最大7リンク。

チェーンを終わらせるとき（現ジョブは walltime まで動き続けます）:

```bash
touch ~/.miltoka/stop_agent_server
```

**既定がオフなのは意図的です。** 勝手に復活するサーバーは、学習に使いたい
A100 の枠を黙って埋めます。エージェントに GPU を使うと決めた期間だけ有効に
してください。サーバーが10分未満で死んだ場合は、同じ失敗を繰り返さないよう
再投入しません。

## 2. 「IP みたいなの」が書き込まれるファイル

起動したジョブは、PBS が実際に割り当てたノードのホスト名・IP・ポート・API キーを
**HPC 上の決め打ちのパス**に書きます（ローカルからはここを見るだけで済みます）。

| ファイル | 用途 |
| --- | --- |
| `~/.miltoka/agent_endpoint.json` | 機械可読。`connect.ps1` が読むのはこれ |
| `~/.miltoka/agent_endpoint.txt`  | 人間用。トンネルのコマンドがそのまま書いてある |
| `~/.miltoka/agent_endpoint.env`  | `source` 用の環境変数 |

`status` は `starting` → `ready` → `stopped` と遷移します。`ready` になるまで
モデルのロード中なので、`ready` を確認してから使ってください。

置き場所を変えたいときは `ENDPOINT_FILE=/path/to/foo.json qsub -V ...`。

JSON の中身の例：

```json
{
  "status": "ready",
  "job_id": "123456.pbs",
  "node_host": "xan05.example.ac.jp",
  "node_ip": "10.1.2.34",
  "proxy_port": 41456,
  "model": "Qwen/Qwen3.6-27B",
  "model_alias": "qwen-agent",
  "api_key": "sk-hpc-....",
  "ssh_tunnel": "ssh -N -L 8787:10.1.2.34:41456 cs20049@login..."
}
```

## 3. トンネルを張る（ローカル PC）

`ssh_tunnel` の行をそのまま実行すれば OK です。ログインノードを踏み台にして、
計算ノードのプロキシポートをローカルの 8787 に持ってきます。

```bash
ssh -N -L 8787:10.1.2.34:41456 cs20049@login.example.ac.jp
```

ログインノードから計算ノードへ直接 TCP が通らないクラスタなら、多段にします：

```bash
ssh -N -J cs20049@login.example.ac.jp -L 8787:127.0.0.1:41456 cs20049@xan05
```

## 4. Claude Code をローカルで走らせる

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN = "sk-hpc-..."   # endpoint.json の api_key
$env:ANTHROPIC_MODEL = "qwen-agent"
$env:ANTHROPIC_SMALL_FAST_MODEL = "qwen-agent"
claude
```

手で貼るのが面倒なら、エンドポイント取得・トンネル・環境変数出力をまとめてやる
ヘルパーがあります：

```powershell
./agent_hpc/local/connect.ps1 -LoginHost login.example.ac.jp -User cs20049
```

`-ShowOnly` を付けるとトンネルは張らず、情報の表示と
`agent_hpc/local/agent_env.ps1`（dot-source 用、API キーを含むので gitignore 済み）
の生成だけを行います。

Codex CLI や aider など OpenAI 互換で話す道具なら、同じポートの
`http://127.0.0.1:8787/v1` をそのまま使えます。

## 4.5 まずは疎通確認とチャット

いきなり Claude Code を繋ぐ前に、素のチャットで動いているか見るのが早いです。
標準ライブラリだけの CLI を置いてあります（pip 不要）。

```powershell
# トンネルは通っているか、何のモデルが載っているか
python agent_hpc/local/chat.py --check

# 対話（Ctrl+C で生成中断、/reset で履歴クリア、/exit で終了）
python agent_hpc/local/chat.py
```

`connect.ps1` を一度実行していれば `agent_hpc/local/agent_env.json` から
URL・キー・モデル名を自動で拾うので、引数は要りません。手で指定するなら：

```powershell
python agent_hpc/local/chat.py --base-url http://127.0.0.1:8787/v1 --api-key sk-hpc-... --model qwen-agent
```

## 主な環境変数

| 変数 | 既定 | 意味 |
| --- | --- | --- |
| `VLLM_MODEL` | `Qwen/Qwen3.6-27B` | 載せるモデル（下記参照） |
| `AGENT_MODEL_ALIAS` | `qwen-agent` | エージェント側が指定する短い名前 |
| `PROXY_PORT` | ジョブ番号から `41000-42999` | 外に見えるポート |
| `VLLM_PORT` | `PROXY_PORT + 2000` | ノード内部専用 |
| `LOCAL_PORT` | `8787` | トンネル先のローカルポート（表示に使うだけ） |
| `MAX_MODEL_LEN` | `131072` | 入らないと言われたら 65536 に落とす |
| `VLLM_TOOL_CALL_PARSER` | `qwen3_coder` | 素の Qwen3 chat 系なら `hermes` |
| `VLLM_REASONING_PARSER` | `qwen3` | 非 thinking モデルを載せるなら空に |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU を増やすなら合わせて変更 |
| `AGENT_API_KEY` | ランダム生成 | 固定したいときだけ指定 |
| `ENDPOINT_FILE` | `~/.miltoka/agent_endpoint.json` | 書き込み先 |
| `LOGIN_HOST` | `PBS_O_HOST` | トンネル文面に出すログインノード名 |

## モデル選定（A100 80GB × 1 枚）

| 候補 | 実体 | VRAM | SWE-bench V. | Terminal-Bench 2.0 |
| --- | --- | --- | --- | --- |
| **Qwen3.6-27B** | 27B dense, bf16, 256K ctx | 約 54GB | **77.2** | **59.3** |
| Qwen3-Coder-Next | 80B/3B MoE, AWQ-INT4 | 約 40–46GB | 70.6 | 36.2 |
| gpt-oss-120b | MXFP4 MoE | 約 63GB | — | — |

名前に反して、**コーディングでも Qwen3.6-27B のほうが強い**。dense 27B なので
bf16 のまま A100 1 枚に載り、量子化に伴う不確実性が無い。対話生成ジョブが同じ
重みを既に落としてあるので、初回のダウンロードも要らない。

数値は Qwen 自身の agent scaffold での測定であり、第三者による再現は限定的な点は
割り引いて読むこと。

## A100 1 枚 vs V100 4 枚

**A100 1 枚を強く推奨します。** V100 は compute capability 7.0 で、

- **bf16 が無い**（fp16 のみ）。最近の重みは bf16 前提で、fp16 だと数値が壊れることがある
- **FlashAttention が使えない**（xformers / eager 止まり）→ 長コンテキストで極端に遅い
- **AWQ/Marlin 系の 4bit カーネルが実質使えない** → 4bit で載せる作戦が取れない
- Qwen3-Next 系の hybrid linear attention カーネルはそもそも Ampere 以降前提

32GB × 4 = 128GB という数字は魅力的に見えますが、TP4 の通信オーバーヘッドを
払ったうえで上記の制約が全部かかるので、エージェント用途では A100 1 枚のほうが
速く・賢く・トラブルが少ないです。V100 は使わない前提で組んであります。

## 補足・別のやり方

- **`qstat` から拾う**：エンドポイントファイルを使わず
  `ssh login "qstat -f <jobid> | grep exec_host"` でノード名を取る方法もありますが、
  ポートと鍵は結局どこかに書く必要があるので、ファイル方式のほうが素直です。
- **リバーストンネル**：ジョブ側から `ssh -R` でログインノードの固定ポートに
  出してもらえば、ローカルは常に同じポートを見れば済みます。ただし計算ノードから
  ログインノードへの鍵なしログインが要るので、まずは今の方式を推奨。
- **Tailscale / frp** のようなオーバーレイを入れるとトンネル不要になりますが、
  クラスタの規約に触れやすいので確認してから。
- **ツール呼び出しが崩れるとき**：`--tool-call-parser` がモデルと合っていない可能性が
  高いです。Qwen3-Coder 系は `qwen3_coder`、素の Qwen3 chat 系は `hermes`。
