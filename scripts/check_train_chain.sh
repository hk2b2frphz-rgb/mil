#!/usr/bin/env bash
set -uo pipefail

# scripts/run_train_chain.pbs の検証ハーネス。
#
# 学習も qsub も実際には行わない。サンドボックスを作り、run_experiment.sh と
# qsub をスタブに差し替えて、チェーンのつなぎ方と暴走ガードだけを確かめる。
# GPU もキューも要らないので、実ジョブを投げる前にこれを通す。
#
# 確かめること:
#   1. 区間の切り方(resume_step / stop_at_step)が正しいか
#   2. 各ジョブが前のジョブのチェックポイントを指しているか
#   3. TOTAL_STEPS に到達したら止まるか(= 無限に投げ続けないか)
#   4. 早期終了でチェックポイントが無いとき、次を投げずに正常終了するか
#   5. 暴走ガード(ジョブ数上限・即死検出)が効くか
#   6. 矛盾した指定(resume_from 無しの resume_step など)を弾くか
#
# 使い方:
#   bash scripts/check_train_chain.sh
#   qsub scripts/check_train_chain.pbs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAIN_SRC="$REPO_ROOT/scripts/run_train_chain.pbs"

if [[ ! -f "$CHAIN_SRC" ]]; then
    echo "ERROR: $CHAIN_SRC が見つかりません。" >&2
    exit 1
fi

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    echo "  [OK]   $1"
}

ng() {
    FAIL=$((FAIL + 1))
    echo "  [FAIL] $1"
}

check() {
    # check <説明> <条件が真なら成功>
    if eval "$2"; then ok "$1"; else ng "$1"; fi
}

setup_sandbox() {
    # $1 = run_experiment.sh スタブの中身
    rm -rf "$SANDBOX/work"
    mkdir -p "$SANDBOX/work/scripts" "$SANDBOX/work/bin"
    cp "$CHAIN_SRC" "$SANDBOX/work/scripts/run_train_chain.pbs"
    printf ':\n' > "$SANDBOX/work/scripts/setup_proxy.sh"
    printf '%s\n' "$1" > "$SANDBOX/work/scripts/run_experiment.sh"

    # qsub スタブ: -v の中身を記録し、その場で次のジョブを実行して連鎖を再現する。
    cat > "$SANDBOX/work/bin/qsub" <<'STUB'
#!/usr/bin/env bash
vars=""
script=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -V) shift ;;
        -v) vars="$2"; shift 2 ;;
        *)  script="$1"; shift ;;
    esac
done
echo "SUBMIT $vars" >> "$SUBMIT_LOG"
# 無限連鎖を万一起こしても止まるよう、ここでも独立に上限を掛ける。
count=$(grep -c '^SUBMIT ' "$SUBMIT_LOG")
if [[ "$count" -gt 20 ]]; then
    echo "harness: submission storm detected, aborting" >&2
    exit 1
fi
env $(echo "$vars" | tr ',' ' ') PBS_JOBID="stub.$count" bash "$script" >> "$RUN_LOG" 2>&1
echo "stub.$count"
STUB
    chmod +x "$SANDBOX/work/bin/qsub" "$SANDBOX/work/scripts/run_experiment.sh"

    # 実行環境には本物の qsub がある。スタブが先に解決されていることを確認して
    # から進む。ここを飛ばすと、検証のつもりで本物のジョブを投げかねない。
    local resolved
    resolved="$(PATH="$SANDBOX/work/bin:$PATH" command -v qsub || true)"
    if [[ "$resolved" != "$SANDBOX/work/bin/qsub" ]]; then
        echo "ERROR: qsub スタブが有効になっていません(解決先: ${resolved:-なし})。" >&2
        echo "       本物の qsub を叩く恐れがあるため中止します。" >&2
        exit 1
    fi

    : > "$SANDBOX/submit.log"
    : > "$SANDBOX/run.log"
}

run_chain() {
    # 引数はすべてチェーンへ渡す環境変数(KEY=VALUE)。
    #
    # PBS_O_WORKDIR をサンドボックスに固定するのが要点。run_train_chain.pbs は
    # 冒頭で `cd "${PBS_O_WORKDIR:-$(pwd)}"` するので、これを外から与えないと
    # PBS 上で実行したときだけ実リポジトリへ移動してしまい、スタブではなく
    # 本物の run_experiment.sh を呼ぶ。ローカルでは PBS_O_WORKDIR が未設定
    # なので再現せず、キュー上でだけ壊れる。
    (
        cd "$SANDBOX/work" || exit 1
        export PATH="$SANDBOX/work/bin:$PATH"
        export PBS_O_WORKDIR="$SANDBOX/work"
        export SUBMIT_LOG="$SANDBOX/submit.log"
        export RUN_LOG="$SANDBOX/run.log"
        env "$@" bash scripts/run_train_chain.pbs
    ) >> "$SANDBOX/run.log" 2>&1
    echo $?
}

# 正常系のスタブ: 指定された stop_at_step にチェックポイントを作る。
STUB_OK='#!/usr/bin/env bash
set -euo pipefail
echo "[stub] EXP=$1 RUN_TS=$RUN_TS max=$HP_MAX_STEPS resume=$HP_RESUME_STEP stop=$HP_STOP_AT_STEP from=${HP_RESUME_FROM:-NONE}"
D="experiments/$1/checkpoints/${RUN_TS}/checkpoints/checkpoint_$(printf "%06d" "$HP_STOP_AT_STEP")/consolidated"
mkdir -p "$D"
touch "$D/lora.safetensors"'

# 早期終了のスタブ: stop_at_step より手前のチェックポイントしか作らない。
STUB_EARLY_STOP='#!/usr/bin/env bash
set -euo pipefail
echo "[stub] early stop before $HP_STOP_AT_STEP"
mkdir -p "experiments/$1/checkpoints/${RUN_TS}/checkpoints/checkpoint_001800/consolidated"'

echo "=================================================================="
echo "run_train_chain.pbs 検証"
echo "sandbox: $SANDBOX"
echo "=================================================================="

# --- 1. 3 ジョブのチェーンが最後まで通る ------------------------------------
echo ""
echo "[1] TOTAL_STEPS=7200 / STEPS_PER_JOB=2400 -> 3 ジョブ"
setup_sandbox "$STUB_OK"
status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=7200 STEPS_PER_JOB=2400 CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "終了コード 0" "[[ '$status' == '0' ]]"

submits="$(grep -c '^SUBMIT ' "$SANDBOX/submit.log" || true)"
check "投入は 2 回(job2, job3 のみ)" "[[ '$submits' == '2' ]]"

check "job1 は 0->2400 を fresh start で走る" \
    "grep -q 'resume=0 stop=2400 from=NONE' '$SANDBOX/run.log'"
check "job2 は 2400->4800" \
    "grep -q 'resume=2400 stop=4800' '$SANDBOX/run.log'"
check "job3 は 4800->7200" \
    "grep -q 'resume=4800 stop=7200' '$SANDBOX/run.log'"
check "max_steps は全ジョブで 7200 のまま(LR schedule の連続性)" \
    "[[ \$(grep -c 'max=7200' '$SANDBOX/run.log') == '3' ]]"
check "job2 は job1 のチェックポイントから再開する" \
    "grep -q 'from=.*job01/checkpoints/checkpoint_002400/consolidated/lora.safetensors' '$SANDBOX/run.log'"
check "job3 は job2 のチェックポイントから再開する" \
    "grep -q 'from=.*job02/checkpoints/checkpoint_004800/consolidated/lora.safetensors' '$SANDBOX/run.log'"
check "TOTAL_STEPS 到達で連鎖が止まる" \
    "grep -q 'Chain complete' '$SANDBOX/run.log'"
check "投入が暴走していない(上限 20 に達していない)" \
    "! grep -q 'submission storm' '$SANDBOX/run.log'"

# --- 2. 割り切れない TOTAL_STEPS ---------------------------------------------
echo ""
echo "[2] TOTAL_STEPS=5000 / STEPS_PER_JOB=2400 -> 3 ジョブ(最後は端数)"
setup_sandbox "$STUB_OK"
status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=5000 STEPS_PER_JOB=2400 CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "終了コード 0" "[[ '$status' == '0' ]]"
check "最終ジョブは 4800->5000 で打ち切られる" \
    "grep -q 'resume=4800 stop=5000' '$SANDBOX/run.log'"
check "stop_at_step が TOTAL_STEPS を超えない" \
    "! grep -qE 'stop=(5[1-9][0-9][0-9]|[6-9][0-9]{3})' '$SANDBOX/run.log'"

# --- 3. 早期終了 --------------------------------------------------------------
echo ""
echo "[3] 早期終了(stop_at_step のチェックポイントが無い)"
setup_sandbox "$STUB_EARLY_STOP"
status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=7200 STEPS_PER_JOB=2400 CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "終了コード 0(失敗ではなく正常終了)" "[[ '$status' == '0' ]]"
submits="$(grep -c '^SUBMIT ' "$SANDBOX/submit.log" || true)"
check "次のジョブを投げない" "[[ '$submits' == '0' ]]"
check "理由がログに出る" \
    "grep -q 'Not submitting the next job' '$SANDBOX/run.log'"

# --- 4. 即死検出(MIN_JOB_SEC) -------------------------------------------------
echo ""
echo "[4] 学習が一瞬で終わった場合の投入抑止"
setup_sandbox "$STUB_OK"
status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=7200 STEPS_PER_JOB=2400 MIN_JOB_SEC=600 PBS_JOBID=100.pbs)"
check "終了コード 1(異常として落とす)" "[[ '$status' == '1' ]]"
submits="$(grep -c '^SUBMIT ' "$SANDBOX/submit.log" || true)"
check "次のジョブを投げない" "[[ '$submits' == '0' ]]"
check "理由がログに出る" "grep -q 'MIN_JOB_SEC' '$SANDBOX/run.log'"

# --- 5. ジョブ数上限ガード ----------------------------------------------------
echo ""
echo "[5] CHAIN_INDEX が上限を超えたら走らない"
setup_sandbox "$STUB_OK"
status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=4800 STEPS_PER_JOB=2400 CHAIN_INDEX=5 CHAIN_ID=x RESUME_FROM=/dev/null CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "終了コード 1" "[[ '$status' == '1' ]]"
check "上限超過として弾く" \
    "grep -q 'exceeds MAX_CHAIN_JOBS' '$SANDBOX/run.log'"

# --- 6. 矛盾した指定を弾く ----------------------------------------------------
echo ""
echo "[6] 不正な指定の拒否"
setup_sandbox "$STUB_OK"

status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=2400 STEPS_PER_JOB=2400 CHAIN_INDEX=2 CHAIN_ID=x RESUME_FROM=/dev/null CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "走る余地が無い CHAIN_INDEX を拒否" "[[ '$status' == '1' ]]"

status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=7200 STEPS_PER_JOB=2400 CHAIN_INDEX=2 CHAIN_ID=x RESUME_FROM=/does/not/exist CHAIN_ALLOW_FAST=1 PBS_JOBID=100.pbs)"
check "存在しない RESUME_FROM を拒否" "[[ '$status' == '1' ]]"

status="$(run_chain EXP_NAME=e SRC_RUN_DIR=d TOTAL_STEPS=abc STEPS_PER_JOB=2400 PBS_JOBID=100.pbs)"
check "数値でない TOTAL_STEPS を拒否" "[[ '$status' == '1' ]]"

# --- 結果 ---------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "PASS: $PASS   FAIL: $FAIL"
echo "=================================================================="
if [[ "$FAIL" -gt 0 ]]; then
    echo ""
    echo "--- run.log (最後の 60 行) ---"
    tail -60 "$SANDBOX/run.log"
    exit 1
fi
echo "チェーンは想定通りに連結し、暴走ガードも効いています。"
