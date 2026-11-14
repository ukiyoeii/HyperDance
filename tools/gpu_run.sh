#!/usr/bin/env bash
# ============================================================================
# 项目专用 GPU 启动脚本（共享服务器礼让规则的代码落实）
#
# 用法：
#   tools/gpu_run.sh [-n 1|2] [-N 任务名] -- <要运行的命令...>
#   tools/gpu_run.sh --check          # 只做检查并打印结果，不启动任何东西
#
# 规则：
#   0. 只允许物理 2 号、5 号卡（UUID 从本机环境变量配置并核对），其余卡一律不用。
#   1. 默认 1 张卡；单个任务最多 2 张；本项目所有并发任务合计最多 2 张。
#   2. 一张卡必须在连续多次采样中都满足：没有任何计算进程、显存占用极低、
#      利用率极低，才算空闲。有进程的卡即使利用率为 0 也不用。
#   3. 任何查询失败/结果无法解析 => 视为不可用（宁可不跑）。
#   4. 没有足够空闲卡 => 退出并报告，绝不抢占、不杀进程、不重置、不改计算模式。
#   5. 启动前一刻再检查一次，然后用 UUID 设置 CUDA_VISIBLE_DEVICES。
#   6. 所有检查快照、所选 UUID、命令、退出码写入 logs/gpu_launch/。
#
# 注意：这台机器没有 Slurm 等正式调度系统，这里的检查只是"礼让式"的，
#       不能保证独占——别人随时可以在同一张卡上启动任务。
# ============================================================================
set -uo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR=$PROJECT/logs/gpu_launch
LEASE_DIR=$LOG_DIR/active           # 记录本项目正在占用的卡（租约文件）
LOCK_FILE=$LOG_DIR/.select.lock
mkdir -p "$LOG_DIR" "$LEASE_DIR"

MAX_PROJECT_GPUS=2                  # 项目合计上限（硬编码，不提供参数放宽）
SAMPLES=${GPU_CHECK_SAMPLES:-5}     # 连续采样次数
INTERVAL=${GPU_CHECK_INTERVAL:-3}   # 采样间隔（秒）
MEM_MAX_MIB=${GPU_IDLE_MEM_MIB:-100}  # 空闲卡显存上限（空卡约 1 MiB）
UTIL_MAX=${GPU_IDLE_UTIL:-5}        # 空闲卡利用率上限（%）
# 允许使用的卡：只有物理 2 号和 5 号（2026-11-14 用户规定；旧项目约定尚未澄清，保守处理）。
# 配置"编号 -> UUID"对应关系：若某天编号与 UUID 对不上（换卡/重排），该卡直接视为不可用。
# 其余卡即使空闲也不使用；仅能配置这两个允许编号的 UUID，不能放宽数量上限。
declare -A ALLOWED_UUID=(
  [2]="${GPU_UUID_2:-}"
  [5]="${GPU_UUID_5:-}"
)
PREFER="2 5"

NUM=1; NAME=job; CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n) NUM=$2; shift 2;;
    -N) NAME=$2; shift 2;;
    --check) CHECK_ONLY=1; shift;;
    --) shift; break;;
    *) echo "未知参数: $1" >&2; exit 2;;
  esac
done
case "$NUM" in 1|2) ;; *) echo "[gpu_run] 拒绝：-n 只能是 1 或 2" >&2; exit 2;; esac
if [ $CHECK_ONLY -eq 0 ] && [ $# -eq 0 ]; then echo "[gpu_run] 缺少要运行的命令" >&2; exit 2; fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/${STAMP}_${NAME}_$$.log
log() { echo "[gpu_run $(date '+%F %T')] $*" | tee -a "$LOG"; }

log "用户=$(whoami) 主机=$(hostname) 任务=$NAME 申请GPU数=$NUM"
log "命令: $*"

# 正式调度系统检测：有则应优先走调度系统，本脚本不绕过它。
for s in sbatch qsub bsub condor_submit; do
  if command -v $s >/dev/null 2>&1; then
    log "检测到调度系统 ($s)。请通过调度系统申请 GPU，本脚本拒绝直接启动。"; exit 3
  fi
done

# ---- 采样一次：输出当前"看起来空闲"的 UUID 列表；失败返回非 0 ----
snapshot() {
  local gpus apps
  gpus=$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,compute_mode \
          --format=csv,noheader,nounits 2>>"$LOG") || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
          --format=csv,noheader,nounits 2>>"$LOG") || return 1
  [ -n "$gpus" ] || return 1
  {
    echo "--- 快照 $(date '+%F %T') ---"
    echo "$gpus"
    echo "计算进程: ${apps:-（无）}"
    # 权限允许时记录进程所属用户
    echo "$apps" | awk -F', *' 'NF>=2{print $2}' | while read -r pid; do
      [ -n "$pid" ] && ps -o user=,pid=,etime=,comm= -p "$pid" 2>/dev/null | sed 's/^/  进程属主: /'
    done
  } >>"$LOG"
  echo "$gpus" | while IFS=, read -r idx uuid mem util mode; do
    idx=${idx// /}; uuid=${uuid// /}; mem=${mem// /}; util=${util// /}
    # 数值无法解析（如 [N/A]）=> 视为不可用
    [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$uuid" == GPU-* ]] || continue
    [[ "$idx" =~ ^[0-9]+$ ]] || continue
    [ "${ALLOWED_UUID[$idx]:-}" = "$uuid" ] || continue   # 不在白名单或 UUID 不符 => 不用
    echo "$apps" | grep -qF "$uuid" && continue          # 有任何计算进程 => 不用
    [ "$mem" -le "$MEM_MAX_MIB" ] || continue             # 显存被占（含非计算进程）=> 不用
    [ "$util" -le "$UTIL_MAX" ] || continue
    echo "$idx $uuid"
  done
  return 0
}

# ---- 进程是否仍是"同一个"进程：PID 存在且启动时刻一致（防止 PID 被系统复用后误判）----
proc_start() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
same_proc() { [ -n "$1" ] && [ -n "$2" ] && [ "$(proc_start "$1")" = "$2" ]; }

# ---- 清理失效租约，返回本项目当前占用的 UUID ----
# 租约有效 = 启动脚本本身或它启动的命令仍在运行。两者都不在了才删除租约，
# 这样即使启动脚本被 kill -9（来不及清理）而训练进程还活着，名额仍然算被占用。
leased_uuids() {
  local f
  for f in "$LEASE_DIR"/*.lease; do
    [ -e "$f" ] || continue
    if same_proc "$(sed -n 's/^pid=//p' "$f")" "$(sed -n 's/^pid_start=//p' "$f")" || \
       same_proc "$(sed -n 's/^child=//p' "$f")" "$(sed -n 's/^child_start=//p' "$f")"; then
      sed -n 's/^uuid=//p' "$f"
    else
      rm -f "$f"
    fi
  done
}

# ---- 第二道保险：直接向 nvidia-smi 询问"当前用户的进程实际占着哪些卡" ----
# 不依赖租约文件；即使租约丢失，也不会让本用户合计超过上限。
# （会把本用户其他项目的 GPU 进程也算进来，偏保守。）
my_busy_uuids() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>>"$LOG") || return 1
  echo "$apps" | while IFS=, read -r uuid pid; do
    uuid=${uuid// /}; pid=${pid// /}
    [ -n "$pid" ] || continue
    [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(whoami)" ] && echo "$uuid"
  done
  return 0
}

# ---- 连续采样，取每次都空闲的卡的交集 ----
stable_free() {
  local i cur free=""
  for i in $(seq 1 "$SAMPLES"); do
    cur=$(snapshot) || { log "nvidia-smi 查询失败 => 全部视为不可用"; return 1; }
    if [ "$i" -eq 1 ]; then free="$cur"
    else free=$(comm -12 <(echo "$free" | sort) <(echo "$cur" | sort)); fi
    [ -n "$free" ] || break
    [ "$i" -lt "$SAMPLES" ] && sleep "$INTERVAL"
  done
  echo "$free"
}

exec 9>"$LOCK_FILE"
flock 9     # 串行化本项目内的选卡，避免两个任务同时选中同一张卡

MYBUSY=$(my_busy_uuids) || { log "nvidia-smi 查询失败 => 不启动"; exit 5; }
HELD=$( { leased_uuids; echo "$MYBUSY"; } | grep . | sort -u || true)
HELD_N=$(echo -n "$HELD" | grep -c . || true)
log "本项目/本用户当前已占用 GPU 数: $HELD_N（租约 + 实际进程）"
if [ $CHECK_ONLY -eq 0 ] && [ $((HELD_N + NUM)) -gt $MAX_PROJECT_GPUS ]; then
  log "拒绝：项目合计将超过 $MAX_PROJECT_GPUS 张（已占 $HELD_N，申请 $NUM）。"; exit 4
fi

log "连续采样 $SAMPLES 次（间隔 ${INTERVAL}s）确认空闲……"
FREE=$(stable_free) || exit 5
# 去掉本项目自己已占用的卡
if [ -n "$HELD" ]; then FREE=$(echo "$FREE" | grep -vFf <(echo "$HELD") || true); fi
log "稳定空闲的卡（编号 UUID）: $(echo "$FREE" | tr '\n' ';')"

# 按偏好顺序挑选
CHOSEN=""; COUNT=0
for p in $PREFER; do
  [ $COUNT -lt "$NUM" ] || break
  line=$(echo "$FREE" | awk -v p="$p" '$1==p{print $2}')
  [ -n "$line" ] && { CHOSEN="$CHOSEN $line"; COUNT=$((COUNT+1)); }
done
CHOSEN=$(echo $CHOSEN)

if [ $CHECK_ONLY -eq 1 ]; then
  log "仅检查模式：若现在启动将选择: ${CHOSEN:-（无可用卡）}"; echo "详细日志: $LOG"; exit 0
fi
if [ $COUNT -lt "$NUM" ]; then
  log "没有足够的空闲 GPU（需要 $NUM，找到 $COUNT）。停止，不抢占。"; exit 6
fi

# 启动前一刻再检查一次
FINAL=$(snapshot) || { log "最终检查查询失败 => 不启动"; exit 5; }
for u in $CHOSEN; do
  echo "$FINAL" | grep -qF "$u" || { log "最终检查时 $u 已不空闲 => 不启动。"; exit 6; }
done

UUIDS=$(echo "$CHOSEN" | tr ' ' ',')
LEASE=$LEASE_DIR/$$.lease
# 租约在仍持有锁时写入（先写临时文件再改名，保证别的任务看到的一定是完整租约）
{ echo "pid=$$"; echo "pid_start=$(proc_start $$)"; echo "name=$NAME"; echo "start=$STAMP"
  for u in $CHOSEN; do echo "uuid=$u"; done; } >"$LEASE.tmp" && mv "$LEASE.tmp" "$LEASE" \
  || { log "租约写入失败 => 不启动"; exit 7; }
CHILD=""
cleanup() {   # 正常结束、报错、被 Ctrl-C/TERM/HUP 中断时都会执行
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then return; fi   # 子进程还活着就保留租约
  rm -f "$LEASE" "$LEASE.tmp"
}
trap cleanup EXIT
flock -u 9; exec 9>&-

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$UUIDS
export DANCE_NUM_GPUS=$NUM          # 分布式训练的进程数必须等于它（最多 2）
log "已选 GPU UUID: $UUIDS"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  DANCE_NUM_GPUS=$DANCE_NUM_GPUS"
log "提示：空闲检查不等于独占，他人仍可能随后使用同一张卡。"
log "启动命令……"

"$@" &
CHILD=$!
{ echo "child=$CHILD"; echo "child_start=$(proc_start $CHILD)"; } >>"$LEASE"
# 收到中断信号时转发给子进程，让它自己退出，然后再释放租约
trap 'log "收到中断信号，转发给子进程 $CHILD"; kill -TERM "$CHILD" 2>/dev/null' TERM INT HUP
RC=0
while :; do
  wait "$CHILD"; RC=$?
  kill -0 "$CHILD" 2>/dev/null || break     # wait 可能被信号打断，子进程真正结束才跳出
done
log "命令结束，退出码=$RC"
exit $RC
