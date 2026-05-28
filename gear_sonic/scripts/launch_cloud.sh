#!/usr/bin/env bash
# launch_cloud.sh — 阿里云 dsw isaaclab:2.3.2 容器启动 GR00T 训练的标准包装。
#
# Why: 云端环境与本地差三件事,每次手敲都会忘:
#   1) PATH 里没 `python`,IsaacSim 的 python 在 /workspace/isaaclab/_isaac_sim/...
#      正确入口是 /workspace/isaaclab/isaaclab.sh -p(它会 source setup_python_env.sh)
#   2) headless=True 不能漏(没显示器,不带会卡 PhysX 开窗口)
#   3) ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
#      —— 云端 SMPL 数据路径与默认值不同
#
# 详见 D1_23DOF_DEBUG_LOG.md "[2026-05-27] 云端启动失败的两类陷阱"。
#
# Usage:
#   bash gear_sonic/scripts/launch_cloud.sh <exp_yaml_short> <exp_var> [extra hydra args...]
#
#   <exp_yaml_short> :  sonic_release.yaml 的短名,例如 "sonic_release"、"sonic_release_idle"、
#                       "sonic_release_23dof_h6_idle"。脚本会自动加前缀
#                       manager/universal_token/all_modes/。
#   <exp_var>        :  实验变体后缀,落到 logs_rl 目录名里。例如 "29dof_control"。
#   [extra args]     :  额外传给 hydra 的 override,如 num_envs=2048、+checkpoint=other.pt。
#
# 默认 warm-start checkpoint 是 sonic_release/last.pt。如要换 ckpt,
# 用 extra args 覆盖,例如:
#   bash gear_sonic/scripts/launch_cloud.sh sonic_release_idle 29dof_idle \
#        +checkpoint=logs_rl/.../last.pt
#
# 如要 from-scratch (不 warm-start) 训练,设 NO_CHECKPOINT=1:
#   NO_CHECKPOINT=1 bash gear_sonic/scripts/launch_cloud.sh \
#        sonic_release_23dof_h8_fromscratch h8_v1
# 脚本会跳过默认 +checkpoint=...,train_agent_trl.py:180 在 cfg.checkpoint
# is None 时直接 from random init 开训。
#
# Examples (这次 session 用过的):
#   bash gear_sonic/scripts/launch_cloud.sh sonic_release           29dof_control
#   bash gear_sonic/scripts/launch_cloud.sh sonic_release_idle      29dof_idle
#   bash gear_sonic/scripts/launch_cloud.sh sonic_release_23dof_h6_idle h6_resume
#
# 输出:
#   - 后台进程 PID 打印到 stdout
#   - 训练 stdout/stderr 写到 /tmp/<exp_var>.log
#   - 起完做一次 nvidia-smi 确认显存

set -euo pipefail

# --- 参数 ---
if [[ $# -lt 2 ]]; then
  sed -n '/^# Usage:/,/^# Examples/p' "$0" | sed 's/^# \?//'
  exit 1
fi
EXP_SHORT="$1"
EXP_VAR="$2"
shift 2
EXTRA_ARGS=("$@")

# --- 路径与解释器 ---
REPO_ROOT="${REPO_ROOT:-/mnt/workspace/lgy/GR00T-WholeBodyControl}"
ISAACLAB="${ISAACLAB:-/workspace/isaaclab/isaaclab.sh}"
LOG_FILE="${LOG_FILE:-/tmp/${EXP_VAR}.log}"

if [[ ! -x "$ISAACLAB" ]]; then
  echo "[launch_cloud] ERROR: $ISAACLAB 不存在或没执行权限。" >&2
  echo "[launch_cloud] 容器是不是变了?改 \$ISAACLAB 环境变量再试。" >&2
  exit 2
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[launch_cloud] ERROR: REPO_ROOT=$REPO_ROOT 不存在。" >&2
  exit 2
fi
cd "$REPO_ROOT"

# --- 预检 ---
echo "[launch_cloud] 预检..."

#  1. 检查同名 log 是否还在写(避免 nohup 覆盖正在跑的 run)
if [[ -f "$LOG_FILE" ]] && fuser "$LOG_FILE" >/dev/null 2>&1; then
  echo "[launch_cloud] ERROR: $LOG_FILE 还有进程在写,你是不是已经起过同名的 run?" >&2
  echo "[launch_cloud]        换 exp_var 或 kill 掉旧 run。" >&2
  exit 3
fi

#  2. 检查 yaml 文件存在
EXP_YAML="$REPO_ROOT/gear_sonic/config/exp/manager/universal_token/all_modes/${EXP_SHORT}.yaml"
if [[ ! -f "$EXP_YAML" ]]; then
  echo "[launch_cloud] ERROR: 找不到 $EXP_YAML" >&2
  echo "[launch_cloud]        是不是忘了 scp 到云端?(本地改完后)" >&2
  exit 4
fi

#  3. 检查 GPU 余量(粗略)
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  GPU_FREE=$((GPU_TOTAL - GPU_USED))
  echo "[launch_cloud] GPU memory: ${GPU_USED}/${GPU_TOTAL} MB used, ${GPU_FREE} MB free"
  if [[ $GPU_FREE -lt 16000 ]]; then
    echo "[launch_cloud] WARNING: 余量 <16GB,4096 envs 一个 run 大概要 15GB,可能会撞 OOM。" >&2
    echo "[launch_cloud]          继续 ctrl+C 取消,或加 num_envs=2048 之类的 override。"
    sleep 3
  fi
fi

#  4. 检查同时跑的 train_agent_trl 进程数
N_TRAIN=$(pgrep -fc "train_agent_trl.py" 2>/dev/null || echo 0)
if [[ ${N_TRAIN} -ge 2 ]]; then
  echo "[launch_cloud] WARNING: 已经有 ${N_TRAIN} 个训练进程在跑,加这个就 $((N_TRAIN+1)) 个,GPU 可能扛不住。" >&2
  sleep 3
fi

# --- 默认 hydra 参数(云端必加) ---
DEFAULT_ARGS=(
  "+exp=manager/universal_token/all_modes/${EXP_SHORT}"
  "headless=True"
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered"
  "exp_var=${EXP_VAR}"
)

# from-scratch 跳过默认 ckpt(NO_CHECKPOINT=1)。否则照常 warm-start。
if [[ "${NO_CHECKPOINT:-0}" == "1" ]]; then
  echo "[launch_cloud] NO_CHECKPOINT=1: 跳过默认 +checkpoint=sonic_release/last.pt,from-scratch 训练。"
else
  DEFAULT_ARGS+=("+checkpoint=sonic_release/last.pt")
fi

# 用户的 extra args 放最后,可以 override DEFAULT_ARGS(例如换 ckpt)
ALL_ARGS=("${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}")

echo "[launch_cloud] 启动:"
echo "  ISAACLAB:     $ISAACLAB -p"
echo "  Script:       gear_sonic/train_agent_trl.py"
echo "  Hydra args:"
for a in "${ALL_ARGS[@]}"; do echo "    $a"; done
echo "  Log:          $LOG_FILE"
echo ""

# --- 起 ---
nohup "$ISAACLAB" -p gear_sonic/train_agent_trl.py \
  "${ALL_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &
PID=$!
echo "[launch_cloud] PID=${PID}"

# --- 起来后 60s 看一眼 ---
sleep 60
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[launch_cloud] ERROR: PID ${PID} 已经死了,看 $LOG_FILE 末尾:" >&2
  tail -n 40 "$LOG_FILE" >&2
  exit 5
fi

echo "[launch_cloud] 启动 60s 后存活,最后 30 行:"
tail -n 30 "$LOG_FILE"
echo ""
echo "[launch_cloud] 持续看日志:  tail -F $LOG_FILE"
echo "[launch_cloud] kill 该 run:  kill -9 $PID"
