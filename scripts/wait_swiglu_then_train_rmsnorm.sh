#!/usr/bin/env bash
# 等待当前「A1+B1 + 仅 SwiGLU」训练结束，再自动启动「A1+B1 + 仅 RMSNorm」20k。
#
# 用法（仓库根目录）：
#   nohup bash scripts/wait_swiglu_then_train_rmsnorm.sh >> checkpoints/chain_swiglu_rmsnorm.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHAIN_LOG="${ROOT}/checkpoints/chain_swiglu_rmsnorm.log"
TRAIN_LOG="${ROOT}/checkpoints/train_a1b1_rmsnorm.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }

log "等待 SwiGLU 训练结束（匹配: train.py train ... --swiglu）..."
while pgrep -f 'python.*train\.py train.*--swiglu' >/dev/null 2>&1; do
  sleep 120
  if [ -f checkpoints/train_a1b1_swiglu.log ]; then
    tail -1 checkpoints/train_a1b1_swiglu.log 2>/dev/null | tee -a "$CHAIN_LOG" || true
  else
    # 若未 tee 到专用日志，仍提示在等
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 仍在训练 swiglu ..." >> "$CHAIN_LOG"
  fi
done

log "SwiGLU 训练已结束，等待 GPU 释放..."
sleep 30

log "开始 RMSNorm 单变量：--backbone a1b1 --rmsnorm（20k step）"
exec python train.py train \
  --data_root /root/autodl-tmp \
  --split manifest_full_out \
  --device cuda \
  --max_steps 20000 \
  --backbone a1b1 \
  --rmsnorm \
  2>&1 | tee -a "$TRAIN_LOG"
