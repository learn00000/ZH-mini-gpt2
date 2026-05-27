#!/usr/bin/env bash
# 等待当前 B1（--rope --talking_heads）训练结束，再自动启动 B2（--rope --b2）。
# 用法（仓库根目录）：
#   nohup bash scripts/wait_b1_then_train_b2.sh >> checkpoints/chain_b1_b2.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHAIN_LOG="${ROOT}/checkpoints/chain_b1_b2.log"
B2_LOG="${ROOT}/checkpoints/train_b2.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }

log "等待 B1 进程结束（匹配: train.py train ... talking_heads）..."
while pgrep -f 'python.*train\.py train.*talking_heads' >/dev/null 2>&1; do
  sleep 120
  if [ -f checkpoints/train_a1b1.log ]; then
    tail -1 checkpoints/train_a1b1.log 2>/dev/null | tee -a "$CHAIN_LOG" || true
  fi
done

log "B1 已结束。等待 GPU 显存释放..."
sleep 30

log "开始 B2：RoPE + attn_dropout=0.2（约 20k step，预计 ~2h）"
exec python train.py train \
  --data_root /root/autodl-tmp \
  --split manifest_full_out \
  --device cuda \
  --max_steps 20000 \
  --rope \
  --b2 \
  2>&1 | tee -a "$B2_LOG"
