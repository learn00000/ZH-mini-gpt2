#!/usr/bin/env bash
# 等待当前「GPT-2 + A1+B1 + 仅 SwiGLU」训练结束，再自动启动「GPT-2 + A1+B1 + 仅 RMSNorm」20k。
#
# 用法（仓库根目录，gpt 环境）：
#   nohup bash scripts/wait_gpt2_swiglu_then_rmsnorm.sh >> checkpoints/chain_gpt2_swiglu_rmsnorm.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/root/miniconda3/envs/gpt/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python)"
fi

CHAIN_LOG="${ROOT}/checkpoints/chain_gpt2_swiglu_rmsnorm.log"
TRAIN_LOG="${ROOT}/checkpoints/train_gpt2_a1b1_rmsnorm.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }

log "等待 GPT-2 SwiGLU 训练结束（匹配: train.py train ... gpt2 ... --swiglu）..."
while pgrep -f 'python.*train\.py train.*--tokenizer gpt2.*--swiglu' >/dev/null 2>&1 \
   || pgrep -f 'python.*train\.py train.*gpt2.*--swiglu' >/dev/null 2>&1; do
  sleep 120
  # 从最新 ckpt 目录的 loss 日志推断进度（若存在）
  latest="$(ls -td checkpoints/2026* 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" && -f "$latest/loss_history.json" ]]; then
    step="$("$PYTHON" -c "
import json,sys
d=json.load(open(sys.argv[1]))
s=d.get('train_log',{}).get('step',[])
print(s[-1] if s else '?')
" "$latest/loss_history.json" 2>/dev/null || echo "?")"
    log "仍在训练 swiglu，最近目录 $latest train_step≈$step/20000"
  else
    log "仍在训练 swiglu ..."
  fi
done

log "SwiGLU 训练已结束，等待 GPU 释放 30s..."
sleep 30

log "开始 GPT-2 RMSNorm 单变量：--tokenizer gpt2 --backbone a1b1 --rmsnorm（20k step）"
exec "$PYTHON" train.py train \
  --data_root /root/autodl-tmp \
  --split manifest_full_out \
  --tokenizer gpt2 \
  --device cuda \
  --max_steps 20000 \
  --backbone a1b1 \
  --rmsnorm \
  2>&1 | tee -a "$TRAIN_LOG"
