#!/usr/bin/env bash
# =============================================================================
# scripts/run.sh  –  Launch training on a single node
# =============================================================================
# Usage:
#   bash scripts/run.sh <node_rank>   (0, 1, 2, or 3)
#
# Run this on EACH of the 4 nodes with its correct rank.
# Node 0 must be started last (or at the same time), because all
# other nodes rendezvous with node 0's address.
#
# Example:
#   On node-0 (192.168.0.1):  bash scripts/run.sh 0
#   On node-1 (192.168.0.2):  bash scripts/run.sh 1
#   On node-2 (192.168.0.3):  bash scripts/run.sh 2
#   On node-3 (192.168.0.4):  bash scripts/run.sh 3
# =============================================================================
set -euo pipefail

# ─── Cluster configuration ────────────────────────────────────────────────────
# Change MASTER_ADDR to the actual IP / hostname of node-0.
MASTER_ADDR="${MASTER_ADDR:-192.168.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES=4
NPROC_PER_NODE=1     # 1 GPU per machine

# ─── Argument: which node is this? ───────────────────────────────────────────
NODE_RANK="${1:?Usage: bash scripts/run.sh <node_rank>  (0, 1, 2, or 3)}"
if [[ "$NODE_RANK" -lt 0 || "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "ERROR: node_rank must be 0–$((NNODES-1)), got $NODE_RANK" >&2
    exit 1
fi

# ─── Optional NCCL tuning ────────────────────────────────────────────────────
# Uncomment and adjust for your network interface / IB setup:
# export NCCL_SOCKET_IFNAME=eth0       # restrict to a specific NIC
# export NCCL_IB_DISABLE=1             # disable InfiniBand if not available
# export NCCL_DEBUG=INFO               # verbose NCCL logging for debugging

# ─── Make sure we're in the project root ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "============================================================"
echo " Node rank  : ${NODE_RANK} / $((NNODES-1))"
echo " Master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs/node  : ${NPROC_PER_NODE}"
echo " World size : $((NNODES * NPROC_PER_NODE))"
echo " Backend    : nccl  (Ring AllReduce)"
echo "============================================================"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id=ddp_run_$(date +%s) \
    train.py
