#!/bin/bash
# =============================================================================
# entrypoint.sh  –  Docker container entry point for bare-metal multi-node
# =============================================================================
# Required environment variables (set with docker run -e):
#   NODE_RANK    – this container's rank (0, 1, 2, or 3)
#   MASTER_ADDR  – IP / hostname of node-0
#
# Optional:
#   MASTER_PORT  (default: 29500)
#   NCCL_SOCKET_IFNAME  (e.g. eth0)
#   NCCL_IB_DISABLE     (1 if no InfiniBand)
# =============================================================================
set -euo pipefail

: "${NODE_RANK:?ERROR: NODE_RANK environment variable is required (0-3)}"
: "${MASTER_ADDR:?ERROR: MASTER_ADDR environment variable is required}"

export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES=4
export NPROC_PER_NODE=1
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# NCCL tuning defaults (can be overridden from outside)
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

echo "============================================================"
echo " Container start"
echo " NODE_RANK   : ${NODE_RANK}"
echo " MASTER_ADDR : ${MASTER_ADDR}:${MASTER_PORT}"
echo " WORLD_SIZE  : ${WORLD_SIZE}"
echo " Backend     : nccl (Ring AllReduce)"
echo "============================================================"

exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py
