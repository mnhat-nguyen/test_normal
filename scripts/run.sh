#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch scripts for 4-node distributed training
#
# Copy the whole project to each machine, then run:
#   bash scripts/run.sh <node_rank> [train|baseline]
#
# Example — algorithm run, node 0:
#   bash scripts/run.sh 0 train
#
# Example — baseline run, node 1:
#   bash scripts/run.sh 1 baseline
# ─────────────────────────────────────────────────────────────────────────────

MASTER_ADDR="192.168.0.11"      # ← node-0's actual IP (matches config.py)
MASTER_PORT="20015"
NNODES=4
NPROC_PER_NODE=1                # 1 GPU per machine

NODE_RANK=${1:?"Usage: bash scripts/run.sh <node_rank> [train|baseline]"}
SCRIPT=${2:-train}              # default to algorithm run

if [[ "$SCRIPT" == "baseline" ]]; then
    ENTRY="train_baseline.py"
else
    ENTRY="train.py"
fi

echo "Starting node rank ${NODE_RANK} — running ${ENTRY} ..."

NCCL_SOCKET_IFNAME=enp8s0 \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=enp8s0 \
NCCL_P2P_DISABLE=0 \
NCCL_TIMEOUT=7200 \
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${ENTRY}
