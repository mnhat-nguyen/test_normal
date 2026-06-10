#!/usr/bin/env bash
# =============================================================================
# scripts/launch_all.sh  –  Launch training on all 4 nodes from one terminal
# =============================================================================
# Prerequisites:
#   • Password-less SSH access to all 4 nodes (ssh-copy-id)
#   • The project is at PROJECT_DIR on every node (rsync or git clone)
#   • Python + PyTorch installed on every node (see README.md §1)
#
# Usage:
#   bash scripts/launch_all.sh
#
# Adjust NODE_IPS and PROJECT_DIR below before running.
# =============================================================================
set -euo pipefail

# ─── Cluster configuration ───────────────────────────────────────────────────
NODE_IPS=(
    "192.168.0.1"   # node 0  ← this is also MASTER_ADDR
    "192.168.0.2"   # node 1
    "192.168.0.3"   # node 2
    "192.168.0.4"   # node 3
)
SSH_USER="${SSH_USER:-$(whoami)}"         # SSH login user on each node
PROJECT_DIR="${PROJECT_DIR:-~/distributed_training}"   # path on each node
MASTER_ADDR="${NODE_IPS[0]}"
MASTER_PORT="29500"

LOG_DIR="./launch_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Launching 4-node DDP training (Ring AllReduce via NCCL)"
echo " Master : ${MASTER_ADDR}:${MASTER_PORT}"
echo " Logs   : ${LOG_DIR}/"
echo "============================================================"

# ─── Sync code to all nodes (optional) ───────────────────────────────────────
if command -v rsync &>/dev/null; then
    echo "[*] Syncing project to all nodes ..."
    for i in "${!NODE_IPS[@]}"; do
        ip="${NODE_IPS[$i]}"
        echo "    rsync → ${SSH_USER}@${ip}:${PROJECT_DIR}"
        rsync -az --exclude '.git' --exclude '__pycache__' --exclude 'pyddp-env' \
            "$(dirname "${BASH_SOURCE[0]}")/.." \
            "${SSH_USER}@${ip}:${PROJECT_DIR}" &
    done
    wait
    echo "[*] Sync done."
fi

# ─── Build the remote torchrun command ───────────────────────────────────────
build_cmd() {
    local rank=$1
    cat <<CMD
cd ${PROJECT_DIR} && \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
bash scripts/run.sh ${rank}
CMD
}

# ─── Launch all nodes in parallel ────────────────────────────────────────────
PIDS=()
for i in "${!NODE_IPS[@]}"; do
    ip="${NODE_IPS[$i]}"
    log_file="${LOG_DIR}/node${i}.log"
    echo "[*] Starting node ${i} on ${ip}  (log → ${log_file})"
    ssh -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" \
        "$(build_cmd "$i")" \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "[*] All nodes launched. Tailing node-0 log (Ctrl+C to detach):"
echo "    Full logs in ${LOG_DIR}/"
echo ""
tail -f "${LOG_DIR}/node0.log" &
TAIL_PID=$!

# ─── Wait for completion ──────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "[+] Node ${i} completed successfully."
    else
        echo "[!] Node ${i} FAILED (pid ${pid}). See ${LOG_DIR}/node${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

kill "$TAIL_PID" 2>/dev/null || true

if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: $FAILED node(s) failed." >&2
    exit 1
fi
echo ""
echo "[✓] Training complete. Results are on node-0 at: ${PROJECT_DIR}/results/"
