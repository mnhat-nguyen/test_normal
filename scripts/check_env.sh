#!/usr/bin/env bash
# =============================================================================
# scripts/check_env.sh  –  Pre-flight check on each node
# =============================================================================
# Run this on every node before starting distributed training.
# It verifies Python, PyTorch, CUDA, NCCL, and port availability.
#
# Usage:
#   bash scripts/check_env.sh [master_ip] [master_port]
# =============================================================================
set -euo pipefail

MASTER_ADDR="${1:-192.168.0.1}"
MASTER_PORT="${2:-29500}"

OK=0
FAIL=0

pass() { echo "  [✓] $*"; OK=$((OK + 1)); }
fail() { echo "  [✗] $*" >&2; FAIL=$((FAIL + 1)); }
info() { echo "  [i] $*"; }

echo "======================================================"
echo " Pre-flight check  –  $(hostname)  ($(date))"
echo "======================================================"

# ── Python ────────────────────────────────────────────────────────────────────
echo ""
echo "[1] Python"
if python3 --version &>/dev/null; then
    pass "$(python3 --version)"
else
    fail "python3 not found. Install Python 3.9+"
fi

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
echo ""
echo "[2] PyTorch + CUDA"
PY_CHECK=$(python3 - <<'EOF'
import sys
try:
    import torch
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA    {torch.version.cuda}  (device: {torch.cuda.get_device_name(0)})")
        print(f"GPUs    {torch.cuda.device_count()}")
    else:
        print("CUDA    NOT available", file=sys.stderr)
        sys.exit(1)
except ImportError as e:
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)
EOF
) 2>&1
if echo "$PY_CHECK" | grep -q "CUDA NOT"; then
    fail "CUDA not available: $PY_CHECK"
elif echo "$PY_CHECK" | grep -q "ImportError"; then
    fail "PyTorch import error: $PY_CHECK"
else
    while IFS= read -r line; do pass "$line"; done <<< "$PY_CHECK"
fi

# ── NCCL ──────────────────────────────────────────────────────────────────────
echo ""
echo "[3] NCCL"
NCCL_VER=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "")
if [[ -n "$NCCL_VER" ]]; then
    pass "NCCL version: $NCCL_VER"
else
    fail "Cannot detect NCCL version (may still work; check torch.cuda.nccl)"
fi

# ── torchrun ──────────────────────────────────────────────────────────────────
echo ""
echo "[4] torchrun"
if command -v torchrun &>/dev/null; then
    pass "$(torchrun --version 2>&1 | head -1)"
else
    fail "torchrun not in PATH. Is PyTorch installed correctly?"
fi

# ── Network: ping master ──────────────────────────────────────────────────────
echo ""
echo "[5] Network → master ($MASTER_ADDR)"
if ping -c1 -W2 "$MASTER_ADDR" &>/dev/null; then
    pass "Ping to $MASTER_ADDR OK"
else
    fail "Cannot ping master $MASTER_ADDR – check networking / firewall"
fi

# ── Port check ────────────────────────────────────────────────────────────────
echo ""
echo "[6] Port $MASTER_PORT"
if [[ "$(hostname -I | awk '{print $1}')" == "$MASTER_ADDR" ]] || \
   [[ "$(hostname)" == "$MASTER_ADDR" ]]; then
    # We are the master: check port is free
    if ! ss -tlnp | grep -q ":${MASTER_PORT} "; then
        pass "Port $MASTER_PORT is free on this host (good, we are master)"
    else
        fail "Port $MASTER_PORT is already in use. Kill the process or change MASTER_PORT."
    fi
else
    # We are a worker: check we can reach master's port (skip if not yet open)
    info "Not the master node; port reachability will be checked at runtime."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Results: $OK passed, $FAIL failed"
echo "======================================================"
if [[ "$FAIL" -gt 0 ]]; then
    echo "Fix the $FAIL issue(s) above before running training." >&2
    exit 1
fi
echo " All checks passed. This node is ready."
