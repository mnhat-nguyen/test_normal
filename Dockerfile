# =============================================================================
# Dockerfile  –  Multi-node DDP training image (bare-metal, no k8s)
# =============================================================================
# Build:
#   docker build -t ddp-train:latest .
#
# Run on each node (substitute NODE_RANK and MASTER_ADDR):
#   docker run --gpus all --network host \
#       -e NODE_RANK=0 \
#       -e MASTER_ADDR=192.168.0.1 \
#       -e MASTER_PORT=29500 \
#       -v /path/to/data:/app/data \
#       -v /path/to/results:/app/results \
#       ddp-train:latest
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies (OpenSSH is useful for cluster debug; remove if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client \
        net-tools \
        iputils-ping \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source
COPY config.py train.py train_baseline.py sleep_injector.py ./
COPY models/    models/
COPY data/      data/
COPY straggler/ straggler/
COPY utils/     utils/
COPY scripts/   scripts/
RUN chmod +x scripts/*.sh

# ── Runtime entrypoint ────────────────────────────────────────────────────────
# NODE_RANK must be set by the caller (docker run -e NODE_RANK=...)
# MASTER_ADDR must also be set (docker run -e MASTER_ADDR=...)
# Optional: MASTER_PORT (default 29500), NCCL_SOCKET_IFNAME, NCCL_DEBUG
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
