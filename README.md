# Distributed Training — 4-Node DDP with Ring AllReduce

PyTorch DDP training across **4 nodes × 1 GPU each** with NCCL Ring AllReduce
and real-time straggler detection via EWMA + AMP switching.

---

## How Ring AllReduce works here

NCCL (the backend under `dist.init_process_group(backend='nccl')`) automatically
implements **Ring AllReduce** for gradient averaging. With 4 nodes the ring has
`N=4` participants. Each backward pass triggers 2×(N-1) = **6 ring steps**:

```
scatter-reduce  : 3 steps  (each node accumulates 1/N of the gradient ring)
allgather       : 3 steps  (each accumulated chunk is broadcast back around)
```

No extra code is needed — DDP hooks into `.backward()` and NCCL handles the ring
topology. `find_unused_parameters=False` (set in `train.py`) avoids the overhead
of scanning parameters for gradient presence.

---

## Project structure

```
distributed_training/
├── train.py               ← entry point; torchrun target
├── train_baseline.py      ← baseline (no straggler mitigation)
├── config.py              ← all hyperparameters + env-var overrides
├── sleep_injector.py      ← straggler simulation
├── compare.py             ← compare algo vs baseline results
├── Dockerfile             ← containerised multi-node deployment
├── entrypoint.sh          ← Docker entry point (reads NODE_RANK env var)
├── requirements.txt
├── models/
│   └── resnet.py          ← ResNet-18/50/101/152 adapted for CIFAR
├── data/
│   └── dataloader.py      ← CIFAR-10/100 with DistributedSampler
├── straggler/
│   ├── detector.py        ← EWMA + MAD straggler detector
│   ├── gscm.py            ← Gradient Scale Consistency Mechanism
│   └── evaluator.py       ← detection precision/recall metrics
├── utils/
│   └── logger.py          ← rank-aware logger
└── scripts/
    ├── run.sh             ← launch one node (bash scripts/run.sh <rank>)
    ├── launch_all.sh      ← SSH-orchestrated launch on all 4 nodes
    └── check_env.sh       ← pre-flight environment check
```

---

## Quick-start (bare metal, no Docker)

### Step 1 — Identify your nodes

Note each machine's IP address. Node 0 is the **master** (rendezvous point).

| Node | IP            | Role   |
|------|---------------|--------|
| 0    | 192.168.0.1   | Master |
| 1    | 192.168.0.2   | Worker |
| 2    | 192.168.0.3   | Worker |
| 3    | 192.168.0.4   | Worker |

### Step 2 — Install dependencies on every node

```bash
# On every one of the 4 nodes:
git clone <your-repo-url> distributed_training
cd distributed_training
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch with CUDA 12.8 support:
```
torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Adjust `--index-url` if your CUDA version differs (e.g. `cu121` for CUDA 12.1).

### Step 3 — Configure the master address

Open `config.py` and set `master_addr` to **node-0's IP**:
```python
master_addr: str = '192.168.0.1'   # ← your actual node-0 IP
```
Or export it as an environment variable (no file edit needed):
```bash
export MASTER_ADDR=192.168.0.1
```

### Step 4 — Pre-flight check (optional but recommended)

Run on every node before training:
```bash
bash scripts/check_env.sh 192.168.0.1 29500
```
This verifies PyTorch, CUDA, NCCL, network reachability, and port availability.

### Step 5 — Open firewall port on node-0

Node-0 must accept inbound TCP on `MASTER_PORT` (default **29500**) from all
worker nodes. With `ufw`:
```bash
# On node-0 only:
sudo ufw allow 29500/tcp
```
Or with `iptables`:
```bash
sudo iptables -I INPUT -p tcp --dport 29500 -j ACCEPT
```

### Step 6 — Launch training

Run **one of the following on each node** (in any order — torchrun with c10d
rendezvous will wait for all 4 to connect before starting):

```bash
# On node-0  (rank 0):
MASTER_ADDR=192.168.0.1 bash scripts/run.sh 0

# On node-1  (rank 1):
MASTER_ADDR=192.168.0.1 bash scripts/run.sh 1

# On node-2  (rank 2):
MASTER_ADDR=192.168.0.1 bash scripts/run.sh 2

# On node-3  (rank 3):
MASTER_ADDR=192.168.0.1 bash scripts/run.sh 3
```

Or, if you have password-less SSH from a control machine, let the orchestration
script do all of this in one command:
```bash
# Edit NODE_IPS and SSH_USER in scripts/launch_all.sh, then:
bash scripts/launch_all.sh
```
Live logs appear in `./launch_logs_<timestamp>/node{0-3}.log`.

---

## Option B — Docker deployment

### Build the image (do this on every node, or push to a registry)

```bash
docker build -t ddp-train:latest .
```

### Run on each node

```bash
# Node 0 — master
docker run --rm --gpus all --network host \
    -e NODE_RANK=0 \
    -e MASTER_ADDR=192.168.0.1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/checkpoints:/app/checkpoints \
    ddp-train:latest

# Node 1
docker run --rm --gpus all --network host \
    -e NODE_RANK=1 \
    -e MASTER_ADDR=192.168.0.1 \
    -v $(pwd)/data:/app/data \
    ddp-train:latest

# Node 2
docker run --rm --gpus all --network host \
    -e NODE_RANK=2 \
    -e MASTER_ADDR=192.168.0.1 \
    -v $(pwd)/data:/app/data \
    ddp-train:latest

# Node 3
docker run --rm --gpus all --network host \
    -e NODE_RANK=3 \
    -e MASTER_ADDR=192.168.0.1 \
    -v $(pwd)/data:/app/data \
    ddp-train:latest
```

`--network host` is required so NCCL can bind to the host NIC for inter-node
communication. `--gpus all` exposes the GPU inside the container.

---

## Configuration reference

All fields in `config.py` can be overridden with an environment variable of the
same name (upper-cased):

```bash
# Example: larger batch, fewer epochs, different model
BATCH_SIZE=256 EPOCHS=20 MODEL_NAME=resnet18 bash scripts/run.sh 0
```

| Parameter         | Default       | Description                                   |
|-------------------|---------------|-----------------------------------------------|
| `MASTER_ADDR`     | `192.168.0.1` | IP / hostname of node-0                       |
| `MASTER_PORT`     | `29500`       | Rendezvous port (must be open on node-0)      |
| `DIST_TIMEOUT`    | `600`         | Seconds to wait for all workers during init   |
| `BACKEND`         | `nccl`        | Communication backend (use nccl for GPU)      |
| `MODEL_NAME`      | `resnet50`    | `resnet18 / 50 / 101 / 152`                   |
| `DATASET`         | `cifar10`     | `cifar10` or `cifar100`                       |
| `EPOCHS`          | `100`         | Total training epochs                         |
| `BATCH_SIZE`      | `128`         | Per-worker batch size (effective = 128×4=512) |
| `LR`              | `0.1`         | Initial learning rate                         |
| `INJECT_SLEEP`    | `True`        | Enable straggler simulation                   |
| `WINDOW_SIZE`     | `20`          | EWMA detector sliding window                  |
| `EWMA_LAMBDA`     | `0.3`         | EWMA smoothing factor λ                       |

---

## NCCL tuning for your network

Set these before launching if you hit connectivity issues:

```bash
# Restrict NCCL to a specific network interface (e.g. your fast interconnect)
export NCCL_SOCKET_IFNAME=eth0

# Disable InfiniBand if your cluster only has Ethernet
export NCCL_IB_DISABLE=1

# Enable verbose NCCL debug output (useful when workers can't connect)
export NCCL_DEBUG=INFO

# Surface async errors immediately instead of hanging
export NCCL_ASYNC_ERROR_HANDLING=1
```

---

## Expected output (node-0 log)

```
14:02:11 [rank 0] INFO  Worker 0/4 ready | device=cuda:0 | backend=nccl (Ring AllReduce) | master=192.168.0.1:29500
14:02:11 [rank 0] INFO  Model : resnet50  dataset : cifar10
14:02:11 [rank 0] INFO  Train batches: 98  Test batches: 20
14:02:13 [rank 0] INFO  Epoch   0 | Batch    0/ 98 | Loss 2.3101 | AMP OFF | X_t   342.1 ms | Z   342.1 | UCL     0.0 | LCL   0.0
...
14:08:55 [rank 0] INFO  ── Epoch   0 summary  train_loss=1.8234  train_acc=32.41%  test_loss=1.7102  test_acc=37.20%  lr=0.09995  total_train_time=402.1s
14:08:55 [rank 0] INFO     Detection — P=0.812  R=0.743  F1=0.776  Acc=0.901
```

Checkpoints are saved to `./checkpoints/algo_epoch_NNN.pt` by rank 0.
Metrics JSON is saved to `./results/algo_metrics.json`.

---

## Troubleshooting

**Workers hang at startup and never connect**
- Check that port 29500 is open on node-0: `nc -zv 192.168.0.1 29500`
- All nodes must be able to reach `MASTER_ADDR:MASTER_PORT`
- Try `NCCL_DEBUG=INFO` to see what NCCL is trying to do

**"Address already in use" on port 29500**
- A previous run didn't clean up: `kill $(lsof -t -i:29500)` on node-0
- Or change `MASTER_PORT` to a different free port on all nodes

**NCCL timeout / hang during allreduce**
- Check all 4 nodes can reach each other on high-numbered ports (NCCL uses
  random ephemeral ports for peer connections, not just 29500)
- `export NCCL_SOCKET_IFNAME=eth0` to force the right NIC

**"RuntimeError: CUDA error: no kernel image is available for execution"**
- PyTorch was built for a different CUDA version
- Reinstall with the correct `--index-url`: check `nvidia-smi` for your CUDA version

**One node crashes mid-training**
- DDP will hang on the surviving nodes waiting for the dead worker's allreduce
- Kill all `torchrun` processes (`pkill -f torchrun`) and restart from a checkpoint
- Set `DIST_TIMEOUT=120` to fail faster instead of hanging indefinitely
