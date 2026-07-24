import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
train.py  –  DDP training with EWMA-MAD straggler mitigation + GSCM
=====================================================================
Single-process test (1 machine, 1 GPU):
    python train.py

4-node distributed (run on EACH of the 4 machines):
    torchrun \
        --nproc_per_node=1 \
        --nnodes=4 \
        --node_rank=<0|1|2|3> \
        --master_addr=<IP of node-0> \
        --master_port=29500 \
        train.py

torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically.
When run directly with python, those env vars are absent and
default to 0 / 0 / 1, so all distributed code is bypassed.
"""

import os
import time
import json
import datetime
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from config          import TrainConfig
from models          import get_model
from data            import get_dataloaders
from straggler       import StraglerDetector, GSCM
from straggler.gscm  import GSCM_SCALE
from sleep_injector  import SleepInjector
from utils           import get_logger


# ──────────────────────────────────────────────────────────────────────────────
# Distributed setup / teardown
# ──────────────────────────────────────────────────────────────────────────────

def setup(rank: int, world_size: int, config: TrainConfig) -> None:
    torch.cuda.set_device(0)
    if world_size > 1:
        os.environ.setdefault('MASTER_ADDR', config.master_addr)
        os.environ.setdefault('MASTER_PORT', config.master_port)
        dist.init_process_group(
            backend=config.backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=120),
        )


def cleanup(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Single training step
# ──────────────────────────────────────────────────────────────────────────────

def train_step(
    model,
    inputs:       torch.Tensor,
    targets:      torch.Tensor,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    scaler:       GradScaler,
    gscm:         GSCM,
    amp_active:   bool,
    device:       torch.device,
    injector=None,             # SleepInjector | None
    batch_idx:    int   = 0,
    last_x_t:     float = 0.0, # CLEAN (compute-only) x_t from previous batch
    world_size:   int   = 1,
):
    """
    Execute one full batch iteration and return (loss_value, outputs, X_t, injected_delay).

    X_t is the COMPUTATION time (ms) only:
        injected sleep (if any) + forward pass

    The timer STOPS BEFORE loss.backward(), so the entire backward — which
    in DDP fuses gradient computation with the all_reduce network sync — is
    EXCLUDED from X_t. This project targets COMPUTATION stragglers (matching
    Korel), not communication, so X_t must reflect compute, not the network
    all_reduce that would otherwise dominate iteration time on a slow
    interconnect and drown out the signal the detector watches.

    Sleep is injected INSIDE the timed window so a straggler node detects its
    OWN compute stall directly in its own X_t (self-detection).

    Returns
    -------
    loss_val        : float
    outputs         : torch.Tensor  (logits, still on device)
    x_t             : float         (forward + injected sleep, ms; excl. backward/all_reduce)
    injected_delay  : float         (seconds slept this step, 0.0 if none)
    """
    optimizer.zero_grad()

    # ── GSCM: agree on gradient scale BEFORE the timed section ───────────────
    global_scale = gscm.sync_scale(amp_active, scaler if amp_active else None)

    # ── Start timer (COMPUTE only — all_reduce excluded below) ───────────────
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── Sleep injection — INSIDE the timer, fed with CLEAN history ───────────
    injected_delay = 0.0
    if injector is not None:
        t_sleep_start  = time.perf_counter()
        injector.maybe_sleep(batch_idx, last_x_t)   # last_x_t is clean, no snowball
        injected_delay = time.perf_counter() - t_sleep_start   # seconds

    # ── Forward ───────────────────────────────────────────────────────────────
    if amp_active:
        with autocast('cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
    else:
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

    # ── Stop timer BEFORE backward — X_t excludes backward + all_reduce ──────
    # X_t measures forward + injected sleep only. The entire backward (which
    # in DDP fuses gradient compute with the all_reduce network sync) falls
    # outside the timer, so X_t reflects computation-straggler cost without
    # the communication cost this project handles separately.
    torch.cuda.synchronize()
    x_t = (time.perf_counter() - t_start) * 1000.0   # ms — forward + sleep only
#check this
    t_backward_start = time.perf_counter()
    # ── Backward + all_reduce happen AFTER the timer ─────────────────────────
    if amp_active:
        print(f"amp onbatch {batch_idx} : x_t {x_t:.3f}ms ")
        scaler.scale(loss).backward()
    else:
        scaled_loss = GSCM.scale_loss(loss, global_scale)
        scaled_loss.backward()

    allreduce_ms = (time.perf_counter() - t_backward_start) * 1000.0   # backward + all_reduce
    t_scaler_start = time.perf_counter()

    # ── Unscale ───────────────────────────────────────────────────────────────
    if amp_active:
        scaler.unscale_(optimizer)
    else:
        GSCM.unscale_gradients(model, global_scale)
    t_scaler = (time.perf_counter() - t_scaler_start) * 1000.0 #scaler time
    t_optimizer_start = time.perf_counter()
    # ── Optimizer step ────────────────────────────────────────────────────────
    if amp_active:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    t_optimizer = (time.perf_counter() - t_optimizer_start) * 1000.0 #optimizer time
    print(f" batch {batch_idx} : backward + all_reduce {allreduce_ms:.3f}ms ")
    print(f" batch {batch_idx} : scaler {t_scaler:.3f}ms ")
    print(f" batch {batch_idx} : optimizer {t_optimizer:.3f}ms ")
    return loss.item(), outputs, x_t, injected_delay


# ──────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler:    GradScaler,
    detector:  StraglerDetector,
    gscm:      GSCM,
    injector,                       # SleepInjector | None
    device:    torch.device,
    epoch:     int,
    config:    TrainConfig,
    logger,
    world_size: int = 1,
    amp_warmup: dict = None,        # ONE-TIME latch, owned by main(), persists
):
    """
    AMP warmup latch (one-time for the whole run)
    ---------------------------------------------
    amp_warmup is {'count': int, 'done': bool}, created ONCE in main() and
    passed in every epoch, so it is NOT reset per epoch.

    During warmup (before done):
      - detector.amp_flag is still read and detector.update() still runs, so
        the detector's window W stays protected from straggler-contaminated
        values exactly as designed.
      - BUT amp_active (whether AMP actually runs) is forced False, so no real
        FP16 compute happens and no cross-node desync is introduced during the
        unstable early phase.
      - Each batch the detector WANTS AMP increments count; at count >= 3 the
        latch flips done=True PERMANENTLY.

    After warmup (done): amp_active follows detector.amp_flag freely, forever.
    """
    if amp_warmup is None:
        amp_warmup = {'count': 0, 'done': True}   # no warmup if not provided

    model.train()
    total_loss = correct = total = 0
    n_batches  = len(loader)
    last_x_t   = 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        amp_wanted = detector.amp_flag        # ← ALWAYS the detector's real decision

        if amp_warmup['done']:
            amp_active = amp_wanted            # warmup over → AMP follows detector freely
        else:
            if amp_wanted:
                amp_warmup['count'] += 1
                if amp_warmup['count'] >= 3:
                    amp_warmup['done'] = True  # LATCH — permanent, never gates again
            amp_active = False                 # during warmup → AMP suppressed

        # Cold-start (epoch 0, batch 0): CUDA/cuDNN init outlier — skip
        # injector + detector update so neither is contaminated by the spike.
        is_cold_start = (epoch == 0 and i == 0)
        step_injector = None if is_cold_start else injector

        loss_val, outputs, x_t, injected_delay = train_step(
            model, inputs, targets,
            optimizer, criterion, scaler,
            gscm, amp_active, device,
            injector=step_injector,
            batch_idx=i,
            last_x_t=last_x_t,
            world_size=world_size,
        )
        last_x_t = x_t

        # Detector still updates normally (driven by amp_wanted via amp_flag),
        # so window protection works even during warmup.
        is_boundary = (i == 0) or (i == n_batches - 1)
        if not is_boundary and not is_cold_start:
            detector.update(x_t)   # full x_t, sleep included

        total_loss += loss_val
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % config.log_interval == 0:
            sleep_tag  = ' [SLEEP]' if (injector and injector.is_sleeping) else ''
            warmup_tag = '' if amp_warmup['done'] else f" [WARMUP {amp_warmup['count']}/3]"
            logger.info(
                f"Epoch {epoch:>3d} | Batch {i:>4d}/{n_batches} | "
                f"Loss {loss_val:.4f} | "
                f"AMP {'ON ' if amp_active else 'OFF'} | "
                f"X_t {x_t:>7.1f} ms | "
                f"Z {detector.Z or 0.0:>7.1f} | "
                f"UCL {detector.UCL if detector.UCL != float('inf') else 0.0:>7.1f} | "
                f"LCL {detector.LCL:>7.1f}"
                f"{sleep_tag}{warmup_tag}"
            )

    avg_loss = total_loss / n_batches
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler, detector, scaler, world_size):
    ckpt = torch.load(path, map_location='cpu')
    model_obj = model.module if world_size > 1 else model
    model_obj.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    detector.load_state_dict(ckpt['detector'])
    return ckpt['epoch']


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config: TrainConfig = None) -> None:
    if config is None:
        config = TrainConfig()

    rank       = int(os.environ.get('RANK',       0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup(rank, world_size, config)
    device = torch.device('cuda:0')
    logger = get_logger(rank)

    logger.info(f"Worker {rank}/{world_size} ready  |  device={device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(config.model_name, config.dataset).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[0])
    logger.info(f"Model : {config.model_name}  dataset : {config.dataset}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(config, rank, world_size)
    logger.info(
        f"Train batches: {len(train_loader)}  "
        f"Test batches: {len(test_loader)}"
    )

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(config.milestones), gamma=config.gamma
        )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(
        'cuda',
        init_scale=GSCM_SCALE,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1_000_000_000,  # never actually reaches next growth step
    )

    # ── Straggler detection ───────────────────────────────────────────────────
    detector = StraglerDetector(
        window_size = config.window_size,
        n_min       = config.n_min,
        k           = config.k,
        ewma_lambda = config.ewma_lambda,
    )
    gscm = GSCM(device)

    # ── Sleep injector (straggler simulation) ─────────────────────────────────
    injector = SleepInjector(
        prob_on            = config.sleep_prob_on,
        prob_off           = config.sleep_prob_off,
        check_interval     = config.sleep_check_interval,
        duration_ratio     = config.sleep_duration_ratio,
        seed               = config.sleep_seed,
    ) if config.inject_sleep else None

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    if config.resume and os.path.isfile(config.resume):
        start_epoch = load_checkpoint(
            config.resume, model, optimizer, scheduler, detector, scaler, world_size
        )
        logger.info(f"Resumed from {config.resume}  (epoch {start_epoch})")

    # ── Metrics tracking ──────────────────────────────────────────────────────
    os.makedirs(config.results_dir, exist_ok=True)
    metrics            = []
    cumulative_train_s = 0.0

    # One-time AMP warmup latch (persists across ALL epochs, never resets).
    amp_warmup = {'count': 0, 'done': False}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        epoch_t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, detector, gscm, injector,
            device, epoch, config, logger,
            world_size=world_size,
            amp_warmup=amp_warmup,
        )
        cumulative_train_s += time.perf_counter() - epoch_t0

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        metrics.append({
            'epoch':              epoch,
            'cumulative_train_s': round(cumulative_train_s, 3),
            'train_loss':         round(train_loss, 6),
            'train_acc':          round(train_acc,  4),
            'test_loss':          round(test_loss,  6),
            'test_acc':           round(test_acc,   4),
        })

        logger.info(
            f"── Epoch {epoch:>3d} summary  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  "
            f"total_train_time={cumulative_train_s:.1f}s"
        )

        if rank == 0:
            results_path = os.path.join(config.results_dir, 'algo_metrics.json')
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
            save_checkpoint(
                {
                    'epoch':     epoch + 1,
                    'model':     model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler':    scaler.state_dict(),
                    'detector':  detector.state_dict(),
                    'test_acc':  test_acc,
                },
                path=os.path.join(config.checkpoint_dir, f'algo_epoch_{epoch:03d}.pt'),
            )

    cleanup(world_size)


def parse_args():
    import argparse
    from models import list_models
    parser = argparse.ArgumentParser(description='DDP training with straggler mitigation')

    parser.add_argument('--model_name',   type=str,   help=f'Model name. Choices: {list_models()}')
    parser.add_argument('--dataset',      type=str,   choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_root',    type=str)

    parser.add_argument('--epochs',       type=int)
    parser.add_argument('--batch_size',   type=int)
    parser.add_argument('--lr',           type=float)
    parser.add_argument('--momentum',     type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--scheduler',    type=str,   choices=['cosine', 'multistep'])

    parser.add_argument('--window_size',  type=int)
    parser.add_argument('--n_min',        type=int)
    parser.add_argument('--k',            type=float)
    parser.add_argument('--ewma_lambda',  type=float)

    parser.add_argument('--inject_sleep',         type=lambda x: x.lower() == 'true')
    parser.add_argument('--sleep_prob_on',         type=float)
    parser.add_argument('--sleep_prob_off',        type=float)
    parser.add_argument('--sleep_check_interval',  type=int)
    parser.add_argument('--sleep_duration_ratio',  type=float)
    parser.add_argument('--sleep_seed',            type=int)

    parser.add_argument('--log_interval',   type=int)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--results_dir',    type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = TrainConfig()
    for key, val in vars(args).items():
        if val is not None and hasattr(config, key):
            setattr(config, key, val)
    main(config)