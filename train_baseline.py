"""
train_baseline.py  –  baseline training without straggler detection
====================================================================
Identical training setup to train.py but:
  • No StraglerDetector — always runs in FP32
  • No GSCM            — no gradient scale sync needed
  • Same SleepInjector with the same seed as train.py so both runs
    face identical straggler conditions

Saves per-epoch metrics to results/baseline_metrics.json for comparison.

Run:
    python train_baseline.py
"""

import os
import time
import json
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config          import TrainConfig
from models          import get_model
from data            import get_dataloaders
from sleep_injector  import SleepInjector
from utils           import get_logger


# ──────────────────────────────────────────────────────────────────────────────
# Distributed setup / teardown (same guards as train.py)
# ──────────────────────────────────────────────────────────────────────────────

def setup(rank: int, world_size: int, config: TrainConfig) -> None:
    torch.cuda.set_device(0)
    if world_size > 1:
        os.environ.setdefault('MASTER_ADDR', config.master_addr)
        os.environ.setdefault('MASTER_PORT', config.master_port)
        dist.init_process_group(backend=config.backend,
                                rank=rank, world_size=world_size)


def cleanup(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Single training step  —  plain FP32, no AMP, no GSCM
# ──────────────────────────────────────────────────────────────────────────────

def train_step(model, inputs, targets, optimizer, criterion,
               injector=None, batch_idx=0, last_x_t=0.0, world_size=1):
    """
    One full batch: forward + backward-compute + optimizer step.

    X_t = COMPUTATION time (ms) only — forward + backward-compute + step +
    injected sleep. The DDP all_reduce (network) is EXCLUDED via
    model.no_sync() and triggered separately outside the timer, identical
    to train.py, so both scripts' X_t measure the same thing (compute, not
    communication). This project targets COMPUTATION stragglers.

    Returns
    -------
    loss_val, outputs, x_t (compute ms, excl. all_reduce), injected_delay
    """
    optimizer.zero_grad()

    # ── Start timer (COMPUTE only) ────────────────────────────────────────────
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── Sleep injection — INSIDE the timer (matches train.py) ────────────────
    injected_delay = 0.0
    if injector is not None:
        t_sleep_start  = time.perf_counter()
        injector.maybe_sleep(batch_idx, last_x_t)
        injected_delay = time.perf_counter() - t_sleep_start   # seconds

    # ── Forward ───────────────────────────────────────────────────────────────
    outputs = model(inputs)
    loss    = criterion(outputs, targets)

    # ── Backward WITHOUT all_reduce (deferred via no_sync) ───────────────────
    sync_ctx = model.no_sync() if world_size > 1 else contextlib.nullcontext()
    with sync_ctx:
        loss.backward()

    # ── Stop timer — X_t = compute + sleep, NO all_reduce ────────────────────
    torch.cuda.synchronize()
    x_t = (time.perf_counter() - t_start) * 1000.0   # ms — COMPUTE only

    # ── all_reduce OUTSIDE the timer (communication, excluded from X_t) ──────
    if world_size > 1:
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

    optimizer.step()

    return loss.item(), outputs, x_t, injected_delay


# ──────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion,
                injector, device, epoch, config, logger, world_size=1):
    model.train()
    total_loss = correct = total = 0
    n_batches  = len(loader)
    last_x_t   = 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Cold-start skip (epoch 0, batch 0): CUDA/cuDNN init outlier — don't
        # let it feed the injector. Handled by passing injector=None for it.
        is_cold_start = (epoch == 0 and i == 0)
        step_injector = None if is_cold_start else injector

        loss_val, outputs, x_t, injected_delay = train_step(
            model, inputs, targets, optimizer, criterion,
            injector=step_injector,
            batch_idx=i,
            last_x_t=last_x_t,
            world_size=world_size,
        )
        # Strip injected sleep so the injector's feedback stays clean —
        # identical to train.py, prevents sleep-duration snowball.
        last_x_t = x_t - (injected_delay * 1000.0)

        total_loss += loss_val
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % config.log_interval == 0:
            sleep_tag = ' [SLEEP]' if (injector and injector.is_sleeping) else ''
            logger.info(
                f"Epoch {epoch:>3d} | Batch {i:>4d}/{n_batches} | "
                f"Loss {loss_val:.4f} | "
                f"X_t {x_t:>7.1f} ms"
                f"{sleep_tag}"
            )

    return total_loss / n_batches, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config     = TrainConfig()
    rank       = int(os.environ.get('RANK',       0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup(rank, world_size, config)
    device = torch.device('cuda:0')
    logger = get_logger(rank, log_dir='./logs_baseline')

    logger.info(f"[BASELINE] Worker {rank}/{world_size} | device={device}")
    logger.info(f"Model: {config.model_name}  Dataset: {config.dataset}  "
                f"Sleep injection: {config.inject_sleep}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(config.model_name, config.dataset).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[0])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(config, rank, world_size)

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=config.lr,
                                 momentum=config.momentum,
                                 weight_decay=config.weight_decay)

    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(config.milestones), gamma=config.gamma)

    # ── Sleep injector ────────────────────────────────────────────────────────
    # Use the SAME seed as train.py so both face identical straggler patterns
    injector = SleepInjector(
        prob_on            = config.sleep_prob_on,
        prob_off           = config.sleep_prob_off,
        check_interval     = config.sleep_check_interval,
        duration_ratio     = config.sleep_duration_ratio,
        seed               = config.sleep_seed,
    ) if config.inject_sleep else None

    # ── Metrics tracking ──────────────────────────────────────────────────────
    os.makedirs(config.results_dir, exist_ok=True)
    metrics            = []
    cumulative_train_s = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(config.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        epoch_t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            injector, device, epoch, config, logger,
            world_size=world_size,
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
            f"total_train_time={cumulative_train_s:.1f}s"
        )

        if rank == 0:
            # Save metrics after every epoch
            results_path = os.path.join(config.results_dir, 'baseline_metrics.json')
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
            save_checkpoint(
                {
                    'epoch':    epoch + 1,
                    'model':    model_state,
                    'test_acc': test_acc,
                },
                path=os.path.join(config.checkpoint_dir,
                                  f'baseline_epoch_{epoch:03d}.pt'),
            )

    cleanup(world_size)


if __name__ == '__main__':
    main()