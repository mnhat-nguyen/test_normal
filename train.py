"""
train.py  –  entry point for both single-process testing and torchrun
======================================================================
Single-process test (1 machine, 1 GPU):
    python train.py

4-node distributed (run on EACH of the 4 machines via torchrun):
    torchrun \
        --nproc_per_node=1 \
        --nnodes=4 \
        --node_rank=<0|1|2|3> \
        --master_addr=<IP of node-0> \
        --master_port=29500 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=<IP of node-0>:29500 \
        train.py

Gradient synchronisation:
    PyTorch DDP uses NCCL backend, which implements Ring AllReduce
    for gradient averaging. Each of the 4 nodes holds 1 GPU, so
    WORLD_SIZE=4. The ring passes gradient chunks around 4 nodes
    in 2*(N-1) = 6 steps total (scatter-reduce + allgather).

torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically.
When run directly with python, those env vars are absent and
default to 0 / 0 / 1, so all distributed code is bypassed.

NaN fix summary
---------------
Three problems caused NaN loss in late epochs when AMP activates:

  1. Pre-backward loss check (sync across workers)
     If ANY worker has a non-finite loss after the forward pass, ALL
     workers skip backward. Without this, one worker with NaN loss
     still participates in DDP allreduce, spreading NaN to everyone.

  2. Post-unscale overflow sync (sync across workers)
     After unscaling gradients, if ANY worker has Inf/NaN gradients,
     ALL workers skip optimizer.step(). Without this, AMP workers
     skip silently while Normal workers still update, causing divergence.

  3. Gradient clipping
     Caps gradient norm before optimizer.step() to prevent late-epoch
     overflow where weights have grown large enough to push FP16 over
     its 65504 limit.

  4. Reduced GradScaler init_scale
     Default 65536 starts at the FP16 overflow boundary. Starting at
     256 gives the scaler room to grow safely over time.
"""

import os
import datetime
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from config              import TrainConfig
from models              import get_model
from data                import get_dataloaders
from straggler           import StraglerDetector, GSCM
from straggler.evaluator import DetectionEvaluator
from sleep_injector      import SleepInjector
from utils               import get_logger


# ──────────────────────────────────────────────────────────────────────────────
# Distributed setup / teardown
# ──────────────────────────────────────────────────────────────────────────────

def setup(rank: int, world_size: int, config: TrainConfig) -> None:
    torch.cuda.set_device(0)   # 1 GPU per node → always cuda:0

    if world_size > 1:
        os.environ.setdefault('MASTER_ADDR', config.master_addr)
        os.environ.setdefault('MASTER_PORT', config.master_port)
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')

        dist.init_process_group(
            backend    = config.backend,
            init_method= 'env://',
            rank       = rank,
            world_size = world_size,
            timeout    = datetime.timedelta(seconds=config.dist_timeout),
        )
        _warmup = torch.zeros(1, device='cuda:0')
        dist.all_reduce(_warmup, op=dist.ReduceOp.SUM)
        dist.barrier()


def cleanup(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Shared sync helper
# ──────────────────────────────────────────────────────────────────────────────

def _any_worker_flag(flag: bool, device: torch.device, world_size: int) -> bool:
    """
    Return True if ANY worker passes flag=True.
    Uses all_reduce(MAX) so all workers reach the same decision.
    Falls back gracefully when not distributed.
    """
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0.0


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
    world_size:   int,
):
    """
    Execute one full batch iteration.

    Returns
    -------
    loss_val : float   (nan when batch is skipped)
    outputs  : torch.Tensor
    x_t      : float   (ms; 0.0 when batch is skipped)
    skipped  : bool    (True when overflow was detected)
    """
    optimizer.zero_grad()

    # ── GSCM: agree on gradient scale BEFORE the timed section ───────────────
    global_scale = gscm.sync_scale(amp_active, scaler if amp_active else None)

    # ── Start timer ───────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── Forward pass ─────────────────────────────────────────────────────────
    if amp_active:
        with autocast('cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
    else:
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

    # ── Fix 1: Pre-backward loss validity check (synced across all workers) ──
    # If ANY worker has a non-finite loss (Inf or NaN from the forward pass),
    # ALL workers skip backward. This prevents NaN from entering the
    # computation graph or corrupting DDP allreduce.
    loss_bad = not torch.isfinite(loss)
    if _any_worker_flag(loss_bad, device, world_size):
        optimizer.zero_grad()
        if amp_active:
            scaler.update()            # shrink scale for next iteration
        torch.cuda.synchronize()
        x_t = (time.perf_counter() - t_start) * 1000.0
        return float('nan'), outputs, x_t, True

    # ── Backward ──────────────────────────────────────────────────────────────
    if amp_active:
        scaler.scale(loss).backward()
    else:
        scaled_loss = GSCM.scale_loss(loss, global_scale)
        scaled_loss.backward()

    # ── Unscale ───────────────────────────────────────────────────────────────
    if amp_active:
        scaler.unscale_(optimizer)
    else:
        GSCM.unscale_gradients(model, global_scale)

    # ── Fix 2: Post-unscale overflow sync (synced across all workers) ─────────
    # After unscaling, check if ANY worker has Inf/NaN in gradients.
    # If so, ALL workers skip optimizer.step() together.
    # Without this, AMP workers skip silently while Normal workers update,
    # causing model divergence that produces NaN in future batches.
    grad_bad = any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    if _any_worker_flag(grad_bad, device, world_size):
        optimizer.zero_grad()
        if amp_active:
            scaler.update()            # shrink scale
        torch.cuda.synchronize()
        x_t = (time.perf_counter() - t_start) * 1000.0
        return float('nan'), outputs, x_t, True

    # ── Fix 3: Gradient clipping ──────────────────────────────────────────────
    # Caps gradient norm before optimizer.step() to prevent late-epoch
    # overflow where weights have grown large enough for FP16 to overflow.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ── Optimizer step ────────────────────────────────────────────────────────
    if amp_active:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # ── Stop timer ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    x_t = (time.perf_counter() - t_start) * 1000.0   # ms

    return loss.item(), outputs, x_t, False


# ──────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler:     GradScaler,
    detector:   StraglerDetector,
    gscm:       GSCM,
    injector,
    evaluator:  DetectionEvaluator,
    device:     torch.device,
    epoch:      int,
    config:     TrainConfig,
    logger,
    world_size: int,
):
    model.train()
    total_loss      = 0.0
    correct = total = 0
    n_batches       = len(loader)
    skipped_batches = 0
    last_x_t        = 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if injector is not None:
            injector.maybe_sleep(i, last_x_t)

        amp_active = detector.amp_flag

        loss_val, outputs, x_t, skipped = train_step(
            model, inputs, targets,
            optimizer, criterion, scaler,
            gscm, amp_active, device, world_size,
        )

        # Skipped batch: do not update metrics or detector
        if skipped:
            skipped_batches += 1
            continue

        last_x_t = x_t

        is_boundary = (i == 0) or (i == n_batches - 1)
        if not is_boundary:
            detector.update(x_t)
            if injector is not None:
                evaluator.record(
                    actually_sleeping=injector.is_sleeping,
                    amp_active=detector.amp_flag,
                )

        total_loss += loss_val
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % config.log_interval == 0:
            sleep_tag = ' [SLEEP]' if (injector and injector.is_sleeping) else ''
            logger.info(
                f"Epoch {epoch:>3d} | Batch {i:>4d}/{n_batches} | "
                f"Loss {loss_val:.4f} | "
                f"AMP {'ON ' if amp_active else 'OFF'} | "
                f"X_t {x_t:>7.1f} ms | "
                f"Z {detector.Z or 0.0:>7.1f} | "
                f"UCL {detector.UCL if detector.UCL != float('inf') else 0.0:>7.1f} | "
                f"LCL {detector.LCL:>7.1f}"
                f"{sleep_tag}"
            )

    if skipped_batches > 0:
        logger.info(
            f"Epoch {epoch:>3d} | Skipped {skipped_batches}/{n_batches} "
            f"batches due to overflow"
        )

    valid_batches = n_batches - skipped_batches
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
    accuracy = 100.0 * correct / total    if total > 0        else 0.0
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


def load_checkpoint(path: str, model, optimizer, scheduler,
                    detector, scaler, world_size: int):
    ckpt      = torch.load(path, map_location='cpu')
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

def main() -> None:
    config     = TrainConfig()

    rank       = int(os.environ.get('RANK',       0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup(rank, world_size, config)
    device = torch.device('cuda:0')
    logger = get_logger(rank)

    logger.info(
        f"Worker {rank}/{world_size} ready | device={device} | "
        f"backend={config.backend} (Ring AllReduce) | "
        f"master={config.master_addr}:{config.master_port}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(config.model_name, config.dataset).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[0], find_unused_parameters=False)
    logger.info(f"Model : {config.model_name}  dataset : {config.dataset}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(config, rank, world_size)
    logger.info(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")

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

    # ── Fix 4: Reduced GradScaler init_scale ──────────────────────────────────
    # Default 65536 starts right at the FP16 overflow boundary (65504).
    # Starting at 256 gives the scaler room to grow safely; it increases
    # automatically every growth_interval steps when no overflow is detected.
    scaler = GradScaler('cuda', init_scale=256, growth_interval=100)

    # ── Straggler detection ───────────────────────────────────────────────────
    detector  = StraglerDetector(
        window_size = config.window_size,
        n_min       = config.n_min,
        k           = config.k,
        ewma_lambda = config.ewma_lambda,
    )
    gscm      = GSCM(device)
    evaluator = DetectionEvaluator()

    # ── Sleep injector ────────────────────────────────────────────────────────
    injector = SleepInjector(
        prob_on        = config.sleep_prob_on,
        prob_off       = config.sleep_prob_off,
        check_interval = config.sleep_check_interval,
        duration_ratio = config.sleep_duration_ratio,
        seed           = config.sleep_seed,
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

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        epoch_t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, detector, gscm, injector, evaluator,
            device, epoch, config, logger, world_size,
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

            det_m = evaluator.log_epoch(epoch)
            logger.info(
                f"   Detection — P={det_m['precision']:.3f}  "
                f"R={det_m['recall']:.3f}  F1={det_m['f1']:.3f}  "
                f"Acc={det_m['accuracy']:.3f}"
            )
            evaluator.save(config.results_dir)

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

    evaluator.report(logger)
    cleanup(world_size)


if __name__ == '__main__':
    main()