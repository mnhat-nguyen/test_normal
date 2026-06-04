"""
train.py  –  DDP training with EWMA-MAD straggler mitigation + GSCM
=====================================================================
Single-process test (1 machine, 1 GPU):
    python train.py

4-node distributed (run on EACH of the 4 machines):
    torchrun \\
        --nproc_per_node=1 \\
        --nnodes=4 \\
        --node_rank=<0|1|2|3> \\
        --master_addr=<IP of node-0> \\
        --master_port=29500 \\
        train.py

torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically.
When run directly with python, those env vars are absent and
default to 0 / 0 / 1, so all distributed code is bypassed.
"""

import os
import time
import json
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from config          import TrainConfig
from models          import get_model
from data            import get_dataloaders
from straggler       import StraglerDetector, GSCM, DetectionEvaluator
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
):
    """
    Execute one full batch iteration and return (loss_value, outputs, X_t).

    X_t is the wall-clock time (ms) for the complete step:
        forward  +  backward (incl. DDP allreduce)  +  optimizer step

    The GSCM scale sync happens BEFORE the timer starts so that
    communication overhead does not inflate X_t measurements.

    Returns
    -------
    loss_val : float
    outputs  : torch.Tensor  (logits, still on device)
    x_t      : float         (iteration time in ms)
    """
    optimizer.zero_grad()

    # ── GSCM: agree on gradient scale BEFORE the timed section ───────────────
    # All workers call dist.all_reduce here; Normal workers learn the AMP
    # scale so they can match it.  This is intentionally outside the timer.
    global_scale = gscm.sync_scale(amp_active, scaler if amp_active else None)

    # ── Start timer ───────────────────────────────────────────────────────────
    # X_t = time for forward + backward (allreduce) + optimizer step
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ── Forward ───────────────────────────────────────────────────────────────
    if amp_active:
        with autocast('cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        # GradScaler scales loss by global_scale internally (we aligned it
        # in sync_scale), so backward produces grads * global_scale
        scaler.scale(loss).backward()
    else:
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        # GSCM: manually match AMP workers' gradient magnitude
        # If no worker is in AMP mode global_scale == 1.0  →  no-op
        scaled_loss = GSCM.scale_loss(loss, global_scale)
        scaled_loss.backward()
        # ↑ DDP allreduce fires inside .backward() via registered hooks.
        # At this point all workers' gradients are averaged AND still
        # carry the factor global_scale.

    # ── Unscale ───────────────────────────────────────────────────────────────
    if amp_active:
        # scaler.unscale_() divides by global_scale (same value we aligned to)
        scaler.unscale_(optimizer)
    else:
        # Normal workers remove the manual scale factor
        GSCM.unscale_gradients(model, global_scale)

    # ── Optimizer step ────────────────────────────────────────────────────────
    if amp_active:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # ── Stop timer ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    x_t = (time.perf_counter() - t_start) * 1000.0   # ms

    return loss.item(), outputs, x_t


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
    evaluator: DetectionEvaluator,
    device:    torch.device,
    epoch:     int,
    config:    TrainConfig,
    logger,
):
    model.train()
    total_loss = correct = total = 0
    n_batches  = len(loader)
    last_x_t   = 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Inject artificial sleep BEFORE the timed step
        if injector is not None:
            injector.maybe_sleep(i, last_x_t)

        # Read current mode from detector (set by previous iteration's update)
        amp_active = detector.amp_flag

        loss_val, outputs, x_t = train_step(
            model, inputs, targets,
            optimizer, criterion, scaler,
            gscm, amp_active, device,
        )
        last_x_t = x_t

        # ── Detector update ───────────────────────────────────────────────────
        # Skip first and last batch of each epoch (as in the paper) because
        # those timings include data-loading warm-up / epoch-boundary effects.
        is_boundary = (i == 0) or (i == n_batches - 1)
        if not is_boundary:
            detector.update(x_t)
            if injector is not None:
                evaluator.record(
                    actually_sleeping=injector.is_sleeping,
                    amp_active=detector.amp_flag,
                )

        # ── Metrics ───────────────────────────────────────────────────────────
        total_loss += loss_val
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # ── Logging ───────────────────────────────────────────────────────────
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

def main() -> None:
    config     = TrainConfig()

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
    scaler = GradScaler('cuda')

    # ── Straggler detection ───────────────────────────────────────────────────
    detector = StraglerDetector(
        window_size = config.window_size,
        n_min       = config.n_min,
        k           = config.k,
        ewma_lambda = config.ewma_lambda,
    )
    gscm      = GSCM(device)
    evaluator = DetectionEvaluator()

    # ── Sleep injector (straggler simulation) ─────────────────────────────────
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
            device, epoch, config, logger,
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
