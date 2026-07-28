"""
config.py – all hyperparameters in one place.

Every field can be overridden by an environment variable of the same name
(upper-cased), so you can tune runs without editing this file:

    MASTER_ADDR=10.0.0.1 EPOCHS=50 bash scripts/run.sh 0
"""
import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default):
    """Return env var cast to the same type as *default*, or *default*."""
    val = os.environ.get(key.upper())
    if val is None:
        return default
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


@dataclass
class TrainConfig:
    # ── Distributed ──────────────────────────────────────────────────────────
    # backend 'nccl' uses NCCL Ring AllReduce for gradient synchronisation.
    # Each of the 4 nodes holds 1 GPU; WORLD_SIZE=4, ring has 4 participants.
    backend:      str = field(default_factory=lambda: _env('BACKEND',     'nccl'))
    master_addr:  str = field(default_factory=lambda: _env('MASTER_ADDR', '192.168.0.1'))
    master_port:  str = field(default_factory=lambda: _env('MASTER_PORT', '29500'))
    # seconds to wait for all workers during init / barrier; raise if cluster is slow
    dist_timeout: int = field(default_factory=lambda: _env('DIST_TIMEOUT', 600))

    # ── Model / Dataset ───────────────────────────────────────────────────────
    model_name:   str = field(default_factory=lambda: _env('MODEL_NAME', 'resnet50'))
    dataset:      str = field(default_factory=lambda: _env('DATASET',    'cifar10'))
    data_root:    str = field(default_factory=lambda: _env('DATA_ROOT',  './data'))
    num_workers:  int = field(default_factory=lambda: _env('NUM_WORKERS', 4))

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:       int   = field(default_factory=lambda: _env('EPOCHS',       50))
    batch_size:   int   = field(default_factory=lambda: _env('BATCH_SIZE',   256))
    # Gradient accumulation: all_reduce + optimizer step fire once per this
    # many micro-batches → ~N× less network communication. Effective batch
    # size = batch_size × accum_steps × world_size.
    accum_steps:  int   = field(default_factory=lambda: _env('ACCUM_STEPS',  8))
    lr:           float = field(default_factory=lambda: _env('LR',           0.1))
    momentum:     float = field(default_factory=lambda: _env('MOMENTUM',     0.9))
    weight_decay: float = field(default_factory=lambda: _env('WEIGHT_DECAY', 5e-4))

    # ── LR Schedule ───────────────────────────────────────────────────────────
    scheduler:    str   = field(default_factory=lambda: _env('SCHEDULER', 'cosine'))
    milestones:   tuple = (100, 150)
    gamma:        float = field(default_factory=lambda: _env('GAMMA', 0.1))

    # ── Straggler Detection ───────────────────────────────────────────────────
    window_size:  int   = field(default_factory=lambda: _env('WINDOW_SIZE', 30))
    n_min:        int   = field(default_factory=lambda: _env('N_MIN',       10))
    k:            float = field(default_factory=lambda: _env('K',           2.5))
    ewma_lambda:  float = field(default_factory=lambda: _env('EWMA_LAMBDA', 0.15))

    # ── Sleep injection ───────────────────────────────────────────────────────
    inject_sleep:         bool  = field(default_factory=lambda: _env('INJECT_SLEEP',         True))
    sleep_prob_on:        float = field(default_factory=lambda: _env('SLEEP_PROB_ON',        0.50))
    sleep_prob_off:       float = field(default_factory=lambda: _env('SLEEP_PROB_OFF',       0.35))
    sleep_check_interval: int   = field(default_factory=lambda: _env('SLEEP_CHECK_INTERVAL', 10))
    sleep_duration_ratio: float = field(default_factory=lambda: _env('SLEEP_DURATION_RATIO', 0.08))
    # FIXED absolute sleep per straggle batch (ms). Used instead of
    # duration_ratio so BOTH train.py and baseline inject identical sleep,
    # making the speedup comparison fair (see sleep_injector.py).
    sleep_ms:             float = field(default_factory=lambda: _env('SLEEP_MS', 500.0))
    sleep_seed:           int   = field(default_factory=lambda: _env('SLEEP_SEED',           42))
    # Number of clean (non-sleeping) iterations used to calibrate the FROZEN
    # reference_ms that sleep duration is based on — see sleep_injector.py.
    # Must be reached before any sleep event can have a real duration.
    sleep_calibration_window: int = field(default_factory=lambda: _env('SLEEP_CALIBRATION_WINDOW', 10))

    # ── Logging / Checkpointing ───────────────────────────────────────────────
    log_interval:     int           = field(default_factory=lambda: _env('LOG_INTERVAL', 20))
    checkpoint_dir:   str           = field(default_factory=lambda: _env('CHECKPOINT_DIR', './checkpoints'))
    results_dir:      str           = field(default_factory=lambda: _env('RESULTS_DIR',    './results'))
    resume:           Optional[str] = None