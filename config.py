from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # ── Distributed ──────────────────────────────────────────────────────────
    backend:      str = 'nccl'
    master_addr:  str = '192.168.0.11'  # node-0 IP
    master_port:  str = '20015'

    # ── Model / Dataset ───────────────────────────────────────────────────────
    # ResNet      : resnet18 | resnet34 | resnet50 | resnet101 | resnet152
    # VGG         : vgg11 | vgg13 | vgg16 | vgg19
    # DenseNet    : densenet121 | densenet161 | densenet169 | densenet201
    # EfficientNet: efficientnet_b0 | efficientnet_b1 | efficientnet_b2 | efficientnet_b3
    # MobileNet   : mobilenet_v2 | mobilenet_v3_small | mobilenet_v3_large
    model_name:   str = 'resnet50'
    dataset:      str = 'cifar10'       # cifar10 | cifar100
    data_root:    str = './data'
    num_workers:  int = 4

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:       int   = 100
    batch_size:   int   = 128           # per-worker batch size
    lr:           float = 0.1
    momentum:     float = 0.9
    weight_decay: float = 5e-4

    # ── LR Schedule ───────────────────────────────────────────────────────────
    scheduler:    str   = 'cosine'      # cosine | multistep
    milestones:   tuple = (100, 150)    # used only when scheduler='multistep'
    gamma:        float = 0.1

    # ── Straggler Detection ───────────────────────────────────────────────────
    #   X_t = wall-clock time (ms) for one full batch:
    #         forward + backward + DDP allreduce + optimizer step
    window_size:  int   = 20            # N      : sliding window max size
    n_min:        int   = 10            # N_min  : cold-start threshold
    k:            float = 3.0           # MAD scaling factor (shared for UCL / LCL)
    ewma_lambda:  float = 0.3           # lambda : EWMA smoothing (lower -> smoother)

    # ── Sleep injection (straggler simulation) ────────────────────────────────
    # Both baseline and algorithm runs use the same settings so the
    # comparison is fair under identical straggler conditions.
    inject_sleep:         bool  = True
    sleep_prob_on:        float = 0.20  # probability of entering sleep state
    sleep_prob_off:       float = 0.20  # probability of leaving  sleep state
    sleep_check_interval: int   = 10    # batches between re-rolls
    sleep_duration_ratio: float = 1.5   # sleep = ratio x recent avg iter time
    sleep_seed:           int   = 42    # fix seed so both runs see same pattern

    # ── Logging / Checkpointing / Results ─────────────────────────────────────
    log_interval:     int           = 20
    checkpoint_dir:   str           = './checkpoints'
    results_dir:      str           = './results'
    resume:           Optional[str] = None
