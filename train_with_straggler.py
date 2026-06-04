"""
train_with_straggler.py
-----------------------
DDP training with:
  - EWMA-MAD straggler detection       (straggler.StragglerDetector)
  - AMP ↔ FP32 mode management         (straggler.AmpManager)
  - Gradient Scale Consistency (GSCM)  (straggler.sync_grad_scale)
  - Optional straggler simulation      (straggler.StragglerSimulator)

Algorithm recap (per batch)
---------------------------
1.  Measure wall-clock time X_t for the full batch iteration.
2.  EWMA:  Z_t = λ·X_t + (1-λ)·Z_{t-1}
3.  Mode logic:
      AMP active   → Z_t < LCL  → deactivate (recovered)
      AMP inactive → Z_t > UCL  → activate   (straggler)
                   → X_t < UCL  → admit to window, recompute UCL/LCL
4.  GSCM: all_reduce(MAX) over gradient scales so DDP averaging is
    numerically consistent across mixed-precision workers.

Usage
-----
    # Without simulation
    torchrun --standalone --nproc_per_node=4 train_with_straggler.py \
        50 5 --batch_size 32

    # With burst simulation on GPU 1
    torchrun --standalone --nproc_per_node=4 train_with_straggler.py \
        50 5 --batch_size 32 \
        --sim_mode burst --sim_target_rank 1 \
        --sim_slowdown_ms 200 --sim_start_batch 50 --sim_duration 30

    # With periodic simulation
    torchrun --standalone --nproc_per_node=4 train_with_straggler.py \
        50 5 --batch_size 32 \
        --sim_mode periodic --sim_target_rank 2 \
        --sim_slowdown_ms 150 --sim_start_batch 20 \
        --sim_duration 20 --sim_interval 80

    # With random simulation
    torchrun --standalone --nproc_per_node=4 train_with_straggler.py \
        50 5 --batch_size 32 \
        --sim_mode random --sim_target_rank 0 \
        --sim_slowdown_ms 100 --sim_prob 0.1
"""

import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from straggler import StragglerDetector, AmpManager, sync_grad_scale, StragglerSimulator


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model:         torch.nn.Module,
        train_data:    DataLoader,
        optimizer:     torch.optim.Optimizer,
        save_every:    int,
        snapshot_path: str,
        # ---- detector params ----
        window_size:  int   = 20,
        n_min:        int   = 10,
        k:            float = 3.0,
        ewma_lambda:  float = 0.3,
        # ---- simulator params ----
        sim_mode:         str   = "none",
        sim_target_rank:  int   = 0,
        sim_slowdown_ms:  float = 200.0,
        sim_start_batch:  int   = 50,
        sim_duration:     int   = 30,
        sim_interval:     int   = 100,
        sim_prob:         float = 0.05,
    ) -> None:
        self.local_rank  = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.device      = torch.device(f"cuda:{self.local_rank}")

        self.model         = model.to(self.device)
        self.train_data    = train_data
        self.optimizer     = optimizer
        self.save_every    = save_every
        self.epochs_run    = 0
        self.snapshot_path = snapshot_path
        self.batch_idx     = 0   # global batch counter across all epochs

        if os.path.exists(snapshot_path):
            print(f"[GPU{self.global_rank}] Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

        # ---- Straggler detector (no inter-worker comms) ----
        self.detector = StragglerDetector(
            window_size=window_size,
            n_min=n_min,
            k=k,
            ewma_lambda=ewma_lambda,
        )

        # ---- AMP / FP32 mode manager ----
        self.amp_manager = AmpManager(
            global_rank=self.global_rank,
            device=self.device,
        )

        # ---- Straggler simulator (pass-through when mode="none") ----
        self.simulator = StragglerSimulator(
            target_rank=sim_target_rank,
            global_rank=self.global_rank,
            mode=sim_mode,
            slowdown_ms=sim_slowdown_ms,
            start_batch=sim_start_batch,
            duration=sim_duration,
            interval=sim_interval,
            prob=sim_prob,
        )

    # ------------------------------------------------------------------
    # Snapshot I/O
    # ------------------------------------------------------------------

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"[GPU{self.global_rank}] Resuming from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN":  epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    # ------------------------------------------------------------------
    # Core batch execution (with GSCM)
    # ------------------------------------------------------------------

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor):
        """
        Forward + backward + optimizer step.

        GSCM protocol
        -------------
        1. All workers all_reduce(MAX) their local gradient scale → S.
        2. AMP  workers align their GradScaler to S.
           FP32 workers multiply loss by S before backward.
        3. FP32 workers divide their gradients by S after DDP allreduce.
        4. AMP  workers go through scaler.step(); FP32 step directly.
        """
        # 1. Agree on global gradient scale
        global_scale = sync_grad_scale(self.amp_manager.get_scale(), self.device)

        # 2. Align AMP scaler to global scale
        self.amp_manager.align_scale(global_scale)

        self.optimizer.zero_grad()

        if self.amp_manager.amp_active:
            # AMP path
            with autocast():
                output = self.model(source)
                loss   = F.cross_entropy(output, targets)
            self.amp_manager.scale(loss).backward()
        else:
            # FP32 path — manual loss scaling (GSCM)
            output = self.model(source)
            loss   = F.cross_entropy(output, targets)
            (loss * global_scale).backward()

        # 3. Post-allreduce correction for FP32 workers
        if not self.amp_manager.amp_active and global_scale != 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(global_scale)

        # 4. Optimizer step
        if self.amp_manager.amp_active:
            self.amp_manager.step(self.optimizer)
            self.amp_manager.update()
        else:
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Timed batch wrapper — simulation + detection
    # ------------------------------------------------------------------

    def _run_batch_timed(self, source: torch.Tensor, targets: torch.Tensor):
        """
        1. Ask simulator to inject delay if this batch is a straggler batch.
        2. Time the full iteration (CUDA-synchronised).
        3. Feed X_t to the detector.
        4. Apply the resulting AMP / FP32 mode decision.
        """
        # Simulation: inject artificial slowdown before the batch
        self.simulator.maybe_inject(self.batch_idx)

        # CUDA-accurate timing
        torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()

        self._run_batch(source, targets)

        torch.cuda.synchronize(self.device)
        x_t = (time.perf_counter() - t_start) * 1000.0   # ms

        # Detection
        amp_should_be_active = self.detector.update(x_t)

        # Mode transition (logs on change)
        self.amp_manager.apply_mode(
            activate=amp_should_be_active,
            z=self.detector.Z,
            ucl=self.detector.UCL,
            lcl=self.detector.LCL,
        )

        self.batch_idx += 1

    # ------------------------------------------------------------------
    # Epoch / training loop
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | "
            f"Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source  = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch_timed(source, targets)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train_objs():
    train_set = MyTrainDataset(2048, input_dim=20, num_class=2)
    model     = torch.nn.Linear(20, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(
    save_every:    int,
    total_epochs:  int,
    batch_size:    int,
    snapshot_path: str   = "snapshot.pt",
    # detector
    window_size:  int   = 20,
    n_min:        int   = 10,
    k:            float = 3.0,
    ewma_lambda:  float = 0.3,
    # simulator
    sim_mode:         str   = "none",
    sim_target_rank:  int   = 0,
    sim_slowdown_ms:  float = 200.0,
    sim_start_batch:  int   = 50,
    sim_duration:     int   = 30,
    sim_interval:     int   = 100,
    sim_prob:         float = 0.05,
):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(
        model, train_data, optimizer, save_every, snapshot_path,
        window_size=window_size, n_min=n_min, k=k, ewma_lambda=ewma_lambda,
        sim_mode=sim_mode, sim_target_rank=sim_target_rank,
        sim_slowdown_ms=sim_slowdown_ms, sim_start_batch=sim_start_batch,
        sim_duration=sim_duration, sim_interval=sim_interval, sim_prob=sim_prob,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DDP training with EWMA-MAD straggler mitigation + GSCM"
    )

    # ---- positional ----
    parser.add_argument("total_epochs", type=int, help="Total epochs to train")
    parser.add_argument("save_every",   type=int, help="Snapshot frequency (epochs)")

    # ---- training ----
    parser.add_argument("--batch_size",    default=32,   type=int,   help="Per-device batch size (default: 32)")
    parser.add_argument("--snapshot_path", default="snapshot.pt", type=str)

    # ---- detector ----
    parser.add_argument("--window_size",  default=20,  type=int,   help="Sliding window size N (default: 20)")
    parser.add_argument("--n_min",        default=10,  type=int,   help="Cold-start threshold (default: 10)")
    parser.add_argument("--k",            default=3.0, type=float, help="MAD scaling factor (default: 3.0)")
    parser.add_argument("--ewma_lambda",  default=0.3, type=float, help="EWMA smoothing factor λ (default: 0.3)")

    # ---- simulator ----
    parser.add_argument("--sim_mode",         default="none",  type=str,   help="Simulation mode: burst | periodic | random | none (default: none)")
    parser.add_argument("--sim_target_rank",  default=0,       type=int,   help="Which GPU rank to slow down (default: 0)")
    parser.add_argument("--sim_slowdown_ms",  default=200.0,   type=float, help="Extra delay per straggler batch in ms (default: 200)")
    parser.add_argument("--sim_start_batch",  default=50,      type=int,   help="Batch index where simulation starts (default: 50)")
    parser.add_argument("--sim_duration",     default=30,      type=int,   help="Number of straggler batches per episode (default: 30)")
    parser.add_argument("--sim_interval",     default=100,     type=int,   help="Batches between periodic episodes (default: 100)")
    parser.add_argument("--sim_prob",         default=0.05,    type=float, help="Per-batch straggler probability for random mode (default: 0.05)")

    args = parser.parse_args()

    main(
        args.save_every,
        args.total_epochs,
        args.batch_size,
        snapshot_path=args.snapshot_path,
        window_size=args.window_size,
        n_min=args.n_min,
        k=args.k,
        ewma_lambda=args.ewma_lambda,
        sim_mode=args.sim_mode,
        sim_target_rank=args.sim_target_rank,
        sim_slowdown_ms=args.sim_slowdown_ms,
        sim_start_batch=args.sim_start_batch,
        sim_duration=args.sim_duration,
        sim_interval=args.sim_interval,
        sim_prob=args.sim_prob,
    )
