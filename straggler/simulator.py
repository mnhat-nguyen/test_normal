"""
straggler/simulator.py
----------------------
StragglerSimulator — injects artificial slowdowns to exercise the detector
without waiting for real hardware anomalies.

This is the modular counterpart to sleep_injector.py.  Both can coexist:
  • sleep_injector.SleepInjector  → probabilistic state-machine model
    (used by train.py / train_baseline.py for fair head-to-head comparison)
  • straggler.StragglerSimulator  → explicit schedule (burst / periodic / random)
    (used when you want deterministic, reproducible slowdown patterns)

Simulation modes
----------------
burst     One continuous episode of `duration` straggler batches starting
          at `start_batch`.  After the episode the worker returns to normal.

periodic  Straggler for `duration` batches every `interval` batches,
          repeating indefinitely from `start_batch`.

random    Each batch has an independent probability `prob` of being a
          straggler batch (1-batch episodes, seeded for reproducibility).

none      No simulation — transparent pass-through (zero overhead).

Usage
-----
    from straggler.simulator import StragglerSimulator

    sim = StragglerSimulator(
        target_rank  = 1,
        global_rank  = rank,
        mode         = "burst",
        slowdown_ms  = 200,
        start_batch  = 50,
        duration     = 30,
    )

    # Inside the batch loop, call before the timed training step:
    sim.maybe_inject(batch_idx)
"""

import time
import random


class StragglerSimulator:
    """
    Parameters
    ----------
    target_rank : int    Global rank of the worker to slow down.
    global_rank : int    This worker's global rank.
    mode        : str    "burst" | "periodic" | "random" | "none".
    slowdown_ms : float  Extra wall-clock delay injected per straggler batch (ms).

    Mode-specific
    -------------
    burst / periodic
        start_batch : int   Batch index where the (first) episode begins.
        duration    : int   Consecutive straggler batches per episode.
    periodic only
        interval    : int   Total batches between episode starts (> duration).
    random only
        prob        : float Per-batch straggler probability [0, 1].
    """

    VALID_MODES = {"burst", "periodic", "random", "none"}

    def __init__(
        self,
        target_rank:  int,
        global_rank:  int,
        mode:         str   = "burst",
        slowdown_ms:  float = 200.0,
        # burst / periodic
        start_batch:  int   = 50,
        duration:     int   = 30,
        # periodic only
        interval:     int   = 100,
        # random only
        prob:         float = 0.05,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got '{mode}'"
            )

        self.target_rank = target_rank
        self.global_rank = global_rank
        self.mode        = mode
        self.slowdown_ms = slowdown_ms
        self.start_batch = start_batch
        self.duration    = duration
        self.interval    = interval
        self.prob        = prob

        self._is_target  = (global_rank == target_rank)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def maybe_inject(self, batch_idx: int) -> bool:
        """
        Call once per batch iteration BEFORE the timed training step.

        Parameters
        ----------
        batch_idx : int   Zero-based global batch index across the entire run.

        Returns
        -------
        bool   True if a delay was injected this call.
        """
        if not self._is_target:
            return False
        if self._is_straggler_batch(batch_idx):
            time.sleep(self.slowdown_ms / 1000.0)
            return True
        return False

    def is_straggler_batch(self, batch_idx: int) -> bool:
        """
        Returns True if this batch is scheduled as a straggler for the target
        worker.  Useful for assertions and offline analysis without injecting.
        """
        if not self._is_target:
            return False
        return self._is_straggler_batch(batch_idx)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_straggler_batch(self, batch_idx: int) -> bool:
        if self.mode == "none":
            return False

        if self.mode == "burst":
            return self.start_batch <= batch_idx < self.start_batch + self.duration

        if self.mode == "periodic":
            offset = batch_idx - self.start_batch
            if offset < 0:
                return False
            return (offset % self.interval) < self.duration

        if self.mode == "random":
            # Deterministic per batch_idx via a seeded RNG so results are
            # reproducible across runs with the same seed.
            return random.Random(batch_idx).random() < self.prob

        return False
