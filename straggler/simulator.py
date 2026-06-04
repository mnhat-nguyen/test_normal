"""
straggler/simulator.py
----------------------
StragglerSimulator — injects artificial slowdowns into a designated
worker so the detector can be exercised without waiting for real
hardware anomalies.

Simulation modes
----------------
burst     A single straggler episode of fixed length starting at a
          given batch index.  After the episode the worker returns to
          normal speed.

periodic  The worker becomes a straggler for `duration` batches every
          `interval` batches, repeating indefinitely.

random    Each batch has an independent probability `prob` of being a
          straggler batch.  Episodes last exactly 1 batch.

none      No simulation — acts as a transparent pass-through.

Usage
-----
    from straggler.simulator import StragglerSimulator

    sim = StragglerSimulator(
        target_rank   = 1,          # only GPU 1 is slowed down
        global_rank   = self.global_rank,
        mode          = "burst",
        slowdown_ms   = 200,        # extra sleep injected per straggler batch
        start_batch   = 50,         # episode starts at batch 50
        duration      = 30,         # lasts 30 batches
    )

    # Inside the training loop, call before _run_batch:
    sim.maybe_inject(batch_idx)
"""

import time
import random


class StragglerSimulator:
    """
    Parameters
    ----------
    target_rank : int
        The global rank of the worker that should be slowed down.
        All other workers are unaffected.
    global_rank : int
        This worker's global rank.
    mode : str
        One of  "burst" | "periodic" | "random" | "none".
    slowdown_ms : float
        Extra wall-clock delay (milliseconds) injected on straggler batches.

    Mode-specific parameters
    ------------------------
    burst
        start_batch : int   Batch index where the episode begins.
        duration    : int   Number of consecutive straggler batches.

    periodic
        start_batch : int   Batch index where the first episode begins.
        duration    : int   Number of straggler batches per episode.
        interval    : int   Total batches between episode starts (must be
                            > duration).

    random
        prob : float   Probability [0, 1] that any given batch is a straggler.
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
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.target_rank = target_rank
        self.global_rank = global_rank
        self.mode        = mode
        self.slowdown_ms = slowdown_ms
        self.start_batch = start_batch
        self.duration    = duration
        self.interval    = interval
        self.prob        = prob

        # True only for the designated straggler worker
        self._is_target = (global_rank == target_rank)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def maybe_inject(self, batch_idx: int):
        """
        Call once per batch iteration before _run_batch.
        Injects a sleep on the target worker when the simulation
        schedule says this batch should be a straggler.

        Parameters
        ----------
        batch_idx : int   Zero-based index of the current batch across
                          the entire training run (not reset per epoch).
        """
        if not self._is_target:
            return
        if self._is_straggler_batch(batch_idx):
            self._inject_delay()

    def is_straggler_batch(self, batch_idx: int) -> bool:
        """
        Returns True if this batch is a simulated straggler batch for
        the target worker.  Useful for logging / assertions in tests.
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
            # Deterministic per batch_idx so all calls agree for the same idx
            rng = random.Random(batch_idx)
            return rng.random() < self.prob

        return False

    def _inject_delay(self):
        time.sleep(self.slowdown_ms / 1000.0)
