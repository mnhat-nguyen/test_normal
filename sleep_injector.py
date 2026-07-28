"""
sleep_injector.py
-----------------
Simulates resource contention by injecting sleep into training iterations.
Both train.py (with algorithm) and train_baseline.py use the same class
with the same seed so both runs face identical straggler conditions and
the comparison is fair.
"""
import time
import random
import json
from collections import deque


class SleepInjector:
    """
    Mimics multi-tenant resource contention (paper Section V-B).

    State machine per worker:
      - Every `check_interval` batches, re-roll the sleep state.
      - If currently NOT sleeping: enter sleep with probability `prob_on`.
      - If currently sleeping:     leave  sleep with probability `prob_off`.
      - While sleeping: sleep for `duration_ratio` × recent average iter time.

    The very first iteration of the whole run is skipped entirely: it's a
    cold-start outlier (CUDA context init, cuDNN autotuning, lazy kernel
    compilation) whose inflated time would otherwise pollute `_recent` and
    make every subsequent sleep duration too large. The training loops
    initialise last_x_t = 0.0 and only ever pass 0.0 on that very first
    call, so `last_iter_ms <= 0` uniquely identifies it.

    Recording: every call stores the sleep it applied (ms) in `last_sleep_ms`
    (0.0 if it didn't sleep). The training loop reads this to build a
    per-batch record that train.py can later REPLAY for a fair comparison.
    """

    def __init__(
        self,
        prob_on:         float = 0.20,
        prob_off:        float = 0.20,
        check_interval:  int   = 10,
        duration_ratio:  float = 0.80,
        seed:            int   = 42,
    ) -> None:
        self.prob_on        = prob_on
        self.prob_off       = prob_off
        self.interval       = check_interval
        self.ratio          = duration_ratio
        self.rng            = random.Random(seed)

        self.sleeping       = False
        self._recent: deque = deque(maxlen=20)   # recent iter times for avg
        self.last_sleep_ms  = 0.0                # sleep applied on the last call

    # ─────────────────────────────────────────────────────────────────────────
    def maybe_sleep(self, batch_idx: int, last_iter_ms: float) -> bool:
        """
        Call this BEFORE the timed training step.

        Parameters
        ----------
        batch_idx    : current batch index within the epoch
        last_iter_ms : measured X_t of the previous batch (ms)

        Returns
        -------
        sleeping : bool  — whether a sleep was injected this call
        """
        self.last_sleep_ms = 0.0

        # Skip the very first iteration of the run (cold-start outlier).
        # last_iter_ms is 0.0 only on that first call, since every real
        # iteration produces a positive X_t. Returning early here means the
        # cold-start time is never added to _recent and no sleep is injected
        # on batch 0.
        if last_iter_ms <= 0:
            return False

        self._recent.append(last_iter_ms)

        # Re-roll sleep state every `interval` batches
        if batch_idx % self.interval == 0:
            if self.sleeping:
                if self.rng.random() < self.prob_off:
                    self.sleeping = False
            else:
                if self.rng.random() < self.prob_on:
                    self.sleeping = True

        if self.sleeping and self._recent:
            avg_ms    = sum(self._recent) / len(self._recent)
            sleep_sec = (avg_ms * self.ratio) / 1000.0
            time.sleep(sleep_sec)
            self.last_sleep_ms = sleep_sec * 1000.0
            print(f"[SLEEP] batch {batch_idx} : sleeping for {sleep_sec*1000:.3f}ms ")
            return True

        return False

    @property
    def is_sleeping(self) -> bool:
        return self.sleeping


class ReplaySleepInjector:
    """
    Drop-in replacement for SleepInjector that REPLAYS a recorded sleep
    pattern instead of generating its own. Same maybe_sleep / is_sleeping
    interface, so train.py needs no other changes.

    `pattern` is a dict {"epoch:batch": sleep_ms} produced by train_baseline.
    Call set_epoch(e) at the start of each epoch so the (epoch, batch) key
    matches the recording.
    """

    def __init__(self, pattern: dict):
        self.pattern       = pattern
        self.epoch         = 0
        self.sleeping      = False
        self.last_sleep_ms = 0.0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def maybe_sleep(self, batch_idx: int, last_iter_ms: float) -> bool:
        self.last_sleep_ms = 0.0
        self.sleeping      = False

        if last_iter_ms <= 0:            # same cold-start guard
            return False

        ms = self.pattern.get(f"{self.epoch}:{batch_idx}", 0.0)
        if ms > 0:
            time.sleep(ms / 1000.0)
            self.sleeping      = True
            self.last_sleep_ms = ms
            print(f"[SLEEP] batch {batch_idx} : sleeping for {ms:.3f}ms (replay) ")
            return True

        return False

    @property
    def is_sleeping(self) -> bool:
        return self.sleeping