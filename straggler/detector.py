"""
straggler/detector.py
---------------------
Per-worker EWMA-MAD straggler detector with a self-purifying sliding window.

One instance runs per worker independently — no inter-worker communication.

Algorithm (called once per non-boundary batch)
----------------------------------------------
1. EWMA update
       Z_t = λ·X_t + (1-λ)·Z_{t-1}
   Z is initialised to X_t on the very first call.

2. Mode logic
   AMP active   → if Z_t < LCL  : deactivate AMP  (worker recovered)
   AMP inactive → if Z_t > UCL  : activate AMP    (straggler detected)
                  elif X_t < UCL : admit X_t to W, recompute UCL/LCL

3. Threshold computation (MAD-based, recomputed every clean admission)
       m   = median(W)
       MAD = median(|x - m|  for x in W)
       UCL = m + k · MAD
       LCL = max(0, m − k · MAD)

4. Self-purifying window
   Only clean observations (X_t < UCL) are ever admitted to W.
   Detection is disabled until |W| >= n_min  (cold-start guard).

Parameters
----------
window_size : int    Maximum observations kept in W              (default 20)
n_min       : int    Cold-start guard — min samples before detection (default 10)
k           : float  MAD scaling factor for UCL and LCL          (default 3.0)
ewma_lambda : float  EWMA smoothing factor λ  (0 < λ ≤ 1)       (default 0.3)
"""

import statistics
from collections import deque
from typing import Optional


class StraglerDetector:
    """
    EWMA-based straggler detector with a self-purifying sliding window.
    """

    def __init__(
        self,
        window_size:  int,
        n_min:        int,
        k:            float,
        ewma_lambda:  float,
    ) -> None:
        self.N          = window_size
        self.n_min      = n_min
        self.k          = k
        self.lam        = ewma_lambda

        self.W: deque           = deque(maxlen=window_size)
        self.Z: Optional[float] = None    # EWMA state; None until first observation
        self.amp_flag: bool     = False
        self.UCL: float         = float('inf')
        self.LCL: float         = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x_t: float) -> bool:
        """
        Ingest one batch iteration time x_t (milliseconds).

        Parameters
        ----------
        x_t : float   Wall-clock time (ms) for the most recent full batch.

        Returns
        -------
        amp_flag : bool
            True  → this worker should run in AMP (FP16) mode.
            False → this worker should run in Normal (FP32) mode.
        """
        # Step 1 — EWMA update
        if self.Z is None:
            self.Z = x_t
        else:
            self.Z = self.lam * x_t + (1.0 - self.lam) * self.Z

        # Step 2 — Mode logic
        if self.amp_flag:
            # Recovery: smoothed value dropped below LCL → revert to Normal
            if self.Z < self.LCL:
                self.amp_flag = False
        else:
            # Straggler detection (only after cold-start window is filled)
            if self.is_warmed_up and self.Z > self.UCL:
                self.amp_flag = True
            else:
                # Clean observation — admit to window and refresh thresholds
                if x_t < self.UCL:
                    self.W.append(x_t)
                    if self.is_warmed_up:
                        self._recompute_thresholds()

        return self.amp_flag

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _recompute_thresholds(self) -> None:
        """UCL / LCL from the current clean window using MAD."""
        data = list(self.W)
        m    = statistics.median(data)
        mad  = statistics.median([abs(x - m) for x in data])
        self.UCL = m + self.k * mad
        self.LCL = max(0.0, m - self.k * mad)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_warmed_up(self) -> bool:
        """True once the window holds at least n_min clean samples."""
        return len(self.W) >= self.n_min

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            'W':        list(self.W),
            'Z':        self.Z,
            'amp_flag': self.amp_flag,
            'UCL':      self.UCL,
            'LCL':      self.LCL,
        }

    def load_state_dict(self, d: dict) -> None:
        self.W        = deque(d['W'], maxlen=self.N)
        self.Z        = d['Z']
        self.amp_flag = d['amp_flag']
        self.UCL      = d['UCL']
        self.LCL      = d['LCL']

    def __repr__(self) -> str:
        z   = f"{self.Z:.2f}"   if self.Z   is not None       else "None"
        ucl = f"{self.UCL:.2f}" if self.UCL != float('inf')   else "inf"
        lcl = f"{self.LCL:.2f}"
        return (
            f"StraglerDetector("
            f"amp={self.amp_flag}, warmed_up={self.is_warmed_up}, "
            f"|W|={len(self.W)}, Z={z}, UCL={ucl}, LCL={lcl})"
        )
