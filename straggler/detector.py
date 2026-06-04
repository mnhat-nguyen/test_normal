"""
straggler/detector.py
---------------------
Per-worker straggler detector.

Each worker independently monitors its own batch iteration time X_t
(wall-clock ms for one full batch: forward + backward + allreduce +
optimizer step).  No inter-worker communication is needed.

Algorithm — called once per batch iteration
-------------------------------------------
1. EWMA update:
       Z_t = λ · X_t + (1 - λ) · Z_{t-1}
   Z is initialised to X_t on the very first call.

2. Mode logic:
   - AMP active   → if Z_t < LCL : deactivate AMP  (worker recovered)
   - AMP inactive →
       if warmed_up AND Z_t > UCL : activate AMP   (straggler detected)
       elif X_t < UCL             : admit X_t to W, recompute thresholds

3. Threshold computation (MAD-based):
       m   = median(W)
       MAD = median(|x - m|  for x in W)
       UCL = m + k · MAD
       LCL = max(0, m - k · MAD)

4. Self-purifying window:
   Only clean observations (X_t < UCL) are ever admitted to W so the
   baseline never drifts upward.  Detection is disabled until |W| >= n_min
   (cold-start guard).

Parameters
----------
window_size : int   Maximum observations kept in W            (default 20)
n_min       : int   Cold-start guard — min samples before detection (default 10)
k           : float MAD scaling factor for UCL and LCL        (default 3.0)
ewma_lambda : float EWMA smoothing factor λ  (0 < λ ≤ 1)     (default 0.3)
"""

import statistics


class StragglerDetector:

    def __init__(
        self,
        window_size: int  = 20,
        n_min:       int  = 10,
        k:           float = 3.0,
        ewma_lambda: float = 0.3,
    ):
        self.window_size = window_size
        self.n_min       = n_min
        self.k           = k
        self.ewma_lambda = ewma_lambda

        # mutable state
        self.window:     list[float]  = []
        self.Z:          float | None = None
        self.UCL:        float | None = None
        self.LCL:        float | None = None
        self.amp_active: bool         = False
        self.warmed_up:  bool         = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, x_t: float) -> bool:
        """
        Run one iteration of the detection algorithm.

        Parameters
        ----------
        x_t : float   Wall-clock time (ms) for the most recent batch.

        Returns
        -------
        bool  True → AMP should be active.  False → FP32 should be active.
        """
        self._update_ewma(x_t)
        self._update_mode(x_t)
        return self.amp_active

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_ewma(self, x_t: float):
        if self.Z is None:
            self.Z = x_t
        else:
            self.Z = self.ewma_lambda * x_t + (1.0 - self.ewma_lambda) * self.Z

    def _update_mode(self, x_t: float):
        if self.amp_active:
            # Recovery: smoothed time dropped below LCL
            if self.LCL is not None and self.Z < self.LCL:
                self.amp_active = False
        else:
            if self.warmed_up and self.UCL is not None and self.Z > self.UCL:
                # Straggler detected — dirty observation, do NOT admit
                self.amp_active = True
            elif self.UCL is None or x_t < self.UCL:
                # Clean observation
                self._admit(x_t)

    def _admit(self, x_t: float):
        self.window.append(x_t)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        if len(self.window) >= self.n_min:
            self.warmed_up = True
            self._recompute_thresholds()

    def _recompute_thresholds(self):
        m   = statistics.median(self.window)
        mad = statistics.median([abs(x - m) for x in self.window])
        self.UCL = m + self.k * mad
        self.LCL = max(0.0, m - self.k * mad)

    def __repr__(self) -> str:
        z   = f"{self.Z:.2f}"   if self.Z   is not None else "None"
        ucl = f"{self.UCL:.2f}" if self.UCL is not None else "None"
        lcl = f"{self.LCL:.2f}" if self.LCL is not None else "None"
        return (
            f"StragglerDetector("
            f"amp={self.amp_active}, warmed_up={self.warmed_up}, "
            f"|W|={len(self.window)}, Z={z}, UCL={ucl}, LCL={lcl})"
        )
