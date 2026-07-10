import statistics
from collections import deque
from typing import Optional


class StraglerDetector:
    """
    EWMA-based straggler detector with a self-purifying sliding window.

    One instance runs per worker, independently.

    Algorithm
    ---------
    For each batch iteration, X_t is the measured wall-clock time (ms) to
    complete one full batch: forward + backward + DDP allreduce + optimizer step.

    1. EWMA update
           Z_t = λ·X_t + (1-λ)·Z_{t-1}

    2. Mode logic
       AMP active  → if Z_t < LCL : deactivate AMP
       AMP inactive→ if Z_t > UCL : activate AMP
                     else if X_t < UCL : treat as clean
                         append X_t to W (sliding window)
                         recompute UCL / LCL from W via MAD

    UCL = median(W) + k · MAD(W)
    LCL = max(0, median(W) − k · MAD(W))

    The window W only ever contains clean observations, so the baseline
    never drifts upward due to straggler contamination.
    Detection is disabled until |W| ≥ n_min (cold-start guard).
    """

    def __init__(
        self,
        window_size:  int,
        n_min:        int,
        k:            float,
        ewma_lambda:  float,
    ) -> None:
        """
        Parameters
        ----------
        window_size  : N      – maximum samples kept in the sliding window
        n_min        : N_min  – minimum samples before detection is enabled
        k            : MAD scaling factor (same for UCL and LCL)
        ewma_lambda  : λ      – EWMA weight on the newest observation
                                (0 < λ ≤ 1; smaller → heavier smoothing)
        """
        self.N   = window_size
        self.n_min = n_min
        self.kl   = k+1
        self.ku   = k-1
        self.lam = ewma_lambda

        self.W: deque          = deque(maxlen=window_size)
        self.Z: Optional[float] = None   # EWMA state; None until first observation
        self.amp_flag: bool     = False
        self.UCL: float         = float('inf')
        self.LCL: float         = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, x_t: float) -> bool:
        """
        Ingest one batch iteration time x_t (milliseconds).

        Returns
        -------
        amp_flag : bool
            True  → this worker should be running in AMP mode.
            False → this worker should be running in Normal (FP32) mode.
        """
        # ── Step 1 : EWMA ────────────────────────────────────────────────────
        if self.Z is None:
            self.Z = x_t                             # initialise on first call
        else:
            self.Z = self.lam * x_t + (1.0 - self.lam) * self.Z

        # ── Step 2 : Mode logic ───────────────────────────────────────────────
        if self.amp_flag:
            # Recovery: smoothed value has dropped below LCL → revert to Normal
            if self.Z < self.LCL:
                self.amp_flag = False

        else:
            # Straggler detection (only after cold-start window is filled)
            if self.is_warmed_up and self.Z > self.UCL:
                self.amp_flag = True
            else:
                # X_t looks clean → admit to window and refresh thresholds
                if x_t < self.UCL:
                    self.W.append(x_t)
                    if self.is_warmed_up:
                        self._recompute_thresholds()

        return self.amp_flag

    # ─────────────────────────────────────────────────────────────────────────
    def _recompute_thresholds(self) -> None:
        data = list(self.W)
        m    = statistics.median(data)
        mad  = statistics.median([abs(x - m) for x in data])

        # Scale MAD to be a consistent estimator of std-dev under normality
        # (standard constant, see Rousseeuw & Croux 1993). Without this,
        # raw MAD understates spread by ~1.5x versus a Gaussian sigma.
        # mad_scaled = mad * 1.4826

        # Floor the scaled MAD so the control band can never collapse to
        # near-zero when the underlying X_t distribution is extremely stable.
        # Without this floor, tiny natural jitter (GPU/OS scheduling noise)
        # crosses UCL/LCL on its own and the detector flaps ON/OFF with no
        # real straggler present — exactly the pattern of a near-constant
        # baseline (MAD -> 0) making the band only 1-2ms wide.
        # min_mad = max(1.0, 0.01 * m)   # at least 1ms, or 1% of the median
        # mad_eff = max(mad_scaled, min_mad)

        self.UCL = m + self.ku * mad
        self.LCL = max(0.0, m - self.kl * mad)

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def is_warmed_up(self) -> bool:
        """True once the window holds at least n_min clean samples."""
        return len(self.W) >= self.n_min

    # ── Checkpoint helpers ────────────────────────────────────────────────────
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