import statistics
from collections import deque
from typing import Optional


class StraglerDetector:
    """
    EWMA-based Straggler Detection with Self-Purifying Sliding Window.

    One instance runs per worker, independently. Implements Algorithm 1
    from algorithm.pdf:

    Input : N (window size), N_min (cold-start threshold),
            k (MAD scaling factor), λ (EWMA smoothing factor)
    Init  : W ← [], Z ← None, amp_flag ← 0, UCL ← ∞, LCL ← 0

    For each iteration with duration X_t:
      1. EWMA update:
             Z ← X_t                          if Z is None
             Z ← λ·X_t + (1-λ)·Z              otherwise

      2. Mode logic:
         if amp_flag == 1:
             if Z < LCL: amp_flag ← 0
         else:
             if |W| >= N_min and Z > UCL:
                 amp_flag ← 1
             else:
                 if X_t < UCL:                 # clean observation — NOT Z,
                                                # since Z lags X_t and would
                                                # let already-elevated values
                                                # slip into W during that lag
                     append X_t to W (drop oldest if |W| > N)
                     if |W| >= N_min:           # recompute thresholds
                         m   ← median(W)
                         MAD ← median(|x - m| for x in W)
                         UCL ← m + ku·MAD
                         LCL ← max(0, m - kl·MAD)

    Both UCL and LCL are derived from the SAME window W (Normal-mode
    clean observations only). Neither is recomputed while amp_flag == 1
    — W is only updated in Normal mode, so both thresholds stay frozen
    at whatever they were when AMP last activated, until the worker
    returns to Normal mode and starts admitting clean samples again.
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
        k            : base MAD scaling factor; ku/kl derived from it
        ewma_lambda  : λ      – EWMA weight on the newest observation
                                (0 < λ ≤ 1; smaller → heavier smoothing)
        """
        self.N     = window_size
        self.n_min = n_min
        self.kl    = k + 1
        self.ku    = k - 1.5
        self.lam   = ewma_lambda

        self.W: deque            = deque(maxlen=window_size)
        self.Z: Optional[float]  = None   # EWMA state; None until first observation
        self.amp_flag: bool      = False
        self.UCL: float          = float('inf')
        self.LCL: float          = 0.0

        # True once |W| has reached n_min at least once. Monotonic — W only
        # grows via append and never shrinks below n_min once reached, so
        # this never needs to flip back to False.
        self.filled: bool        = False

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
            # Once True, short-circuits — len(self.W) is never checked again.
            if not self.filled:
                self.filled = len(self.W) >= self.n_min

            # Straggler detection (only after cold-start window is filled)
            if self.filled and self.Z > self.UCL:
                self.amp_flag = True
            else:
                # x_t (NOT Z) looks clean → admit to window and refresh
                # thresholds. Using x_t, not the lagging Z, keeps W free of
                # values from the early part of a straggler event.
                if x_t < self.UCL:
                    self.W.append(x_t)
                    if self.filled:
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
        mad_scaled = mad * 1.4826

        # Floor the scaled MAD so the control band can never collapse to
        # near-zero when the underlying X_t distribution is extremely stable.
        # Without this floor, tiny natural jitter (GPU/OS scheduling noise)
        # crosses UCL/LCL on its own and the detector flaps ON/OFF with no
        # real straggler present.
        min_mad = max(1.0, 0.01 * m)   # at least 1ms, or 1% of the median
        mad_eff = max(mad_scaled, min_mad)

        self.UCL = m + self.ku * mad_eff
        self.LCL = max(0.0, m - self.kl * mad_eff)

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            'W':        list(self.W),
            'Z':        self.Z,
            'amp_flag': self.amp_flag,
            'UCL':      self.UCL,
            'LCL':      self.LCL,
            'filled':   self.filled,
        }

    def load_state_dict(self, d: dict) -> None:
        self.W        = deque(d['W'], maxlen=self.N)
        self.Z        = d['Z']
        self.amp_flag = d['amp_flag']
        self.UCL      = d['UCL']
        self.LCL      = d['LCL']
        self.filled   = d.get('filled', len(self.W) >= self.n_min)