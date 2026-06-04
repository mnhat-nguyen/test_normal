"""
straggler/amp_manager.py
------------------------
Manages the GradScaler lifecycle and AMP ↔ FP32 mode transitions
for a single worker.

Responsibilities
----------------
- Own the single GradScaler instance.
- Replace the scaler on mode transitions (GradScaler.enabled is
  read-only after construction, so we reconstruct on each flip).
- Align the scaler's internal scale to the globally agreed GSCM value.
- Log every transition with worker identity, Z, UCL / LCL.
- Delegate scale / step / update so callers never touch the scaler directly.
"""

import torch
from torch.cuda.amp import GradScaler


class AmpManager:
    """
    Parameters
    ----------
    global_rank : int           Worker's global rank (for log messages).
    device      : torch.device  CUDA device for this worker.
    """

    def __init__(self, global_rank: int, device: torch.device):
        self.global_rank = global_rank
        self.device      = device
        self._scaler     = GradScaler(enabled=False)   # starts in FP32

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def amp_active(self) -> bool:
        return self._scaler.is_enabled()

    @property
    def scaler(self) -> GradScaler:
        return self._scaler

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def apply_mode(
        self,
        activate: bool,
        z:   float,
        ucl: float | None,
        lcl: float | None,
    ):
        """
        Switch to the requested mode if it differs from the current one.

        Parameters
        ----------
        activate : bool   True → AMP,  False → FP32.
        z        : float  Current EWMA value Z_t (for logging).
        ucl      : float | None  Current UCL (for logging).
        lcl      : float | None  Current LCL (for logging).
        """
        was_active = self.amp_active

        if activate and not was_active:
            self._scaler = GradScaler(enabled=True)
            ucl_str = f"{ucl:.2f}" if ucl is not None else "N/A"
            print(
                f"[GPU{self.global_rank}] ⚠  Straggler detected — "
                f"switching to AMP (FP16).  "
                f"Z={z:.2f} ms  UCL={ucl_str} ms"
            )

        elif not activate and was_active:
            self._scaler = GradScaler(enabled=False)
            lcl_str = f"{lcl:.2f}" if lcl is not None else "N/A"
            print(
                f"[GPU{self.global_rank}] ✓  Worker recovered — "
                f"switching back to FP32.  "
                f"Z={z:.2f} ms  LCL={lcl_str} ms"
            )

    def align_scale(self, global_scale: float):
        """
        Force the GradScaler's internal scale to global_scale (GSCM alignment).
        No-op when AMP is not active.
        """
        if not self.amp_active:
            return
        if self._scaler.get_scale() != global_scale:
            self._scaler._scale = torch.tensor(
                global_scale, dtype=torch.float32, device=self.device
            )

    # ------------------------------------------------------------------
    # GradScaler delegation
    # ------------------------------------------------------------------

    def get_scale(self) -> float:
        """Return current loss scale (1.0 when FP32)."""
        return self._scaler.get_scale() if self.amp_active else 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self._scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer):
        self._scaler.step(optimizer)

    def update(self):
        self._scaler.update()
