"""
straggler/gscm.py
=================
Gradient Scale Consistency Mechanism (GSCM)
============================================
Problem
-------
In DDP, the allreduce averages gradients across all workers.
If worker A is in AMP mode (gradients internally scaled by S via GradScaler)
and worker B is in Normal mode (gradients unscaled), the averaged gradient
is a meaningless mixture of two different magnitudes.

Solution (from the Korel paper)
--------------------------------
Before every backward pass, all workers agree on a single global scale S:

  • AMP workers    → GradScaler already scales loss by S internally.
                     We synchronise S so all AMP workers use the same value.
  • Normal workers → manually multiply loss by S before backward().

After DDP allreduce all gradients carry factor S.
Then every worker removes that factor:
  • AMP workers    → scaler.unscale_(optimizer)  (divides by S)
  • Normal workers → GSCM.unscale_gradients()    (divides by S manually)

Extension hook
--------------
The method `sync_scale` currently uses dist.all_reduce(MAX) so any worker
can determine the global scale from its own scaler.  When "shared timing"
is added later, this method is the natural place to also broadcast per-worker
iteration times for global straggler awareness.
"""

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from typing import Optional


class GSCM:

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ------------------------------------------------------------------
    def sync_scale(
        self,
        amp_active: bool,
        scaler: Optional[GradScaler] = None,
    ) -> float:
        """
        All-reduce to find the global gradient scale S for this iteration.

        Each worker contributes:
          AMP workers    → their current GradScaler scale
          Normal workers → 1.0

        The MAX is taken so that Normal workers can scale up to match
        the AMP workers.  The AMP workers' scalers are then aligned to
        this global value so scaler.unscale_() later undoes exactly S.

        Returns
        -------
        global_scale : float
            The agreed-upon scale S that every worker will use this step.
            Returns 1.0 when no worker is in AMP mode (GSCM is a no-op).
        """
        local_scale = scaler.get_scale() if (amp_active and scaler) else 1.0

        # Single-process: no all_reduce needed, just return local scale
        if not dist.is_initialized():
            return local_scale

        t = torch.tensor([local_scale], dtype=torch.float32, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        global_scale = t.item()

        # Align the AMP worker's internal scaler to the global scale so that
        # scaler.unscale_(optimizer) undoes exactly global_scale later.
        if amp_active and scaler is not None and global_scale != local_scale:
            scaler._scale.fill_(global_scale)   # type: ignore[attr-defined]

        return global_scale

    # ------------------------------------------------------------------
    @staticmethod
    def scale_loss(loss: torch.Tensor, global_scale: float) -> torch.Tensor:
        """
        Normal-mode workers call this to match AMP-mode gradient magnitudes.
        If global_scale == 1.0 (no AMP worker active) this is a no-op.
        """
        if global_scale == 1.0:
            return loss
        return loss * global_scale

    # ------------------------------------------------------------------
    @staticmethod
    def unscale_gradients(
        model: torch.nn.Module,
        global_scale: float,
    ) -> None:
        """
        Normal-mode workers call this after DDP allreduce to remove the
        scale factor that was applied to the loss before backward().
        """
        if global_scale == 1.0:
            return
        inv = 1.0 / global_scale
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(inv)
