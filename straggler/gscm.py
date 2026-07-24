"""
Gradient Scale Consistency Mechanism (GSCM)
============================================
Problem
-------
In DDP, the allreduce averages gradients across all workers.
If worker A is in AMP mode (gradients internally scaled by S via GradScaler)
and worker B is in Normal mode (gradients unscaled), the averaged gradient
is a meaningless mixture of two different magnitudes.

Solution (matches the Korel paper's Fig. 3b / Algorithm 2, line 14)
---------------------------------------------------------------------
The paper's diagram shows exactly ONE all-reduce per iteration — the
same gradient synchronization DDP already performs — followed by a
LOCAL "Unscale Operation" on each worker. There is no separate
communication step to agree on S in the paper's design; S is a value
every worker already knows without needing to ask.

This implementation follows that design: S is a FIXED constant shared
by construction (both files import the same GSCM_SCALE), not a live
value queried from any single worker's GradScaler. This means:

  • AMP workers    → use GSCM_SCALE as a static loss-scale multiplier
                     (in place of GradScaler's own adaptive value) so
                     every AMP worker always applies the exact same S.
  • Normal workers → manually multiply loss by GSCM_SCALE before backward().

After DDP allreduce all gradients carry factor S.
Then every worker removes that factor locally — no communication needed:
  • AMP workers    → divide by GSCM_SCALE (mirrors scaler.unscale_, but
                     against the fixed constant rather than a live value)
  • Normal workers → GSCM.unscale_gradients()  (divides by S manually)

Trade-off vs. a fully adaptive GradScaler
------------------------------------------
A standard GradScaler grows/shrinks its scale dynamically in response to
overflow. Here growth is made practically unreachable (a very large
growth_interval) so the scale stays at GSCM_SCALE for the life of the
run, matching what Normal-mode workers assume — zero communication
needed. Backoff on genuine overflow is left enabled (PyTorch requires
backoff_factor < 1.0 regardless), so a real numerical overflow still
correctly drops the scale for that worker rather than silently
continuing with bad gradients; this is a rare, safety-driven exception
to the fixed-constant assumption, not a routine occurrence.

Trade-off vs. the earlier all_reduce-based design
----------------------------------------------------
The previous (removed) all_reduce version could legitimately return
S=1.0 as a true no-op when no worker anywhere was in AMP mode, letting
scale_loss/unscale_gradients skip entirely. This constant-S version has
no way to know that without communication, so it ALWAYS pays the scale/
unscale cost in Normal mode — even with zero stragglers active. This is
kept as cheap as possible (torch._foreach_mul_, one fused kernel call
rather than a Python loop over every parameter) specifically because it
runs unconditionally on every batch.
"""

import torch
from typing import Optional


# Shared fixed scale — every worker (AMP or Normal) already knows this
# value without any runtime negotiation. This IS "S" from the paper.
GSCM_SCALE: float = 1024  #65536.0   # 2**16, matches AMP's typical initial scale


class GSCM:

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ─────────────────────────────────────────────────────────────────────────
    def sync_scale(
        self,
        amp_active: bool,
        scaler: Optional[object] = None,   # kept for call-site compatibility
    ) -> float:
        """
        Return the global gradient scale S for this iteration.

        No communication happens here — S is a fixed constant known to
        every worker by construction (see module docstring). This matches
        the Korel paper's design: GSCM is a LOCAL correction applied
        around the single gradient all-reduce DDP already performs, not
        an additional synchronization round.

        Returns
        -------
        global_scale : float
            Always GSCM_SCALE when any worker might be running AMP.
            (We can't cheaply know whether ANY worker is AMP without
            communication, so Normal-mode workers always scale their
            loss by GSCM_SCALE — this is a no-op in FP32 arithmetic
            range and costs nothing but one multiply.)
        """
        return GSCM_SCALE

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def scale_loss(loss: torch.Tensor, global_scale: float) -> torch.Tensor:
        """
        Normal-mode workers call this to match AMP-mode gradient magnitudes.
        """
        return loss * global_scale

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def unscale_gradients(
        model: torch.nn.Module,
        global_scale: float,
    ) -> None:
        """
        Normal-mode workers call this after DDP allreduce to remove the
        scale factor that was applied to the loss before backward().

        Uses torch._foreach_mul_ (a single fused multi-tensor kernel call)
        instead of a per-parameter Python for-loop. This runs unconditionally
        every batch in Normal mode (we can't cheaply know if any peer is in
        AMP without communication — see sync_scale docstring), so keeping
        this cheap matters: a plain Python loop over ResNet50's ~161
        parameter tensors was measurably slower than train_baseline.py's
        plain backward()/step(), even with zero stragglers anywhere.
        """
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        inv = 1.0 / global_scale
        torch._foreach_mul_(grads, inv)