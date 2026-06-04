"""
straggler/gscm.py
-----------------
Gradient Scale Consistency Mechanism (GSCM).

Problem
-------
When workers are in mixed precision modes (some AMP/FP16, some FP32),
their gradient magnitudes differ before DDP averages them, corrupting
the update.

Solution
--------
Before every backward pass all workers agree on a single global gradient
scale S via an all_reduce(MAX):

  - AMP  workers advertise GradScaler.get_scale()
  - FP32 workers advertise 1.0

Then:
  - AMP  workers align their GradScaler to S
  - FP32 workers multiply loss by S before backward,
    then divide gradients by S after the allreduce

This ensures all gradients have identical magnitude before averaging.
"""

import torch
import torch.distributed as dist


def sync_grad_scale(local_scale: float, device: torch.device) -> float:
    """
    All-reduce (MAX) the local gradient scale across all workers.

    Parameters
    ----------
    local_scale : float
        This worker's current scale.
        Pass GradScaler.get_scale() for AMP workers, 1.0 for FP32 workers.
    device : torch.device
        The CUDA device for this worker.

    Returns
    -------
    float   The globally agreed gradient scale S.
    """
    scale_tensor = torch.tensor([local_scale], dtype=torch.float32, device=device)
    dist.all_reduce(scale_tensor, op=dist.ReduceOp.MAX)
    return scale_tensor.item()
