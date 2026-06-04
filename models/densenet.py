"""
models/densenet.py
------------------
DenseNet variants adapted for CIFAR-10 / CIFAR-100 (32×32 input).

Differences from the standard ImageNet variants
------------------------------------------------
  • conv0   : 7×7 stride-2 padding-3  →  3×3 stride-1 padding-1
  • pool0   : removed (replaced with Identity)
These changes prevent aggressive spatial downsampling on small CIFAR images.
"""

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'densenet121': tv_models.densenet121,
    'densenet161': tv_models.densenet161,
    'densenet169': tv_models.densenet169,
    'densenet201': tv_models.densenet201,
}


def get_densenet(model_name: str, num_classes: int) -> nn.Module:
    model = _REGISTRY[model_name](weights=None, num_classes=num_classes)
    # Replace the aggressive stem
    model.features.conv0 = nn.Conv2d(
        3, model.features.conv0.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    model.features.pool0 = nn.Identity()
    return model
