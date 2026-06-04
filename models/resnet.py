"""
models/resnet.py
----------------
ResNet variants adapted for CIFAR-10 / CIFAR-100 (32×32 input).

Differences from the standard ImageNet variants
------------------------------------------------
  • conv1   : 7×7 stride-2  →  3×3 stride-1
  • maxpool : removed (replaced with Identity)

These two changes follow the standard CIFAR ResNet convention and
match the setup used in the Korel paper.
"""

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'resnet18':  tv_models.resnet18,
    'resnet34':  tv_models.resnet34,
    'resnet50':  tv_models.resnet50,
    'resnet101': tv_models.resnet101,
    'resnet152': tv_models.resnet152,
}


def get_resnet(model_name: str, num_classes: int) -> nn.Module:
    model = _REGISTRY[model_name](weights=None, num_classes=num_classes)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
