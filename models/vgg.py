"""
models/vgg.py
-------------
VGG variants adapted for CIFAR-10 / CIFAR-100 (32×32 input).

Differences from the standard ImageNet variants
------------------------------------------------
  • The classifier head is replaced with a single Linear(512, num_classes)
    because CIFAR feature maps going into the classifier are 1×1 after the
    five pooling stages, not 7×7 as in ImageNet.
  • adaptive_avg_pool2d(1) ensures the model works regardless of input size.
"""

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'vgg11': tv_models.vgg11,
    'vgg13': tv_models.vgg13,
    'vgg16': tv_models.vgg16,
    'vgg19': tv_models.vgg19,
}


def get_vgg(model_name: str, num_classes: int) -> nn.Module:
    model = _REGISTRY[model_name](weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, num_classes)
    return model
