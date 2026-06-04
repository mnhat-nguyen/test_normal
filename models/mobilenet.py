"""
models/mobilenet.py
-------------------
MobileNet variants for CIFAR-10 / CIFAR-100 (32×32 input).

MobileNetV2 and MobileNetV3 are lightweight models well-suited to
bandwidth-constrained distributed training experiments (lower compute
per batch → faster iteration, easier to observe straggler effects).

The classifier head is replaced to match num_classes.
"""

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'mobilenet_v2':        tv_models.mobilenet_v2,
    'mobilenet_v3_small':  tv_models.mobilenet_v3_small,
    'mobilenet_v3_large':  tv_models.mobilenet_v3_large,
}

_IN_FEATURES = {
    'mobilenet_v2':        1280,
    'mobilenet_v3_small':  1024,
    'mobilenet_v3_large':  1280,
}


def get_mobilenet(model_name: str, num_classes: int) -> nn.Module:
    model       = _REGISTRY[model_name](weights=None)
    in_features = _IN_FEATURES[model_name]

    if model_name == 'mobilenet_v2':
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
    else:
        # MobileNetV3 has a 3-layer classifier; replace only the last Linear
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )

    return model
