"""
models/efficientnet.py
----------------------
EfficientNet variants for CIFAR-10 / CIFAR-100 (32×32 input).

EfficientNet's stem uses a 3×3 stride-2 conv which halves 32→16px on the
first pass.  We leave the stem unchanged but replace the classifier head
to match num_classes.  The model still works well on CIFAR at this size.
"""

import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'efficientnet_b0': tv_models.efficientnet_b0,
    'efficientnet_b1': tv_models.efficientnet_b1,
    'efficientnet_b2': tv_models.efficientnet_b2,
    'efficientnet_b3': tv_models.efficientnet_b3,
}

# in_features for each EfficientNet variant
_IN_FEATURES = {
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
}


def get_efficientnet(model_name: str, num_classes: int) -> nn.Module:
    model = _REGISTRY[model_name](weights=None)
    in_features = _IN_FEATURES[model_name]
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
