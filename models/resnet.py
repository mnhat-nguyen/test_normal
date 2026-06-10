import torch.nn as nn
import torchvision.models as tv_models

_REGISTRY = {
    'resnet18':  tv_models.resnet18,
    'resnet50':  tv_models.resnet50,
    'resnet101': tv_models.resnet101,
    'resnet152': tv_models.resnet152,
}


def get_model(model_name: str, dataset: str) -> nn.Module:
    """
    Build a ResNet variant for CIFAR-10 / CIFAR-100.

    Differences from the ImageNet variant
    ──────────────────────────────────────
    • conv1   : 7×7 stride-2  →  3×3 stride-1  (CIFAR images are 32×32)
    • maxpool : removed (replaced with Identity)
    These two changes follow the standard CIFAR ResNet convention and
    match the setup used in the Korel paper.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    num_classes = 10 if dataset == 'cifar10' else 100

    model = _REGISTRY[model_name](weights=None, num_classes=num_classes)

    # Adapt stem for 32×32 CIFAR input
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model
