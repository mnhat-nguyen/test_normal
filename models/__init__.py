"""
models/__init__.py
------------------
Unified model registry.

Supported models
----------------
ResNet      : resnet18, resnet34, resnet50, resnet101, resnet152
VGG         : vgg11, vgg13, vgg16, vgg19
DenseNet    : densenet121, densenet161, densenet169, densenet201
EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
MobileNet   : mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

All models are adapted for CIFAR-10 / CIFAR-100 (32x32 input).

Usage
-----
    from models import get_model, list_models
    model = get_model('resnet50', 'cifar10')
    print(list_models())
"""

import torch.nn as nn
from .resnet       import get_resnet,        _REGISTRY as _R_RESNET
from .vgg          import get_vgg,           _REGISTRY as _R_VGG
from .densenet     import get_densenet,      _REGISTRY as _R_DENSENET
from .efficientnet import get_efficientnet,  _REGISTRY as _R_EFFICIENTNET
from .mobilenet    import get_mobilenet,     _REGISTRY as _R_MOBILENET

# flat name -> builder function
_BUILDERS = {}
for _n in _R_RESNET:       _BUILDERS[_n] = get_resnet
for _n in _R_VGG:          _BUILDERS[_n] = get_vgg
for _n in _R_DENSENET:     _BUILDERS[_n] = get_densenet
for _n in _R_EFFICIENTNET: _BUILDERS[_n] = get_efficientnet
for _n in _R_MOBILENET:    _BUILDERS[_n] = get_mobilenet


def get_model(model_name: str, dataset: str) -> nn.Module:
    """
    Build and return the requested model adapted for the given dataset.

    Parameters
    ----------
    model_name : str   One of the supported model names (see list_models()).
    dataset    : str   'cifar10' or 'cifar100'.
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'.\nAvailable:\n{_fmt_list()}"
        )
    if dataset not in ('cifar10', 'cifar100'):
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'cifar10' or 'cifar100'."
        )
    num_classes = 10 if dataset == 'cifar10' else 100
    return _BUILDERS[model_name](model_name, num_classes)


def list_models() -> list:
    """Return a sorted list of all supported model names."""
    return sorted(_BUILDERS.keys())


def _fmt_list() -> str:
    families = {
        'resnet':       sorted(_R_RESNET),
        'vgg':          sorted(_R_VGG),
        'densenet':     sorted(_R_DENSENET),
        'efficientnet': sorted(_R_EFFICIENTNET),
        'mobilenet':    sorted(_R_MOBILENET),
    }
    return '\n'.join(
        f"  {fam:14s}: {', '.join(names)}"
        for fam, names in families.items()
    )


__all__ = ['get_model', 'list_models']
