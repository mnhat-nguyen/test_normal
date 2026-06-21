from .resnet import get_model

__all__ = ['get_model']

def list_models():
    return ['resnet18', 'resnet50', 'resnet101', 'resnet152']