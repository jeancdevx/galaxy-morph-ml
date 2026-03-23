"""Model architectures for GalaxyMorph."""

from .resnet import GalaxyMorphResNet50, create_model

__all__ = [
    "GalaxyMorphResNet50",
    "create_model",
]
