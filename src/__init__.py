"""
GalaxyMorph: Galaxy Morphology Classification using CNN

A PyTorch-based pipeline for automatic classification of galaxy morphology
from astronomical images using ResNet50 backbone.
"""

__version__ = "0.1.0"
__author__ = "GalaxyMorph Team"

from . import config
from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]
