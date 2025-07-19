"""
Retina Image Analysis Project
============================

This package provides tools for analyzing retinal images using AI/ML techniques.

Modules:
--------
- data: Data processing and loading utilities
- models: Machine learning model implementations
- training: Training and validation scripts
- inference: Inference and prediction utilities
- api: REST API for serving models
- utils: General utility functions
"""

__version__ = "0.1.0"
__author__ = "Retina Analysis Team"
__email__ = "contact@retinaanalysis.com"

from . import data, models, utils

__all__ = ["data", "models", "utils"]
