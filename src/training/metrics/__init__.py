'''
Top-level namespace for training.metrics.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ConfusionMatrix',
    'ConfusionMatrixConfig',
    # functions
    'build_headmetrics',
    'is_cm_config',
]

# for static check
if typing.TYPE_CHECKING:
    from .conf_matrix import ConfusionMatrix
    from ._types import ConfusionMatrixConfig
    from .factory import build_headmetrics
    from .validator import is_cm_config

def __getattr__(name: str):

    if name == 'ConfusionMatrix':
        return importlib.import_module('.conf_matrix', __package__).ConfusionMatrix
    if name == 'ConfusionMatrixConfig':
        return importlib.import_module('._types', __package__).ConfusionMatrixConfig
    if name == 'build_headmetrics':
        return importlib.import_module('.factory', __package__).build_headmetrics
    if name == 'is_cm_config':
        return importlib.import_module('.validator', __package__).is_cm_config

    raise AttributeError(name)
