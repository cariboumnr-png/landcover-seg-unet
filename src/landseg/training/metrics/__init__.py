'''
Top-level namespace for `landseg.training.metrics`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ConfusionMatrix',
    # functions
    'build_headmetrics',
    'is_cm_config',
]

# for static check
if typing.TYPE_CHECKING:
    from .conf_matrix import ConfusionMatrix
    from .factory import build_headmetrics
    from .validator import is_cm_config

def __getattr__(name: str):

    if name in ['ConfusionMatrix']:
        return getattr(importlib.import_module('.conf_matrix', __package__), name)
    if name in ['build_headmetrics']:
        return getattr(importlib.import_module('.factory', __package__), name)
    if name in ['is_cm_config']:
        return getattr(importlib.import_module('.validator', __package__), name)

    raise AttributeError(name)
