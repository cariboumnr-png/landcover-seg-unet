# pylint: disable=too-many-return-statements
'''
Top-level namespace for training.callback.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'Callback',
    'LoggingCallback',
    'TrainCallback',
    'ValCallback',
    'InferCallback',
    'ProgressCallback',
    # functions
    'build_callbacks'
]

# for static check
if typing.TYPE_CHECKING:
    from .base import Callback
    from .factory import build_callbacks
    from .logging import LoggingCallback
    from .phase_infer import InferCallback
    from .phase_train import TrainCallback
    from .phase_val import ValCallback
    from .progress import ProgressCallback

def __getattr__(name: str):

    if name in ['Callback']:
        return getattr(importlib.import_module('.base', __package__), name)
    if name in ['build_callbacks']:
        return getattr(importlib.import_module('.factory', __package__), name)
    if name in ['LoggingCallback']:
        return getattr(importlib.import_module('.logging', __package__), name)
    if name in ['InferCallback']:
        return getattr(importlib.import_module('.phase_infer', __package__), name)
    if name in ['TrainCallback']:
        return getattr(importlib.import_module('.phase_train', __package__), name)
    if name in ['ValCallback']:
        return getattr(importlib.import_module('.phase_val', __package__), name)
    if name in ['ProgressCallback']:
        return getattr(importlib.import_module('.progress', __package__), name)

    raise AttributeError(name)
