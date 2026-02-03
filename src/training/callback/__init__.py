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
    'TrainerLike',
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
    from training.common import TrainerLike
    from .base import Callback
    from .logging import LoggingCallback
    from .phase_infer import InferCallback
    from .phase_train import TrainCallback
    from .phase_val import ValCallback
    from .progress import ProgressCallback
    from .factory import build_callbacks

def __getattr__(name: str):

    if name == 'TrainerLike':
        return importlib.import_module('training.common').TrainerLike
    if name == 'Callback':
        return importlib.import_module('.base', __package__).Callback
    if name == 'LoggingCallback':
        return importlib.import_module('.logging', __package__).LoggingCallback
    if name == 'TrainCallback':
        return importlib.import_module('.phase_train', __package__).TrainCallback
    if name == 'ValCallback':
        return importlib.import_module('.phase_val', __package__).ValCallback
    if name == 'InferCallback':
        return importlib.import_module('.phase_infer', __package__).InferCallback
    if name == 'ProgressCallback':
        return importlib.import_module('.progress', __package__).ProgressCallback
    if name == 'build_callbacks':
        return importlib.import_module('.factory', __package__).build_callbacks

    raise AttributeError(name)
