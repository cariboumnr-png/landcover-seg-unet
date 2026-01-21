'''
Top-level namespace for training.trainer.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MultiHeadTrainer',
    'TrainerComponents',
    'RuntimeConfig',
    'RuntimeState',
    # functions
    'load',
    'save',
    'multihead_loss',
]

# for static check
if typing.TYPE_CHECKING:
    from .ckpts import load, save
    from .comps import TrainerComponents
    from .config import RuntimeConfig
    from .state import RuntimeState
    from .trainer import MultiHeadTrainer
    from .loss import multihead_loss

def __getattr__(name: str):

    if name in ['load', 'save']:
        return getattr(importlib.import_module('.ckpts', __package__), name)
    if name == 'TrainerComponents':
        return importlib.import_module('.comps', __package__).TrainerComponents
    if name == 'RuntimeConfig':
        return importlib.import_module('.config', __package__).RuntimeConfig
    if name == 'RuntimeState':
        return importlib.import_module('.state', __package__).RuntimeState
    if name == 'MultiHeadTrainer':
        return importlib.import_module('.trainer', __package__).MultiHeadTrainer
    if name == 'multihead_loss':
        return importlib.import_module('.loss', __package__).multihead_loss

    raise AttributeError(name)
