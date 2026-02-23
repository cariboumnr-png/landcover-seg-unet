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
    'get_config',
    'load',
    'save',
    'multihead_loss',
]

# for static check
if typing.TYPE_CHECKING:
    from .ckpts import load, save
    from .comps import TrainerComponents
    from .config import RuntimeConfig, get_config
    from .loss import multihead_loss
    from .state import RuntimeState
    from .trainer import MultiHeadTrainer

def __getattr__(name: str):

    if name in ['load', 'save']:
        return getattr(importlib.import_module('.ckpts', __package__), name)
    if name in ['TrainerComponents']:
        return getattr(importlib.import_module('.comps', __package__), name)
    if name in ['RuntimeConfig', 'get_config']:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['multihead_loss']:
        return getattr(importlib.import_module('.loss', __package__), name)
    if name in ['RuntimeState']:
        return getattr(importlib.import_module('.state', __package__), name)
    if name in ['MultiHeadTrainer']:
        return getattr(importlib.import_module('.trainer', __package__), name)

    raise AttributeError(name)
