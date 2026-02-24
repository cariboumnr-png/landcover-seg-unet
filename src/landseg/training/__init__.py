# pylint: disable=too-many-return-statements
'''
Top-level namespace for training module.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MultiHeadTrainer',
    # functions
    'build_trainer',
]
# for static check
if typing.TYPE_CHECKING:
    from .trainer import MultiHeadTrainer
    from .factory import build_trainer

def __getattr__(name: str):

    if name in ['MultiHeadTrainer']:
        return getattr(importlib.import_module('.trainer', __package__), name)
    if name in ['build_trainer']:
        return getattr(importlib.import_module('.factory', __package__), name)

    raise AttributeError(name)
