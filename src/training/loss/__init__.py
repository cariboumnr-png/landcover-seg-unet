'''
Top-level namespace for training.loss.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'PrimitiveLoss',
    'CompositeLoss',
    'DiceLoss',
    'FocalLoss',
    'LossTypes',
    # functions
    'build_headlosses',
    'is_loss_types',
]
# for static check
if typing.TYPE_CHECKING:
    from .base import PrimitiveLoss
    from .composite import CompositeLoss
    from .primitives import DiceLoss, FocalLoss
    from ._types import LossTypes
    from .factory import build_headlosses
    from .validator import is_loss_types


def __getattr__(name: str):

    if name == 'PrimitiveLoss':
        return importlib.import_module('.base', __package__).PrimitiveLoss
    if name == 'CompositeLoss':
        return importlib.import_module('.composite', __package__).CompositeLoss
    if name in ('DiceLoss', 'FocalLoss'):
        return getattr(importlib.import_module('.primitives', __package__), name)
    if name in ('LossTypes',):
        return importlib.import_module('._types', __package__).LossTypes
    if name == 'build_headlosses':
        return importlib.import_module('.factory', __package__).build_headlosses
    if name == 'is_loss_types':
        return importlib.import_module('.validator', __package__).is_loss_types

    raise AttributeError(name)
