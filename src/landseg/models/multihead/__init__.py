'''
Top-level namespace for `landseg.models.multihead`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BaseMultiheadModel',
    'ConcatConfig',
    'CondConfig',
    'FilmConfig',
    'HeadsState',
    'ModelConfig',
    'MultiHeadUNet',
    # functions
    'get_concat',
    'get_film'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .base import BaseMultiheadModel
    from .concat import get_concat
    from .config import (
        ConcatConfig,
        CondConfig,
        FilmConfig,
        HeadsState,
        ModelConfig,
    )
    from .film import get_film
    from .frame import MultiHeadUNet

def __getattr__(name: str):

    if name in ['BaseMultiheadModel']:
        return getattr(importlib.import_module('.base', __package__), name)
    if name in ['get_concat']:
        return getattr(importlib.import_module('.concat', __package__), name)
    if name in ['ConcatConfig', 'CondConfig', 'FilmConfig', 'HeadsState',
                'ModelConfig']:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['get_film']:
        return getattr(importlib.import_module('.film', __package__), name)
    if name in ['MultiHeadUNet']:
        return getattr(importlib.import_module('.frame', __package__), name)

    raise AttributeError(name)
