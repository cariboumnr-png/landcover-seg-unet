'''
Top-level namespace for models.multihead.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BaseModel',
    'ConcatAdapter',
    'ConcatConfig',
    'CondConfig',
    'FilmConfig',
    'ModelConfig',
    'FilmConditioner',
    'MultiHeadUNet',
    # functions
    'get_concat',
    'get_film'
]

# for static check
if typing.TYPE_CHECKING:
    from .base import BaseModel
    from .concat import ConcatAdapter, get_concat
    from .config import ConcatConfig, CondConfig, FilmConfig, ModelConfig
    from .film import FilmConditioner, get_film
    from .frame import MultiHeadUNet

def __getattr__(name: str):

    if name == 'BaseModel':
        return importlib.import_module('.base', __package__).BaseModel
    if name == ['ConcatAdapter', 'get_concat']:
        return getattr(importlib.import_module('.concat', __package__), name)
    if name in ['ConcatConfig', 'CondConfig', 'FilmConfig', 'ModelConfig']:
        return getattr(importlib.import_module('.base', __package__), name)
    if name == ['FilmConditioner', 'get_film']:
        return getattr(importlib.import_module('.film', __package__), name)
    if name == 'MultiHeadUNet':
        return importlib.import_module('.frame', __package__).MultiHeadUNet

    raise AttributeError(name)
