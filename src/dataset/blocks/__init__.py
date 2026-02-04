'''
Top-level namespace for dataset.blocks.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockLayout',
    'DataBlock',
    # functions
    'build_data_cache',
    # typing
    'BlockMeta',
    'ImageStats'
]

# for static check
if typing.TYPE_CHECKING:
    from ._types import BlockMeta, ImageStats
    from .block import DataBlock
    from .cache import build_data_cache
    from .layout import BlockLayout

def __getattr__(name: str):

    if name in ['BlockMeta', 'ImageStats']:
        return getattr(importlib.import_module('._types', __package__), name)
    if name in ['DataBlock']:
        return getattr(importlib.import_module('.block', __package__), name)
    if name in ['build_data_cache']:
        return getattr(importlib.import_module('.cache', __package__), name)
    if name in ['BlockLayout']:
        return getattr(importlib.import_module('.layout', __package__), name)

    raise AttributeError(name)
