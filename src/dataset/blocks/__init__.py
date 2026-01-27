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
    'RasterBlockCache',
    'RasterBlockLayout',
    'DataBlock',
    # functions
    'build_data_cache',
    'parse_block_name',
    # typing
    'BlockMetaDict',
]

# for static check
if typing.TYPE_CHECKING:
    from .cache import RasterBlockCache, build_data_cache
    from .block import DataBlock
    from .layout import RasterBlockLayout, parse_block_name
    from .typing import BlockMetaDict

def __getattr__(name: str):

    if name in ['RasterBlockCache', 'build_data_cache']:
        return getattr(importlib.import_module('.cache', __package__), name)
    if name == 'DataBlock':
        return importlib.import_module('.block', __package__).DataBlock
    if name in ['RasterBlockLayout', 'parse_block_name']:
        return getattr(importlib.import_module('.layout', __package__), name)
    if name in ['BlockMetaDict']:
        return getattr(importlib.import_module('.typing', __package__), name)

    raise AttributeError(name)
