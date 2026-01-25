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
    'DataBlock',
    'RasterBlockLayout',
    'CacheConfig',
    'CachePaths',
    # functions
    'parse_block_name',
    'tile_rasters',
    'create_block_cache',
    'validate_blocks_cache'
]

# for static check
if typing.TYPE_CHECKING:
    from .cache import (
        CacheConfig,
        CachePaths,
        tile_rasters,
        create_block_cache,
        validate_blocks_cache
    )
    from .block import DataBlock
    from .layout import RasterBlockLayout, parse_block_name

def __getattr__(name: str):

    if name in ['tile_rasters', 'create_block_cache', 'validate_blocks_cache']:
        return getattr(importlib.import_module('.cache', __package__), name)
    if name in ['CacheConfig', 'CachePaths']:
        return getattr(importlib.import_module('.cache', __package__), name)
    if name == 'DataBlock':
        return importlib.import_module('.block', __package__).DataBlock
    if name in ['RasterBlockLayout', 'parse_block_name']:
        return getattr(importlib.import_module('.layout', __package__), name)

    raise AttributeError(name)
