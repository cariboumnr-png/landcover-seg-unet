'''
Top-level namespace for extraction.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'Artifacts',
    'BlockCachePipeline',
    'DataBlock',
    'DataPaths',
    'Windows',
    # functions
    'build_cache',
    # typing
    'BlockMeta',
    'ImageStats',
]

# for static check
if typing.TYPE_CHECKING:
    from .block import DataBlock, BlockMeta, ImageStats
    from .builder import build_cache
    from .cache import BlockCachePipeline, Artifacts, DataPaths, Windows

def __getattr__(name: str):

    if name in ['DataBlock', 'BlockMeta', 'ImageStats']:
        return getattr(importlib.import_module('.block', __package__), name)
    if name in ['build_cache']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['BlockCachePipeline', 'Artifacts', 'DataPaths', 'Windows']:
        return getattr(importlib.import_module('.cache', __package__), name)

    raise AttributeError(name)
