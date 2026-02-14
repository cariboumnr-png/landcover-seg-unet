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
    'BlockCacheBuilder',
    'DataBlock',
    'BuilderConfig',
    # functions

    # typing
    'BlockMeta',
    'ImageStats',
]

# for static check
if typing.TYPE_CHECKING:
    from .block import DataBlock, BlockMeta, ImageStats
    from .cache import BlockCacheBuilder, BuilderConfig

def __getattr__(name: str):

    if name in ['DataBlock', 'BlockMeta', 'ImageStats']:
        return getattr(importlib.import_module('.block', __package__), name)
    if name in ['BlockCacheBuilder', 'BuilderConfig']:
        return getattr(importlib.import_module('.cache', __package__), name)

    raise AttributeError(name)
