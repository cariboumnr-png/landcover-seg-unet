'''
Top-level namespace for `landseg.dataprep.blockbuilder`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockCacheBuilder',
    'BuilderConfig',
    'DataBlock',
    # functions
    'build_blocks',
    # typing
    'BlockMeta',
]

# for static check
if typing.TYPE_CHECKING:
    from .block import BlockMeta, DataBlock
    from .builder import build_blocks
    from .cache import BlockCacheBuilder, BuilderConfig

def __getattr__(name: str):

    if name in ['BlockMeta', 'DataBlock']:
        return getattr(importlib.import_module('.block', __package__), name)
    if name in ['build_blocks']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['BlockCacheBuilder', 'BuilderConfig']:
        return getattr(importlib.import_module('.cache', __package__), name)

    raise AttributeError(name)
