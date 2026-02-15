'''
Top-level namespace for dataprep.

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
    'DataWindows',
    # functions
    'count_label_class',
    'map_rasters',
    'prepare_data',
    'validate_geometry',
    # typing
    'BlockMeta',
    'ImageStats',
    'GeometrySummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .mapper import map_rasters, validate_geometry, GeometrySummary, DataWindows
    from .tiler import DataBlock, BlockMeta, ImageStats, BlockCacheBuilder, BuilderConfig
    # from .partitioner import
    from .pipeline import prepare_data
    from .utils import count_label_class

def __getattr__(name: str):

    if name in ['map_rasters', 'validate_geometry', 'GeometrySummary', 'DataWindows']:
        return getattr(importlib.import_module('.mapper', __package__), name)
    if name in ['DataBlock', 'BlockMeta', 'ImageStats', 'BlockCacheBuilder','BuilderConfig']:
        return getattr(importlib.import_module('.tiler', __package__), name)
    if name in ['prepare_data']:
        return getattr(importlib.import_module('.pipeline', __package__), name)
    if name in ['count_label_class']:
        return getattr(importlib.import_module('.utils', __package__), name)

    raise AttributeError(name)
