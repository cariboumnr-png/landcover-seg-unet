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
    'get_block_builder',
    'map_rasters',
    'prepare_data',
    'validate_geometry',
    # typing
    'BlockMeta',
    'ImageStats',
    'GeometrySummary',
    'InputConfig',
    'ArtifactConfig',
    'RuntimeConfig',
    'DataprepConfigs',
]

# for static check
if typing.TYPE_CHECKING:
    from .config import InputConfig, ArtifactConfig, RuntimeConfig, DataprepConfigs
    from .mapper import map_rasters, validate_geometry, GeometrySummary, DataWindows
    from .blockbuilder import (DataBlock, BlockMeta, ImageStats, BlockCacheBuilder,
                               BuilderConfig, get_block_builder)
    # from .partitioner import
    from .pipeline import prepare_data
    from .utils import count_label_class

def __getattr__(name: str):

    if name in ['map_rasters', 'validate_geometry', 'GeometrySummary',
                'DataWindows']:
        return getattr(importlib.import_module('.mapper', __package__), name)
    if name in ['DataBlock', 'BlockMeta', 'ImageStats', 'BlockCacheBuilder',
                'BuilderConfig']:
        return getattr(importlib.import_module('.blockbuilder', __package__), name)
    if name in ['prepare_data']:
        return getattr(importlib.import_module('.pipeline', __package__), name)
    if name in ['count_label_class']:
        return getattr(importlib.import_module('.utils', __package__), name)

    raise AttributeError(name)
