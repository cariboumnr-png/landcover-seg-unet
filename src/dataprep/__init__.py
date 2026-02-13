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
    'Artifacts',
    'BlockCachePipeline',
    'DataBlock',
    'DataPaths',
    'Windows',
    # functions
    'build_cache',
    'count_label_class',
    'map_rasters',
    'validate_geometry',
    # typing
    'BlockMeta',
    'ImageStats',
    'GeometrySummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .build import (DataBlock, BlockMeta, ImageStats, BlockCachePipeline,
                        Artifacts, DataPaths, Windows, build_cache)
    from .postprocess import count_label_class
    from .preprocess import map_rasters, validate_geometry, GeometrySummary

def __getattr__(name: str):

    if name in ['DataBlock', 'BlockMeta', 'ImageStats', 'BlockCachePipeline',
                'Artifacts', 'DataPaths', 'Windows', 'build_cache']:
        return getattr(importlib.import_module('.build', __package__), name)
    if name in ['count_label_class']:
        return getattr(importlib.import_module('.postprocess', __package__), name)
    if name in ['map_rasters_to_grid', 'validate_geometry', 'GeometrySummary']:
        return getattr(importlib.import_module('.preprocess', __package__), name)

    raise AttributeError(name)
