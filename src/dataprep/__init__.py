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
    'map_rasters_to_grid',
    'validate_geometry',
    # typing
    'BlockMeta',
    'ImageStats',
    'GeometrySummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .extraction import (DataBlock, BlockMeta, ImageStats, BlockCachePipeline,
                         Artifacts, DataPaths, Windows)
    from .preprocess import map_rasters_to_grid, validate_geometry, GeometrySummary

def __getattr__(name: str):

    if name in ['DataBlock', 'BlockMeta', 'ImageStats', 'BlockCachePipeline',
                'Artifacts', 'DataPaths', 'Windows']:
        return getattr(importlib.import_module('.extraction', __package__), name)
    if name in ['map_rasters_to_grid', 'validate_geometry', 'GeometrySummary']:
        return getattr(importlib.import_module('.preprocess', __package__), name)

    raise AttributeError(name)
