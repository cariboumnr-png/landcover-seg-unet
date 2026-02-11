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

    # functions
    'map_rasters_to_grid',
    'validate_geometry',
    # typing
    'GeometrySummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .preprocess import map_rasters_to_grid, validate_geometry, GeometrySummary

def __getattr__(name: str):

    if name in ['map_rasters_to_grid', 'validate_geometry', 'GeometrySummary']:
        return getattr(importlib.import_module('.preprocess', __package__), name)

    raise AttributeError(name)
