'''
Top-level namespace for `landseg.dataprep.mapper`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataWindows',
    # functions
    'map_rasters',
    'validate_geometry',
    # typing
    'GeometrySummary'
]

# for static check
if typing.TYPE_CHECKING:
    from .geometry import GeometrySummary, validate_geometry
    from .mapper import DataWindows, map_rasters

def __getattr__(name: str):

    if name in ['GeometrySummary', 'validate_geometry']:
        return getattr(importlib.import_module('.geometry', __package__), name)
    if name in ['DataWindows', 'map_rasters']:
        return getattr(importlib.import_module('.mapper', __package__), name)

    raise AttributeError(name)
