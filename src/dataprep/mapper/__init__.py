'''
Top-level namespace for preprocess.

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
    from .geometry import validate_geometry, GeometrySummary
    from .mapper import map_rasters, DataWindows

def __getattr__(name: str):

    if name in [ 'validate_geometry', 'GeometrySummary']:
        return getattr(importlib.import_module('.geometry', __package__), name)
    if name in ['map_rasters', 'DataWindows']:
        return getattr(importlib.import_module('.mapper', __package__), name)

    raise AttributeError(name)
