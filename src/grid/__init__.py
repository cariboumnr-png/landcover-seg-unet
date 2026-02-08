'''
Top-level namespace for grid.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'GridLayout',
    'GridSpec',
    # functions
    'prep_world_grid',
    'load_grid',
    'save_grid',
    # typing
    'GridLayoutPayload',
]

# for static check
if typing.TYPE_CHECKING:
    from .builder import prep_world_grid
    from .io import load_grid, save_grid
    from .layout import GridLayout, GridLayoutPayload, GridSpec

def __getattr__(name: str):

    if name in ['prep_world_grid']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['load_grid', 'save_grid']:
        return getattr(importlib.import_module('.io', __package__), name)
    if name in ['GridLayout', 'GridSpec', 'GridLayoutPayload']:
        return getattr(importlib.import_module('.layout', __package__), name)

    raise AttributeError(name)
