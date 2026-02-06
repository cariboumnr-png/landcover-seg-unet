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
]

# for static check
if typing.TYPE_CHECKING:
    from .layout import GridLayout, GridSpec

def __getattr__(name: str):

    if name in ['GridLayout', 'GridSpec']:
        return getattr(importlib.import_module('._types', __package__), name)

    raise AttributeError(name)
