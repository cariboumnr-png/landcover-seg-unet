'''
Top-level namespace for partition.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes

    # functions

    # typing

]

# for static check
if typing.TYPE_CHECKING:
    pass

def __getattr__(name: str):

    if name in ['...']:
        return getattr(importlib.import_module('...', __package__), name)

    raise AttributeError(name)
