'''
Top-level namespace for landseg.cli.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'train_end_to_end'
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .end_to_end import train_end_to_end

def __getattr__(name: str):

    if name in ['train_end_to_end']:
        return getattr(importlib.import_module('.end_to_end', __package__), name)

    raise AttributeError(name)
