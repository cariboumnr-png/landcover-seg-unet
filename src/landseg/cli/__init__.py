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
    'overfit_test',
    'train_end_to_end',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .end_to_end import train_end_to_end
    from .overfit import overfit_test

def __getattr__(name: str):

    if name in ['train_end_to_end']:
        return getattr(importlib.import_module('.end_to_end', __package__), name)
    if name in ['overfit_test']:
        return getattr(importlib.import_module('.overfit', __package__), name)

    raise AttributeError(name)
