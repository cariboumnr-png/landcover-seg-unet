'''
Top-level namespace for `landseg.training.optim`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing
__all__ = [
    # classes
    # functions
    'build_optimization'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .optimizer import build_optimization

def __getattr__(name: str):

    if name in ['build_optimization']:
        return getattr(importlib.import_module('.optimizer', __package__), name)

    raise AttributeError(name)
