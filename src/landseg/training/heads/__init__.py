'''
Top-level namespace for `landseg.training.heads`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'Spec',
    # functions
    'build_headspecs',
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .specs import Spec
    from .factory import build_headspecs

def __getattr__(name: str):

    if name in ['Spec']:
        return getattr(importlib.import_module('.specs', __package__), name)
    if name in ['build_headspecs']:
        return getattr(importlib.import_module('.factory', __package__), name)

    raise AttributeError(name)
