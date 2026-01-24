'''
Top-level namespace for training.heads.

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
]

# for static check
if typing.TYPE_CHECKING:
    from .specs import Spec
    from .factory import build_headspecs

def __getattr__(name: str):

    if name == 'Spec':
        return importlib.import_module('.specs', __package__).Spec
    if name == 'build_headspecs':
        return importlib.import_module('.factory', __package__).build_headspecs

    raise AttributeError(name)
