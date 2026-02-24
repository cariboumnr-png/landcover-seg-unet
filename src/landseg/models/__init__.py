'''
Top-level namespace for `landseg.models`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'build_multihead_unet'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .factory import build_multihead_unet

def __getattr__(name: str):

    if name in ['build_multihead_unet']:
        return getattr(importlib.import_module('.factory', __package__), name)
    raise AttributeError(name)
