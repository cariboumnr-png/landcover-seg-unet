'''
Top-level namespace for `landseg.core`.

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
    'DataSpecsLike'
]

# for static check
if typing.TYPE_CHECKING:
    from .protocols import DataSpecsLike

def __getattr__(name: str):

    if name in ['DataSpecsLike']:
        return getattr(importlib.import_module('.protocols', __package__), name)

    raise AttributeError(name)
