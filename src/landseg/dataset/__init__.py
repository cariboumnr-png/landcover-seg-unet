'''
Top-level namespace for `landseg.dataset`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataSpecs',
    # functions
    'build_dataspec',
    'load_data',
    'validate_schema'
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .builder import DataSpecs, build_dataspec
    from .load import load_data
    from .validate import validate_schema

def __getattr__(name: str):

    if name in ['DataSpecs', 'build_dataspec']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['load_data']:
        return getattr(importlib.import_module('.load', __package__), name)
    if name in ['validate_schema']:
        return getattr(importlib.import_module('.validate', __package__), name)

    raise AttributeError(name)
