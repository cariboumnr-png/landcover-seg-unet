'''
Top-level namespace for dataset.blocks.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes

    'DataBlock',
    # functions

    # typing

]

# for static check
if typing.TYPE_CHECKING:
    from .block import DataBlock


def __getattr__(name: str):

    if name in ['DataBlock']:
        return getattr(importlib.import_module('.block', __package__), name)

    raise AttributeError(name)
