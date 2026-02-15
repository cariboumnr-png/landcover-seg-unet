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
    'count_label_class'
    # typing

]

# for static check
if typing.TYPE_CHECKING:
    from .count import count_label_class

def __getattr__(name: str):

    if name in ['count_label_class']:
        return getattr(importlib.import_module('.count', __package__), name)

    raise AttributeError(name)
