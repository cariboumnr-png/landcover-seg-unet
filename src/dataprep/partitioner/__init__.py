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
    'score_blocks',
    # typing
    'BlockScore',
]

# for static check
if typing.TYPE_CHECKING:
    from. score import BlockScore, score_blocks

def __getattr__(name: str):

    if name in ['BlockScore', 'score_blocks']:
        return getattr(importlib.import_module('.score', __package__), name)

    raise AttributeError(name)
