'''
Top-level namespace for `landseg.dataprep.splitter`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'count_label_class',
    'select_val_blocks',
    'score_blocks',
    'split_blocks',
    # typing
    'BlockScore',
    'ScoreParams',
]

# for static check
if typing.TYPE_CHECKING:
    from .counter import count_label_class
    from .score import BlockScore, ScoreParams, score_blocks
    from .selector import select_val_blocks
    from .splitter import split_blocks

def __getattr__(name: str):

    if name in ['count_label_class']:
        return getattr(importlib.import_module('.counter', __package__), name)
    if name in ['BlockScore', 'ScoreParams', 'score_blocks']:
        return getattr(importlib.import_module('.score', __package__), name)
    if name in ['select_val_blocks']:
        return getattr(importlib.import_module('.selector', __package__), name)
    if name in ['split_blocks']:
        return getattr(importlib.import_module('.splitter', __package__), name)

    raise AttributeError(name)
