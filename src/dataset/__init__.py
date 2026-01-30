# pylint: disable=too-many-return-statements
'''
Top-level namespace for dataset.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockLayout',
    'DataBlock',
    'DataSummary',
    # functions
    'build_data_cache',
    'build_domains',
    'count_label_cls',
    'generate_summary',
    'normalize_dataset',
    'prepare_data',
    'split_dataset',
]

# for static check
if typing.TYPE_CHECKING:
    from .blocks import DataBlock, BlockLayout, build_data_cache
    from .count import count_label_cls
    from .domain import build_domains
    from .factory import prepare_data
    from .norm import normalize_dataset
    from .split import split_dataset
    from .summary import DataSummary, generate_summary

def __getattr__(name: str):

    if name in ['DataBlock', 'BlockLayout', 'build_data_cache']:
        return getattr(importlib.import_module('.blocks', __package__), name)
    if name in ['count_label_cls']:
        return getattr(importlib.import_module('.count', __package__), name)
    if name in ['build_domains']:
        return getattr(importlib.import_module('.domain', __package__), name)
    if name in ['prepare_data']:
        return getattr(importlib.import_module('.factory', __package__), name)
    if name in ['normalize_dataset']:
        return getattr(importlib.import_module('.norm', __package__), name)
    if name in ['split_dataset']:
        return getattr(importlib.import_module('.split', __package__), name)
    if name in ['DataSummary', 'generate_summary']:
        return getattr(importlib.import_module('.summary', __package__), name)

    raise AttributeError(name)
