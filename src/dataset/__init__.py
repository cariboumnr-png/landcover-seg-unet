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
    'DataBlock',
    'DataSummary',
    # functions
    'build_data_cache',
    'build_domains',
    'count_label_cls',
    'normalize_dataset',
    'prepare_data',
    'split_dataset',
    # typing
    'BlockCreationOptions',
    'BlockLayout',
]

# for static check
if typing.TYPE_CHECKING:
    from .blocks import BlockCreationOptions, DataBlock, BlockLayout, build_data_cache
    from .count import count_label_cls
    from .domain import build_domains
    from .factory import prepare_data
    from .norm import normalize_dataset
    from .split import split_dataset
    from .summary import DataSummary

def __getattr__(name: str):

    if name in ['BlockCreationOptions','DataBlock', 'BlockLayout', 'build_data_cache']:
        return getattr(importlib.import_module('.blocks', __package__), name)
    if name in ['count_label_cls']:
        return getattr(importlib.import_module('.count', __package__), name)
    if name == 'build_domains':
        return importlib.import_module('.domain', __package__).build_domains
    if name == 'prepare_data':
        return importlib.import_module('.factory', __package__).prepare_data
    if name == 'normalize_dataset':
        return importlib.import_module('.norm', __package__).normalize_dataset
    if name == 'split_dataset':
        return importlib.import_module('.split', __package__).split_dataset
    if name == 'DataSummary':
        return importlib.import_module('.summary', __package__).DataSummary

    raise AttributeError(name)
