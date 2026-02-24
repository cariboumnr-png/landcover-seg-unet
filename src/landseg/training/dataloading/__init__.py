'''
Top-level namespace for `landseg.training.dataloading`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockConfig',
    'MultiBlockDataset',
    # functions
    'get_dataloaders',
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .dataset import BlockConfig, MultiBlockDataset
    from .loader import get_dataloaders

def __getattr__(name: str):

    if name in ['BlockConfig', 'MultiBlockDataset']:
        return getattr(importlib.import_module('.dataset', __package__), name)
    if name in ['get_dataloaders']:
        return getattr(importlib.import_module('.loader', __package__), name)

    raise AttributeError(name)
