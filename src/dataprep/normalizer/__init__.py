'''
Top-level namespace for normalizer.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes

    # functions
    'get_image_stats',
    'normalize_data_blocks',
    # typing

]

# for static check
if typing.TYPE_CHECKING:
    from .normalizer import normalize_data_blocks
    from .stats import get_image_stats

def __getattr__(name: str):

    if name in ['get_image_stats']:
        return getattr(importlib.import_module('.stats', __package__), name)
    if name in ['normalize_data_blocks']:
        return getattr(importlib.import_module('.normalizer', __package__), name)

    raise AttributeError(name)
