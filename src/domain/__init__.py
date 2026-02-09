'''
Top-level namespace for domain.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes

    # functions
    'map_domain_to_grid',
    'pca_transform',
    # typing

]

# for static check
if typing.TYPE_CHECKING:
    from .resolver import map_domain_to_grid
    from .transform import pca_transform

def __getattr__(name: str):

    if name in ['map_domain_to_grid']:
        return getattr(importlib.import_module('.resolver', __package__), name)
    if name in ['pca_transform']:
        return getattr(importlib.import_module('.transform', __package__), name)

    raise AttributeError(name)
