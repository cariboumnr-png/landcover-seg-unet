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
    'DomainContext',
    'DomainTileMap',
    # functions
    'load_domain',
    'pca_transform',
    'prepare_domain',
    'save_domain',
    # typing

]

# for static check
if typing.TYPE_CHECKING:
    from .io import load_domain, save_domain
    from .mapper import prepare_domain
    from .tilemap import DomainContext, DomainTileMap
    from .transform import pca_transform

def __getattr__(name: str):

    if name in ['load_domain', 'save_domain']:
        return getattr(importlib.import_module('.io', __package__), name)
    if name in ['prepare_domain']:
        return getattr(importlib.import_module('.mapper', __package__), name)
    if name in ['DomainContext', 'DomainTileMap']:
        return getattr(importlib.import_module('.tilemap', __package__), name)
    if name in ['pca_transform']:
        return getattr(importlib.import_module('.transform', __package__), name)

    raise AttributeError(name)
