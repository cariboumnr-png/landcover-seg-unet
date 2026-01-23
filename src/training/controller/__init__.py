'''
Top-level namespace for training.controller.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    'Controller',
    'Phase',
    'generate_phases'
]
# for static check
if typing.TYPE_CHECKING:
    from .controller import Controller
    from .phase import Phase, generate_phases

def __getattr__(name: str):

    if name == 'Controller':
        return importlib.import_module('.controller', __package__).Controller
    if name in ('Phase', 'generate_phases'):
        return getattr(importlib.import_module('.phase', __package__), name)

    raise AttributeError(name)
