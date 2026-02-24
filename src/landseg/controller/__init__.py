'''
Top-level namespace for controller.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'Controller',
    'Phase',
    # functions
    'build_controller',
    'generate_phases',
]
# for static check
if typing.TYPE_CHECKING:
    from .builder import build_controller
    from .controller import Controller
    from .phase import Phase, generate_phases

def __getattr__(name: str):

    if name in ['build_controller']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['Controller']:
        return getattr(importlib.import_module('.controller', __package__), name)
    if name in ['Phase', 'generate_phases']:
        return getattr(importlib.import_module('.phase', __package__), name)

    raise AttributeError(name)
