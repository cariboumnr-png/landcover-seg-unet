'''
Top-level namespace for training.optim.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing
__all__ = [
    # classes
    'ModelWithParams',
    'OptimizerFactory',
    'SchedulerFactory',
    # functions
    'build_optim_config',
    'build_optimization'
]

# for static check
if typing.TYPE_CHECKING:
    from .optimizer import build_optim_config, build_optimization
    from ._types import ModelWithParams, OptimizerFactory, SchedulerFactory

def __getattr__(name: str):

    if name in ('build_optim_config', 'build_optimization'):
        return getattr(importlib.import_module('.optimizer', __package__), name)
    if name in ('ModelWithParams', 'OptimizerFactory', 'SchedulerFactory'):
        return getattr(importlib.import_module('._types', __package__), name)

    raise AttributeError(name)
