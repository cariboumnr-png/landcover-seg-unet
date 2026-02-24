'''
Top-level namespace for `landseg.training.common`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    # types
    'CallBacksLike',
    'CompositeLossLike',
    'DataLoadersLike',
    'DataSpecsLike',
    'HeadLossesLike',
    'HeadMetricsLike',
    'HeadSpecsLike',
    'InferCallbackLike',
    'LoggingCallbackLike',
    'MetricLike',
    'MultiheadModelLike',
    'OptimizationLike',
    'ProgressCallbackLike',
    'RuntimeConfigLike',
    'RuntimeStateLike',
    'SpecLike',
    'TrainerCallbackLike',
    'TrainerComponentsLike',
    'TrainerLike',
    'ValCallbackLike',
]
# for static check
if typing.TYPE_CHECKING:
    from .dataset import DataSpecsLike
    from .trainer import TrainerLike
    from .trainer_comps import (
        CallBacksLike,
        CompositeLossLike,
        DataLoadersLike,
        HeadLossesLike,
        HeadMetricsLike,
        HeadSpecsLike,
        InferCallbackLike,
        LoggingCallbackLike,
        MetricLike,
        MultiheadModelLike,
        OptimizationLike,
        ProgressCallbackLike,
        SpecLike,
        TrainerCallbackLike,
        TrainerComponentsLike,
        ValCallbackLike,
    )
    from .trainer_config import RuntimeConfigLike
    from .trainer_state import RuntimeStateLike

def __getattr__(name: str):

    if name in ['DataSpecsLike']:
        return getattr(importlib.import_module('.dataset', __package__), name)
    if name in ['TrainerLike']:
        return getattr(importlib.import_module('.trainer', __package__), name)
    if name in [
        'CallBacksLike',
        'CompositeLossLike',
        'DataLoadersLike',
        'HeadSpecsLike',
        'HeadLossesLike',
        'HeadMetricsLike',
        'InferCallbackLike',
        'LoggingCallbackLike',
        'MetricLike',
        'MultiheadModelLike',
        'OptimizationLike',
        'ProgressCallbackLike',
        'SpecLike',
        'TrainerCallbackLike',
        'TrainerComponentsLike',
        'ValCallbackLike'
    ]:
        return getattr(importlib.import_module('.trainer_comps', __package__), name)
    if name in ['RuntimeConfigLike']:
        return getattr(importlib.import_module('.trainer_config', __package__), name)
    if name in ['RuntimeStateLike']:
        return getattr(importlib.import_module('.trainer_state', __package__), name)

    raise AttributeError(name)
