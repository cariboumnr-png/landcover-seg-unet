# pylint: disable=missing-function-docstring, too-few-public-methods
'''
Protocols for optimizer creation.
'''

from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch
    import torch.nn
    import torch.optim
    import torch.optim.lr_scheduler

@typing.runtime_checkable
class ModelWithParams(typing.Protocol):
    '''Minimal protocol for a model that contains torch parameters.'''
    def parameters(self) -> typing.Iterable['torch.nn.Parameter']: ...

# a callable that constructs an Optimizer (e.g., torch.optim.AdamW)
p1 = typing.ParamSpec('p1')
OptimizerFactory: typing.TypeAlias = typing.Callable[p1, "torch.optim.Optimizer"]

# a callable that constructs a Scheduler (e.g., CosineAnnealingLR)
p2 = typing.ParamSpec('p2')
SchedulerFactory: typing.TypeAlias = typing.Callable[p2, "torch.optim.lr_scheduler.LRScheduler"]
