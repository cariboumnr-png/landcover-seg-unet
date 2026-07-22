# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

# pylint: disable=missing-function-docstring

'''
Optimizer and scheduler construction utilities.

Defines configuration protocols, registries, and factory functions for
building optimizer and scheduler pairs from user-provided settings.

This module translates lightweight configuration into fully initialized
runtime optimization components used by the training engine.
'''

# standard imports
import typing
# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.engine.runtime.optim as optim

# ---------------------------------Public Type---------------------------------
class OptimConfigShape(typing.Protocol):
    '''Configuration interface for building optimizer and scheduler.'''
    @property
    def opt_cls(self) -> str: ...
    @property
    def lr(self) -> float: ...
    @property
    def weight_decay(self) -> float: ...
    @property
    def sched_cls(self) -> str | None: ...
    @property
    def sched_args(self) -> dict[str, typing.Any]: ...
    @property
    def grad_clip_norm(self) -> float | None: ...

# a callable that constructs an Optimizer (e.g., torch.optim.AdamW)
P = typing.ParamSpec('P')
OptimizerFactory = typing.Callable[P, 'torch.optim.Optimizer']

# a callable that constructs a Scheduler (e.g., CosineAnnealingLR)
S = typing.ParamSpec('S')
SchedulerFactory = typing.Callable[S, 'torch.optim.lr_scheduler.LRScheduler']

_OPTIMIZERS: dict[str, OptimizerFactory] = {
    'AdamW': torch.optim.AdamW,
    'SGD': torch.optim.SGD,
}
_SCHEDULERS: dict[str, SchedulerFactory] = {
    'CosAnneal': torch.optim.lr_scheduler.CosineAnnealingLR,
    'OneCycle': torch.optim.lr_scheduler.OneCycleLR,
}

# -------------------------------Public Function-------------------------------
def build_optimization(
    model: core.MultiheadModelLike,
    config: OptimConfigShape
) -> optim.Optimization:
    '''
    Build an optimization wrapper from model and configuration.

    Constructs a PyTorch optimizer and an optional learning rate
    scheduler based on the provided configuration, returning a unified
    ``Optimization`` wrapper for runtime use.

    Args:
        model: Model exposing parameters to optimize.
        config: Optimization configuration specifying optimizer type,
            learning rate, weight decay, scheduler settings, and
            gradient clipping.

    Returns:
        Optimization:
            Wrapper containing the optimizer and optional scheduler.

    Raises:
        ValueError:
            If the specified optimizer or scheduler is not registered.

    Notes:
        - Scheduler is optional and only created if configured.
        - Scheduler construction uses a registry-backed factory.
        - Configuration is preserved to support future reconfiguration.
    '''

    # build the optimizer
    optimizer = _build_optimizer(
        model,
        optim_cls=config.opt_cls,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # exit if scheduler class in not configured
    if config.sched_cls is None:
        return optim.Optimization(
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm
        )

    # set scheduler
    sched_factory = _SCHEDULERS.get(config.sched_cls)
    if sched_factory is None:
        raise ValueError(f'Unknown scheduler: {config.sched_cls}')
    scheduler = sched_factory(optimizer, **config.sched_args)

    # return
    return optim.Optimization(
        optimizer,
        scheduler,
        grad_clip_norm=config.grad_clip_norm,
        sched_cls=config.sched_cls,
        sched_factory=sched_factory,
        sched_args=config.sched_args
    )

# ------------------------------private  function------------------------------
def _build_optimizer(
    model: core.MultiheadModelLike,
    *,
    optim_cls: str,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    '''Instantiate an optimizer from the registry.'''

    optimizer_class = _OPTIMIZERS.get(optim_cls)
    if optimizer_class is None:
        raise ValueError(f'Unknown optimizer: {optim_cls}')
    return optimizer_class(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
