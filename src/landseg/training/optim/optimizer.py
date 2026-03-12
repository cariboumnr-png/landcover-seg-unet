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

# pylint: disable=missing-function-docstring, too-few-public-methods
'''
Simple optimizer and scheduler factory utilities.

Provides:
    - A minimal Protocol for models exposing `.parameters()`.
    - Registries for common optimizers and schedulers.
    - A factory to build (optimizer, scheduler) pairs from config.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import torch
# local imports
import landseg.core as core

# ---------------------------------Public Type---------------------------------
# a callable that constructs an Optimizer (e.g., torch.optim.AdamW)
p1 = typing.ParamSpec('p1')
OptimizerFactory: typing.TypeAlias = typing.Callable[p1, "torch.optim.Optimizer"]

# a callable that constructs a Scheduler (e.g., CosineAnnealingLR)
p2 = typing.ParamSpec('p2')
SchedulerFactory: typing.TypeAlias = typing.Callable[p2, "torch.optim.lr_scheduler.LRScheduler"]

_OPTIMIZERS: dict[str, OptimizerFactory] = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}
_SCHEDULERS: dict[str, SchedulerFactory] = {
    "CosAnneal": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycle": torch.optim.lr_scheduler.OneCycleLR,
}

# -------------------------------Public Function-------------------------------
@dataclasses.dataclass
class Optimization:
    '''Container for optimizer and optional scheduler.'''
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

# -------------------------------Public Function-------------------------------
def build_optimization(
    model: core.MultiheadModelLike,
    opt_cls: str,
    lr: float,
    weight_decay: float,
    sched_cls: str | None,
    **sched_args
) -> Optimization:
    '''
    Build optimizer and scheduler from a config.

    Expected config options:
    - 'opt_cls'       : optimizer key in _OPTIMIZERS (e.g., 'AdamW')
    - 'lr'            : learning rate (float)
    - 'weight_decay'  : weight decay (float)
    - 'sched_cls'     : scheduler key in _SCHEDULERS or None
    - 'sched_args'    : dict of scheduler kwargs (if sched_cls set)

    Returns:
        Optimization(optimizer, scheduler_or_none).
    '''

    optimizer = _build_optimizer(model, opt_cls, lr, weight_decay)
    scheduler = _build_scheduler(optimizer, sched_cls, **sched_args)
    return Optimization(optimizer=optimizer, scheduler=scheduler)

# ------------------------------private  function------------------------------
def _build_optimizer(
    model: core.MultiheadModelLike,
    optim_cls: str,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    '''Instantiate an optimizer from the registry.'''

    optimizer_class = _OPTIMIZERS.get(optim_cls)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optim_cls}")
    return optimizer_class(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cls: str | None,
    **sched_args
) -> torch.optim.lr_scheduler.LRScheduler | None:
    '''Instantiate a scheduler from the registry if requested.'''

    if sched_cls is None:
        return None
    scheduler = _SCHEDULERS.get(sched_cls)
    if scheduler is None:
        raise ValueError(f"Unknown scheduler: {sched_cls}")
    return scheduler(optimizer, **sched_args)
