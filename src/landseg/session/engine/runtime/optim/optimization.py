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
import typing
# third-party imports
import torch

# aliases
Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler.LRScheduler

# -------------------------------Public Function-------------------------------
class Optimization:
    '''
    Stateful wrapper for optimizer and scheduler.

    Provides explicit APIs for stepping and resetting.
    '''


    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        *,
        sched_cls: str | None = None,
        sched_factory: typing.Callable[..., LRScheduler] | None = None,
        sched_args: dict[str, typing.Any] | None = None
    ):
        '''doc'''
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._sched_cls = sched_cls
        self._sched_factory = sched_factory
        self._sched_args = sched_args

    # -------------------------- public APIs --------------------------
    @property
    def lrs(self) -> list[float]:
        '''Return the learning rates.'''

        return [g['lr'] for g in self.optimizer.param_groups]

    def step_optimizer(self) -> None:
        '''Step the optimizer.'''

        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        '''Clear gradients.'''

        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step_scheduler(self) -> None:
        '''Step scheduler if present.'''

        if self.scheduler is not None:
            self.scheduler.step()

    def reconfigure(
        self,
        *,
        lr: float | None = None,
        sched_cls: str | None = None,
        sched_factory: typing.Callable[
            ..., torch.optim.lr_scheduler.LRScheduler
        ] | None = None,
        sched_args: dict[str, typing.Any] | None = None
    ) -> None:
        '''
        Reconfigure optimizer + scheduler in one call.
        '''

        if lr is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr

        if sched_factory is None or sched_cls is None:
            self.scheduler = None
            self._sched_factory = None
            self._sched_cls = None
            self._sched_args = None
            return

        args = sched_args or {}

        self.scheduler = sched_factory(self.optimizer, **args)
        self._sched_factory = sched_factory
        self._sched_cls = sched_cls
        self._sched_args = args

    def reset_scheduler(
        self,
        *,
        sched_cls: str | None,
        sched_factory: typing.Callable[..., LRScheduler] | None,
        sched_args: dict[str, typing.Any] | None = None
    ) -> None:
        '''
        Rebuild scheduler with new configuration.

        Args:
            sched_cls: name for tracking only (debug/logging)
            sched_factory: callable to construct scheduler
            sched_args: kwargs passed to scheduler
        '''

        if sched_factory is None or sched_cls is None:
            self.scheduler = None
            self._sched_cls = None
            self._sched_factory = None
            self._sched_args = None
            return

        args = sched_args or {}

        self.scheduler = sched_factory(self.optimizer, **args)
        self._sched_cls = sched_cls
        self._sched_factory = sched_factory
        self._sched_args = args

    def rebuild_scheduler(self) -> None:
        '''
        Recreate scheduler using stored configuration.

        Useful after optimizer state changes or resume.
        '''

        if self._sched_factory is None:
            self.scheduler = None
            return

        self.scheduler = self._sched_factory(
            self.optimizer,
            **(self._sched_args or {})
        )
