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
        grad_clip_norm: float | None = None,
        sched_cls: str | None = None,
        sched_factory: typing.Callable[..., LRScheduler] | None = None,
        sched_args: dict[str, typing.Any] | None = None,
    ):
        '''doc'''
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
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
        sched_factory: typing.Callable[..., LRScheduler] | None = None,
        sched_args: dict[str, typing.Any] | None = None,
        disable_scheduler: bool = False,
    ) -> None:
        '''
        Reconfigure optimizer and/or scheduler.
        '''

        # ---------------- lr update ----------------
        if lr is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr

        # ---------------- disable scheduler ----------------
        if disable_scheduler:
            self.scheduler = None
            self._sched_factory = None
            self._sched_cls = None
            self._sched_args = None
            return

        # ---------------- no scheduler changes ----------------
        if (
            sched_cls is None and
            sched_factory is None and
            sched_args is None
        ):
            return

        # ---------------- inherit existing config ----------------
        sched_cls = sched_cls or self._sched_cls
        sched_factory = sched_factory or self._sched_factory

        if sched_factory is None or sched_cls is None:
            raise ValueError(
                'Scheduler configuration incomplete.'
            )

        merged_args = dict(self._sched_args or {})
        if sched_args is not None:
            merged_args.update(sched_args)

        # ---------------- rebuild scheduler ----------------
        self.scheduler = sched_factory(
            self.optimizer,
            **merged_args
        )

        self._sched_cls = sched_cls
        self._sched_factory = sched_factory
        self._sched_args = merged_args
