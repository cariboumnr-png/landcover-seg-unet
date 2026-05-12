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

'''
Optimizer and scheduler orchestration utilities.

Defines a lightweight wrapper around PyTorch optimizers and learning
rate schedulers, providing a unified interface for stepping, gradient
management, and dynamic reconfiguration during runtime.

The module focuses on orchestration-level control rather than optimizer
implementation details.
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
    Runtime wrapper for optimizer and optional scheduler.

    Encapsulates optimizer and scheduler behavior behind a stable API,
    enabling coordinated stepping, gradient management, and dynamic
    reconfiguration during training.

    This class does not implement optimization algorithms; it delegates
    execution to the underlying `PyTorch` components.
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
        '''
        Initialize the optimization wrapper.

        Args:
            optimizer: A `PyTorch` optimizer instance.
            scheduler: Optional learning rate scheduler instance.
            grad_clip_norm: Optional gradient clipping norm applied
                externally.
            sched_cls: Identifier for the scheduler type (for tracking
                only).
            sched_factory: Factory callable used to reconstruct the
                scheduler.
            sched_args: Arguments used to initialize the scheduler.
        '''

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self._sched_cls = sched_cls
        self._sched_factory = sched_factory
        self._sched_args = sched_args

    # -------------------------- public APIs --------------------------
    @property
    def lrs(self) -> list[float]:
        '''Return current learning rates from all parameter groups.'''

        return [g['lr'] for g in self.optimizer.param_groups]

    def step_optimizer(self) -> None:
        '''Perform a single optimizer update step.'''

        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        '''Reset gradients on all optimized parameters.'''

        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step_scheduler(self) -> None:
        '''Advance the scheduler if configured.'''

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
        Update optimizer learning rate and/or scheduler configuration.

        Args:
            lr:
                New learning rate applied to all parameter groups.
            sched_cls:
                Scheduler identifier used for tracking.
            sched_factory:
                Factory callable to rebuild the scheduler.
            sched_args:
                Scheduler initialization arguments (merged with existing).
            disable_scheduler:
                If True, removes the scheduler entirely.

        Notes:
            - Scheduler updates rebuild the scheduler instance using the
              stored optimizer and merged arguments.
            - Partial updates reuse previously stored scheduler settings.
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
