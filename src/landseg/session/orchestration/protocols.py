# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

'''
Protocols for orchestration-level engine and configuration contracts.

Defines typing protocol shapes required by runners and policies to
interact with the underlying epoch engine and monitor configuration.

Public APIs:
    - `EpochEngineLike`: protocol for epoch-level engine interface.
    - `EngineBaseLike`: protocol for sub-engine access.
    - `OrchestrationConfigShape`: protocol for orchestration configs.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.core as core

if typing.TYPE_CHECKING:
    import torch.optim
    import landseg.session.contracts as contracts


# ----- typing aliases
Heads: typing.TypeAlias = list[str] | None


# ----- public types
class EpochEngineLike(typing.Protocol):
    @property
    def trainer(self) -> EngineBaseLike | None: ...
    @property
    def evaluator(self) -> EngineBaseLike | None: ...
    @property
    def training_batch_count(self) -> int: ...
    def run_epoch(self, epoch: int) -> core.SessionStepResults: ...
    def set_head_state(self, active_heads: Heads, frozen_heads: Heads) -> None: ...
    def reset_head_state(self) -> None: ...


class EngineBaseLike(typing.Protocol):
    @property
    def model(self) -> core.MultiheadModelLike: ...
    @property
    def state(self) -> _EngineStateLike: ...
    @property
    def optimization(self) -> _OptimizationLike: ...


class OrchestrationConfigShape(typing.Protocol):
    @property
    def monitor(self) -> _Monitor: ...
    @property
    def single_phase(self) -> contracts.PhaseLike: ...
    @property
    def multi_phases(self) -> typing.Sequence[contracts.PhaseLike]: ...


# ----- private types
class _EngineStateLike(typing.Protocol):
    @property
    def progress(self) -> _Progress: ...


class _Progress(typing.Protocol):
    @property
    def epoch(self) -> int: ...
    @property
    def global_step(self) -> int: ...


class _OptimizationLike(typing.Protocol):
    @property
    def optimizer(self) -> 'torch.optim.Optimizer': ...
    @property
    def scheduler(self) -> 'torch.optim.lr_scheduler.LRScheduler | None': ...
    @property
    def lrs(self) -> list[float]: ...
    def reconfigure(
        self,
        *,
        lr: float | None = None,
        sched_cls: str | None = None,
        sched_factory: (
            typing.Callable[..., torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        sched_args: dict[str, typing.Any] | None = None
    ) -> None: ...


class _Monitor(typing.Protocol):
    @property
    def metric_name(self) -> str: ...
    @property
    def track_heads(self) -> dict[str, float] | None: ...
    @property
    def track_mode(self) -> str: ...
    @property
    def allow_early_stop(self) -> bool: ...
    @property
    def patience(self) -> int | None: ...
    @property
    def min_delta(self) -> float | None: ...
