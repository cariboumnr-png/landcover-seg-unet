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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

'''
Engine runner protocol.
'''

# standard imports
from __future__ import annotations
import typing
# local imoprts
import landseg.core as core
import landseg.session.common as common

class EpochEngineLike(typing.Protocol):
    @property
    def mode(self) -> str: ...
    @property
    def trainer(self) -> BatchEngineLike | None: ...
    @property
    def evaluator(self) -> BatchEngineLike | None: ...
    def run_epoch(self, epoch: int) -> EpochMetricsLike: ...
    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
    ) -> None: ...
    def reset_head_state(self) -> None: ...

class BatchEngineLike(typing.Protocol):
    model: core.MultiheadModelLike
    device: str
    state: common.StateLike
    comps: common.ComponentsLike
    def config_logit_adjust(
        self,
        *,
        enable_train_logit_adjust: bool = True,
        enable_val_logit_adjust: bool = True,
        enable_test_logit_adjust: bool = False,
        **kwargs
    ) -> None: ...

@typing.runtime_checkable
class EpochMetricsLike(typing.Protocol):
    @property
    def training(self) -> _TrainerResultsLke | None: ...
    @property
    def validation(self) -> _EvaluatorResultsLke | None: ...

class _TrainerResultsLke(typing.Protocol):
    @property
    def all_heads(self) -> list[str]: ...
    @property
    def current_bidx(self) -> int: ...
    @property
    def total_loss(self) -> float: ...
    @property
    def mean_total_loss(self) -> float: ...
    @property
    def head_losses(self) -> dict[str, float]: ...
    @property
    def mean_head_losses(self) -> dict[str, float]: ...
    def clear(self) -> None: ...

class _EvaluatorResultsLke(typing.Protocol):
    @property
    def all_heads(self) -> list[str]: ...
    @property
    def monitor_heads(self) -> list[str]: ...
    @property
    def head_metrics(self) -> dict[str, common.AccumulatedMetrics]: ...
    @property
    def target_metrics(self) -> float: ...
