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
    def mode(self) -> typing.Literal['train_eval', 'train_only', 'eval_only']: ...
    @property
    def trainer(self) -> EngineBaseLike | None: ...
    @property
    def evaluator(self) -> EngineBaseLike | None: ...
    def run_epoch(self, epoch: int) -> core.EpochResults: ...
    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
    ) -> None: ...
    def reset_head_state(self) -> None: ...

class EngineBaseLike(typing.Protocol):
    model: core.MultiheadModelLike
    state: _EngineStateLike
    comps: common.ComponentsLike
    device: str

class _EngineStateLike(typing.Protocol):
    progress: _Progress

class _Progress(typing.Protocol):
    epoch: int
    epoch_step: int
    global_step: int
    current_metrics: float
