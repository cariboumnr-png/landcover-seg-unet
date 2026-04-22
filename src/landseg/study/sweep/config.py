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
Protocols for root config dataclass.
'''

# standard imports
from __future__ import annotations
import typing

#
class RootConfigShape(typing.Protocol):
    '''doc'''
    @property
    def pipeline(self) -> _Pipeline: ...
    @property
    def study(self) -> _StudyObjectives: ...
    # methods
    def set_lr(self, lr: float) -> None: ...
    def set_weight_decay(self, weight_decay: float) -> None: ...
    def set_patch_size(self, patch_size: int) -> None: ...
    def set_batch_size(self, batch_size: int) -> None: ...

#
class _Pipeline(typing.Protocol):
    '''doc'''
    @property
    def study_sweep(self) -> _StudySweep: ...

class _StudySweep(typing.Protocol):
    '''doc'''
    @property
    def study_name(self) -> str: ...
    @property
    def objective(self) -> str: ...
    @property
    def storage(self) -> str: ...
    @property
    def direction(self) -> str: ...
    @property
    def n_trials(self) -> int: ...
    @property
    def seed(self) -> int: ...

class _StudyObjectives(typing.Protocol):
    '''doc'''
    @property
    def base(self) -> _BaseObjectives: ...

class _BaseObjectives(typing.Protocol):
    '''doc'''
    @property
    def learning_rate(self) -> tuple[float, float]: ...
    @property
    def weight_decay(self) -> tuple[float, float]: ...
    @property
    def patch_size(self) -> tuple[int, int, int]: ...
    @property
    def batch_size(self) -> tuple[int, int, int]: ...
