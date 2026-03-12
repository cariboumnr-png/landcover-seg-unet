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
Callback-facing trainer runtime config protocols.
'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------trainer runtime config---------------------------
@typing.runtime_checkable
class RuntimeConfigLike(typing.Protocol):
    schedule: ScheduleLike
    precision: PrecisionLike
    optim: OptimConfigLike
    monitor: MonitorLike

@typing.runtime_checkable
class ScheduleLike(typing.Protocol):
    max_epoch: int
    max_step: int | None
    logging_interval: int
    eval_interval: int | None
    checkpoint_interval: int | None
    patience_epochs: int | None
    min_delta: float | None

@typing.runtime_checkable
class MonitorLike(typing.Protocol):
    enabled: tuple[str, ...]
    metric: str
    head: str
    mode: str

@typing.runtime_checkable
class PrecisionLike(typing.Protocol):
    use_amp: bool

@typing.runtime_checkable
class OptimConfigLike(typing.Protocol):
    grad_clip_norm: float | None
