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
Internal runtime configuration structures for the trainer.

Defines typed dataclasses for scheduling, monitoring, precision, and
optimization settings, along with a factory that constructs a unified
RuntimeConfig from user configuration files.
'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------trainer runtime config---------------------------
class ConfigLike(typing.Protocol):
    '''Container holding all trainer runtime configuration sections.'''
    @property
    def heads(self) -> '_Heads': ...
    @property
    def schedule(self) -> '_Schedule': ...
    @property
    def monitor(self) -> '_Monitor': ...
    @property
    def precision(self) -> '_Precision': ...
    @property
    def optimization(self) -> '_OptimConfig': ...
    @property
    def logit_adjust(self) -> '_LogitAdjustConfig': ...

# ------------------------------private dataclass------------------------------
class _Heads(typing.Protocol):
    @property
    def active_heads(self) -> list[str]: ...
    @property
    def frozen_heads(self) -> list[str] | None: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...

class _Schedule(typing.Protocol):
    @property
    def max_epoch(self) -> int: ...
    @property
    def max_step(self) -> int | None: ...
    @property
    def log_every(self) -> int: ...
    @property
    def val_every(self) -> int | None: ...
    @property
    def ckpt_every(self) -> int | None: ...
    @property
    def patience(self) -> int | None: ...
    @property
    def min_delta(self) -> float | None: ...

class _Monitor(typing.Protocol):
    @property
    def metric_name(self) -> str: ...
    @property
    def track_head_name(self) -> str: ...
    @property
    def track_mode(self) -> str: ...

class _Precision(typing.Protocol):
    @property
    def use_amp(self) -> bool: ...

class _OptimConfig(typing.Protocol):
    @property
    def grad_clip_norm(self) -> float | None: ...

class _LogitAdjustConfig(typing.Protocol):
    @property
    def logit_adjust_alpha(self) -> float: ...
    @property
    def enable_train_logit_adjustment(self) -> bool: ...
    @property
    def enable_val_logit_adjustment(self) -> bool: ...
    @property
    def enable_test_logit_adjustment(self) -> bool: ...
