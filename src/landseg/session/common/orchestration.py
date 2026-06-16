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
Internal orchestration runtime configuration protocols.

This module defines structured, type-safe protocol interfaces used by the
training orchestration layer to coordinate execution behavior across
phases, scheduling, and monitoring.

It formalizes how configuration is accessed at runtime by exposing:
- Scheduling controls (validation, inference, checkpoint cadence)
- Monitoring and early-stopping behavior
- Phase definitions for single-phase and multi-phase execution

The ``PhaseLike`` protocol provides an immutable description of a
training phase, specifying *what* is trained and for how long, while
leaving execution strategy to the orchestration engine.

The ``OrchestrationConfigShape`` protocol acts as a unified contract
for runtime configuration, enabling consistent access across the engine
regardless of how user configs are originally defined or loaded.

These protocols enable strong typing, implementation flexibility, and
clear separation between configuration structure and orchestration logic.
'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------trainer runtime config---------------------------
class OrchestrationConfigShape(typing.Protocol):
    '''Unified access interface for all orchestration config sections.'''
    @property
    def schedule(self) -> _Schedule: ...
    @property
    def monitor(self) -> _Monitor: ...
    @property
    def single_phase(self) -> PhaseLike: ...
    @property
    def multi_phases(self) -> typing.Sequence[PhaseLike]: ...

# ------------------------------private dataclass------------------------------
class _Schedule(typing.Protocol):
    @property
    def val_every_n_epoch(self) -> int: ...
    @property
    def infer_every_n_epoch(self) -> int: ...
    @property
    def ckpt_every_n_epoch(self) -> int: ... # current not in use
    @property
    def update_loss_every_n_batch(self) -> int: ...

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

@typing.runtime_checkable
class PhaseLike(typing.Protocol):
    '''Immutable specification of a single training phase.'''
    @property
    def name(self) -> str: ...
    @property
    def num_epochs(self) -> int: ...
    @property
    def start_epoch(self) -> int: ...
    @property
    def lr_scale(self) -> float | None: ...
    @property
    def active_heads(self) -> list[str] | None: ...
    @property
    def frozen_heads(self) -> list[str] | None: ...
