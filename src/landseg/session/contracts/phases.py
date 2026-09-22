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

# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

'''
Internal orchestration runtime configuration protocols.

This module defines structured, type-safe protocols used by the
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
clear separation between config structure and orchestration logic.

Public APIs:
    - `PhaseLike`: protocol defining a single training phase.
'''

# standard imports
from __future__ import annotations
import typing


# ----- public types
@typing.runtime_checkable
class PhaseLike(typing.Protocol):
    '''Specification for an executable training session phase.'''
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
