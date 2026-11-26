# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Training session phase protocol definitions.

Defines the `PhaseLike` structural protocol used across session
orchestration and engine components to describe an executable training
phase without coupling to concrete configuration objects.

Public APIs:
    - `PhaseLike`: Protocol defining a single training phase.
'''

# pylint: disable=missing-function-docstring

# standard imports
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
