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
Phase definition
'''

# standard imports
from __future__ import annotations
import dataclasses

@dataclasses.dataclass(frozen=True)
class Phase:
    '''
    Immutable description of a single training phase.

    A Phase defines *what* should be trained and for how long, but does
    not define *how* training progresses or *when* it stops.
    '''
    name: str
    num_epochs: int
    lr_scale: float | None = None   # currently not in use
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None

@dataclasses.dataclass(frozen=True)
class PhaseProfile:
    '''
    Ordered collection of phases forming a training curriculum.

    PhaseProfile objects are resolved from configuration and consumed
    by orchestration logic. They contain no execution semantics.
    '''
    name: str
    phases: list[Phase]
