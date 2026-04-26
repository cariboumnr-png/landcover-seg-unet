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
Strictly typed container to store epoch-level results.
'''

# standard imports
from __future__ import annotations
import dataclasses
# local imports
import landseg.session.common as common

# aliases
field = dataclasses.field

@dataclasses.dataclass
class TrainerEpochResults:
    '''Trainer aggregated results.'''
    heads: list[str]
    current_bidx: int = 0
    total_loss: float = 0.0
    head_losses: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.head_losses = {h: 0.0 for h in self.heads}

    @property
    def mean_total_loss(self) -> float:
        '''Return current moving average total loss.'''
        return self.total_loss / max(1, self.current_bidx)

    @property
    def mean_head_losses(self) -> dict[str, float]:
        '''Return current moving average per head losses.'''
        n = max(1, self.current_bidx)
        return {h: l / n for h, l in self.head_losses.items()}

@dataclasses.dataclass
class EvaluatorEpochResults:
    '''Evaluator aggregated epoch results.'''
    heads: list[str]
    head_metrics: dict[str, common.AccumulatedMetrics] = field(default_factory=dict)

    def __post_init__(self):
        self.head_metrics = {h: common.AccumulatedMetrics for h in self.heads}
