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
Training phase
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

@dataclasses.dataclass
class Phase:
    '''Training phase container'''
    name: str
    num_epochs: int
    heads: HeadsConfigProto
    logit_adjust: LogitAdjustSchemeProto
    lr_scale: float
    finished: bool = False

    def __str__(self) -> str:
        la = self.logit_adjust
        heads = self.heads
        return '\n'.join([
            f'- Phase Name:\t{self.name}',
            f'- Max Epochs:\t{self.num_epochs}',
            '- Logit Adjustment',
            f'  - Global Alpha:\t{la.logit_adjust_alpha:.2f}',
            f'  - Training Stage:\t{la.enable_train_logit_adjustment}',
            f'  - Validation Stage:\t{la.enable_val_logit_adjustment}',
            f'  - Inference Stage:\t{la.enable_test_logit_adjustment}',
            '- Heads Specs',
            f'  - Active Heads:\t{heads.active_heads}',
            f'  - Frozen Heads:\t{heads.frozen_heads}',
            f'  - Excld. Class:\t{heads.excluded_cls}',
            f'- LR Scale:\t{self.lr_scale}'
        ])

class HeadsConfigProto(typing.Protocol):
    '''Shape of the heads configuration container.'''
    @property
    def active_heads(self) -> list[str]: ...
    @property
    def frozen_heads(self) -> list[str] | None: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...

class LogitAdjustSchemeProto(typing.Protocol):
    '''Shape of the logit adjustment configuration container'''
    @property
    def logit_adjust_alpha(self) -> float: ...
    @property
    def enable_train_logit_adjustment(self) -> bool: ...
    @property
    def enable_val_logit_adjustment(self) -> bool: ...
    @property
    def enable_test_logit_adjustment(self) -> bool: ...
